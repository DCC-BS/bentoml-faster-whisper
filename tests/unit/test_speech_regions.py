from dataclasses import dataclass

import numpy as np
from faster_whisper.transcribe import Segment as FWSegment
from faster_whisper.transcribe import Word as FWWord

import helpers.speech_regions as sr
from helpers.speech_regions import (
    collapse_audio_to_speech,
    diarization_to_speech_intervals,
    group_intervals_by_language,
    restore_and_split_segments,
    speech_intervals_to_chunks,
    turns_to_language_runs,
)


@dataclass
class _Turn:
    start: float
    end: float


def _fw_word(start: float, end: float, word: str) -> FWWord:
    return FWWord(start=start, end=end, word=word, probability=1.0)


def _fw_segment(start, end, words, seg_id=0) -> FWSegment:
    return FWSegment(
        id=seg_id,
        seek=0,
        start=start,
        end=end,
        text="".join(w.word for w in words) if words else "silence",
        tokens=[],
        avg_logprob=0.0,
        compression_ratio=1.0,
        no_speech_prob=0.0,
        words=words,
        temperature=0.0,
    )


def test_empty_input_returns_empty_list():
    assert diarization_to_speech_intervals([]) == []


def test_single_turn_is_padded_and_clamped_at_zero():
    intervals = diarization_to_speech_intervals([_Turn(0.1, 5.0)], pad_s=0.3, merge_gap_s=1.0)

    assert intervals == [(0.0, 5.3)]


def test_overlapping_turns_are_merged():
    intervals = diarization_to_speech_intervals(
        [_Turn(1.0, 4.0), _Turn(3.0, 6.0)],
        pad_s=0.0,
        merge_gap_s=0.0,
    )

    assert intervals == [(1.0, 6.0)]


def test_turns_within_merge_gap_are_merged():
    intervals = diarization_to_speech_intervals(
        [_Turn(1.0, 2.0), _Turn(2.8, 4.0)],
        pad_s=0.0,
        merge_gap_s=1.0,
    )

    assert intervals == [(1.0, 4.0)]


def test_turns_beyond_merge_gap_stay_separate():
    intervals = diarization_to_speech_intervals(
        [_Turn(1.0, 2.0), _Turn(10.0, 12.0)],
        pad_s=0.0,
        merge_gap_s=1.0,
    )

    assert intervals == [(1.0, 2.0), (10.0, 12.0)]


def test_unsorted_input_is_sorted():
    intervals = diarization_to_speech_intervals(
        [_Turn(10.0, 12.0), _Turn(1.0, 2.0)],
        pad_s=0.0,
        merge_gap_s=1.0,
    )

    assert intervals == [(1.0, 2.0), (10.0, 12.0)]


def test_zero_or_negative_length_turns_are_dropped():
    intervals = diarization_to_speech_intervals(
        [_Turn(1.0, 1.0), _Turn(5.0, 4.0)],
        pad_s=0.3,
        merge_gap_s=1.0,
    )

    assert intervals == []


def test_intervals_convert_to_sample_chunks():
    chunks = speech_intervals_to_chunks([(0.0, 1.0), (2.5, 3.0)], audio_length_samples=160000, sampling_rate=16000)

    assert chunks == [
        {"start": 0, "end": 16000},
        {"start": 40000, "end": 48000},
    ]


def test_chunks_are_clamped_to_audio_length():
    # Padding can push the last interval past the end of the file.
    chunks = speech_intervals_to_chunks([(9.5, 10.3)], audio_length_samples=160000, sampling_rate=16000)

    assert chunks == [{"start": 152000, "end": 160000}]


def test_chunks_entirely_past_audio_end_are_dropped():
    chunks = speech_intervals_to_chunks([(11.0, 12.0)], audio_length_samples=160000, sampling_rate=16000)

    assert chunks == []


def test_collapse_audio_to_speech_concatenates_only_speech(monkeypatch):
    decoded = np.arange(160000, dtype=np.float32)  # 10s at 16 kHz
    monkeypatch.setattr(sr, "decode_audio", lambda path, sampling_rate: decoded)

    collapsed = collapse_audio_to_speech("x.wav", [(0.0, 1.0), (5.0, 6.0)], 16000)
    assert collapsed is not None
    audio, chunks, duration = collapsed

    assert duration == 10.0
    assert chunks == [{"start": 0, "end": 16000}, {"start": 80000, "end": 96000}]
    assert audio.shape[0] == 32000
    np.testing.assert_array_equal(audio[:16000], decoded[0:16000])
    np.testing.assert_array_equal(audio[16000:], decoded[80000:96000])


def test_collapse_audio_to_speech_returns_none_without_usable_chunks(monkeypatch):
    decoded = np.arange(16000, dtype=np.float32)  # 1s — the interval is entirely past the end
    monkeypatch.setattr(sr, "decode_audio", lambda path, sampling_rate: decoded)

    assert collapse_audio_to_speech("x.wav", [(5.0, 6.0)], 16000) is None


def test_restore_splits_segment_straddling_the_seam():
    # Two speech regions, (0,2) and (10,12), collapsed back-to-back; 8s of silence removed.
    speech_chunks = [{"start": 0, "end": 32000}, {"start": 160000, "end": 192000}]
    intervals = [(0.0, 2.0), (10.0, 12.0)]

    left = _fw_word(0.5, 1.0, " hello")
    right = _fw_word(3.0, 3.5, " world")  # collapsed time in the second chunk
    seg = _fw_segment(0.5, 3.5, [left, right])

    out = list(restore_and_split_segments([seg], speech_chunks, intervals, 12.0, 16000))

    assert len(out) == 2, "a seam-straddling segment must split into one piece per speech region"
    assert (out[0].start, out[0].end, out[0].text) == (0.5, 1.0, " hello")
    assert (out[1].start, out[1].end, out[1].text) == (11.0, 11.5, " world")
    assert [s.id for s in out] == [0, 1]
    # Every piece lies inside a single speech interval — never spanning the removed gap.
    for piece, (lo, hi) in zip(out, intervals):
        assert lo - 0.1 <= piece.start and piece.end <= hi + 0.1


def test_restore_keeps_single_region_segment_intact():
    speech_chunks = [{"start": 0, "end": 32000}]
    intervals = [(0.0, 2.0)]

    words = [_fw_word(0.2, 0.5, " a"), _fw_word(0.6, 0.9, " b")]
    seg = _fw_segment(0.2, 0.9, words)

    out = list(restore_and_split_segments([seg], speech_chunks, intervals, 2.0, 16000))

    assert len(out) == 1
    assert out[0].words is not None
    assert [w.word for w in out[0].words] == [" a", " b"]
    assert out[0].id == 0


def test_restore_clamps_segment_end_to_duration():
    speech_chunks = [{"start": 0, "end": 32000}]  # interval (0, 2) but the file is only 1.5s
    intervals = [(0.0, 2.0)]

    seg = _fw_segment(0.0, 2.0, None)

    out = list(restore_and_split_segments([seg], speech_chunks, intervals, 1.5, 16000))

    assert len(out) == 1
    assert out[0].start == 0.0
    assert out[0].end == 1.5


def test_restore_clamps_word_boundaries_to_duration():
    speech_chunks = [{"start": 0, "end": 32000}]
    intervals = [(0.0, 2.0)]

    seg = _fw_segment(1.4, 2.0, [_fw_word(1.4, 2.0, " tail")])

    out = list(restore_and_split_segments([seg], speech_chunks, intervals, 1.5, 16000))

    assert len(out) == 1
    assert out[0].words is not None
    assert out[0].words[0].end == 1.5
    assert out[0].end == 1.5


def test_group_intervals_by_language_groups_consecutive_runs():
    intervals = [(0.0, 5.0), (6.0, 10.0), (12.0, 20.0), (21.0, 25.0)]
    languages = ["de", "de", "en", "de"]

    assert group_intervals_by_language(intervals, languages) == [
        ("de", [(0.0, 5.0), (6.0, 10.0)]),
        ("en", [(12.0, 20.0)]),
        ("de", [(21.0, 25.0)]),
    ]


def test_group_intervals_by_language_single_language_is_one_run():
    intervals = [(0.0, 5.0), (6.0, 10.0)]

    assert group_intervals_by_language(intervals, ["de", "de"]) == [("de", [(0.0, 5.0), (6.0, 10.0)])]


def test_long_single_language_run_is_capped_into_bounded_runs():
    # One continuous stretch of same-language speech that would collapse into a single
    # very long decode block must be split into several runs no longer than max_run_s,
    # so Whisper never decodes an over-long block in one go.
    turns = [(float(i), float(i) + 1.0) for i in range(0, 300, 10)]  # 30 turns spanning 0..291
    languages = ["de"] * len(turns)

    runs = turns_to_language_runs(turns, languages, max_run_s=100.0)

    assert len(runs) >= 3
    assert all(language == "de" for language, _ in runs)
    for _, intervals in runs:
        span = intervals[-1][1] - intervals[0][0]
        assert span <= 100.0 + 2 * sr.SPEECH_PAD_S


def test_single_short_run_is_not_split():
    turns = [(0.0, 5.0), (6.0, 10.0)]

    runs = turns_to_language_runs(turns, ["de", "de"], max_run_s=100.0)

    assert len(runs) == 1
    assert runs[0][0] == "de"


def test_turn_longer_than_cap_becomes_its_own_run():
    # A single turn exceeding the cap must not be cut mid-turn; it stays whole.
    turns = [(0.0, 200.0), (201.0, 205.0)]

    runs = turns_to_language_runs(turns, ["de", "de"], max_run_s=100.0)

    assert len(runs) == 2


def _covers(runs, start: float, end: float) -> bool:
    """Is [start, end] fully inside the union of every run's intervals?"""
    intervals = sorted(interval for _, run_intervals in runs for interval in run_intervals)
    covered = start
    for interval_start, interval_end in intervals:
        if interval_start > covered:
            break
        covered = max(covered, interval_end)
    return covered >= end


def test_turn_nested_in_a_longer_turn_does_not_truncate_it():
    """pyannote sorts turns by start, so a short turn nested inside a long one comes
    *after* it. The long turn's speech must still be decoded in full — the nested
    turn's early end must not become the run's boundary.
    """
    turns = [(100.0, 140.0), (101.0, 101.5), (145.0, 150.0)]

    runs = turns_to_language_runs(turns, ["de"] * len(turns), max_run_s=30.0)

    assert _covers(runs, 100.0, 140.0), f"speech inside the long turn was clamped away: {runs}"


def test_overlapping_turns_keep_full_speech_across_a_run_split():
    """Same nesting, but the run cap forces a split. Whatever the split, every second
    of speech has to land in some run: audio dropped here never reaches the decoder.
    """
    turns = [(0.0, 30.0), (31.0, 90.0), (32.0, 32.4), (95.0, 120.0)]

    runs = turns_to_language_runs(turns, ["de"] * len(turns), max_run_s=60.0)

    for start, end in turns:
        assert _covers(runs, start, end), f"turn {start}-{end} not fully covered by any run: {runs}"


def test_nested_turn_does_not_inflate_run_count():
    """The nested turn is inside the long turn's speech, so it must not be emitted as
    a separate micro-run — that would decode <1 s of audio in isolation.
    """
    turns = [(100.0, 140.0), (101.0, 101.5), (145.0, 150.0)]

    runs = turns_to_language_runs(turns, ["de"] * len(turns), max_run_s=30.0)

    assert not any(intervals[-1][1] - intervals[0][0] < 2.0 for _, intervals in runs), (
        f"a sub-second nested turn became its own micro-run: {runs}"
    )
