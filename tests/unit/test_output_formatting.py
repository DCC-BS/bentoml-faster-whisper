"""Regression tests for response/subtitle formatting helpers in utils.core and
the verbose_json response builder."""

from bentoml_faster_whisper.models.transcription_verbose_json_response import TranscriptionVerboseJsonResponse
from bentoml_faster_whisper.utils.core import Segment, Word, segments_to_srt, segments_to_vtt


def _segment(start: float, end: float, text: str, words: list[Word] | None) -> Segment:
    return Segment(
        id=0,
        seek=0,
        start=start,
        end=end,
        text=text,
        tokens=[],
        temperature=0.0,
        avg_logprob=-0.1,
        compression_ratio=1.0,
        no_speech_prob=0.0,
        words=words,
    )


class _Info:
    """Minimal stand-in for faster_whisper TranscriptionInfo."""

    language = "de"
    duration = 42.0


def test_vtt_first_cue_keeps_real_start():
    """The first VTT cue must carry the segment's real start, not 0.0. Speech that
    begins at 30s otherwise gets a first cue spanning 30s of silence, and diverges
    from segments_to_srt which uses the real start."""
    seg = _segment(30.0, 34.0, "hello", None)

    vtt = segments_to_vtt(seg, 0)

    assert "00:00:30.000 --> 00:00:34.000" in vtt
    assert "00:00:00.000 -->" not in vtt


def test_vtt_and_srt_first_cue_agree_on_start():
    seg = _segment(12.5, 15.0, "hello", None)

    vtt = segments_to_vtt(seg, 0)
    srt = segments_to_srt(seg, 0)

    assert "00:00:12.500 -->" in vtt
    assert "00:00:12,500 -->" in srt


def test_verbose_json_tolerates_word_less_segment():
    """A decoded segment can arrive with words=None (faster-whisper occasionally
    emits one even with word_timestamps=True, and restore_and_split yields the
    word-less branch). Building the verbose_json response must not raise an
    AssertionError (HTTP 500); the segment simply contributes no words."""
    seg = _segment(0.0, 2.0, "hello", None)

    response = TranscriptionVerboseJsonResponse.from_segments([seg], _Info())  # type: ignore[arg-type]

    assert response.words == []
    assert response.text == "hello"
    assert len(response.segments) == 1


def test_verbose_json_collects_words_when_present():
    words = [Word(start=0.0, end=1.0, word=" hi", probability=0.9)]
    seg = _segment(0.0, 1.0, " hi", words)

    response = TranscriptionVerboseJsonResponse.from_segments([seg], _Info())  # type: ignore[arg-type]

    assert response.words == words
