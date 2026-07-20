import itertools
import math
from typing import Iterable, Protocol

import numpy as np
from faster_whisper.transcribe import restore_speech_timestamps

from core import Segment, Word
from helpers.logger import get_logger
from helpers.utils import clamp, positive_env

logger = get_logger(__name__)

# Whisper models operate on 16 kHz mono audio; speech chunks are expressed in samples at this rate.
WHISPER_SAMPLE_RATE = 16000

# Padding around each speech turn so Whisper doesn't cut into the first/last phoneme,
# and gap size below which neighbouring turns are decoded as one clip (keeps decoding
# context across short pauses and avoids many tiny seek windows).
SPEECH_PAD_S = 0.3
MERGE_GAP_S = 1.0


# Upper bound on the wall-clock span of speech decoded in a single whisper.transcribe()
# call. Continuous speech (radio, panel discussions) collapses into intervals many minutes
# long; handing such a block to one decode triggers Whisper's long-form seek drift, where
# whole 30 s windows are skipped and never emitted. Splitting the turns into runs no longer
# than this — always at a turn boundary, so no word is cut — keeps each decode short enough
# to stay on track. 60 s is ~two 30 s Whisper windows: enough decode context for quality, but
# short enough that drift does not accumulate (measured: drift reappears around ~90 s). Override
# with WHISPER_MAX_DECODE_RUN_S.
MAX_RUN_S = positive_env("WHISPER_MAX_DECODE_RUN_S", 60.0, float)

# A restored word is snapped into its speech chunk, so its midpoint sits within that
# chunk's original-timeline interval; this only absorbs 2-decimal rounding and the
# duration clamp when matching a word back to its interval.
_SPLIT_TOLERANCE_S = 0.1


class _TimedSegment(Protocol):
    @property
    def start(self) -> float: ...

    @property
    def end(self) -> float: ...


def diarization_to_speech_intervals(
    segments: Iterable[_TimedSegment],
    pad_s: float = SPEECH_PAD_S,
    merge_gap_s: float = MERGE_GAP_S,
) -> list[tuple[float, float]]:
    """Collapse (possibly overlapping) diarization speaker turns into sorted,
    non-overlapping (start, end) speech intervals in seconds.

    Returns an empty list when there are no speech turns — callers must fall back
    to another VAD then, because transcribing without any speech region would mean
    decoding the whole file, silence included.
    """
    return pad_and_merge_intervals(((s.start, s.end) for s in segments), pad_s, merge_gap_s)


def pad_and_merge_intervals(
    intervals: Iterable[tuple[float, float]],
    pad_s: float = SPEECH_PAD_S,
    merge_gap_s: float = MERGE_GAP_S,
    lower_bound_s: float = 0.0,
    upper_bound_s: float = math.inf,
) -> list[tuple[float, float]]:
    """Pad each (start, end) interval by ``pad_s``, clamp everything into
    [lower_bound_s, upper_bound_s], and merge overlaps and gaps <= ``merge_gap_s``
    into sorted non-overlapping intervals. Intervals emptied by the clamp are dropped.
    """
    padded = sorted(
        (max(start - pad_s, lower_bound_s), min(end + pad_s, upper_bound_s)) for start, end in intervals if end > start
    )

    merged: list[list[float]] = []
    for start, end in padded:
        if end <= start:
            continue
        if merged and start - merged[-1][1] <= merge_gap_s:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return [(start, end) for start, end in merged]


def speech_intervals_to_chunks(
    intervals: Iterable[tuple[float, float]],
    audio_length_samples: int,
    sampling_rate: int,
) -> list[dict]:
    """Convert second-based speech intervals into faster-whisper speech-chunk dicts
    ({"start": sample, "end": sample}), clamped to the decoded audio length — the
    padding in diarization_to_speech_intervals can push past the end of the file,
    and an out-of-range chunk would skew SpeechTimestampsMap's silence bookkeeping.
    """
    chunks: list[dict] = []
    for start_s, end_s in intervals:
        start = min(int(start_s * sampling_rate), audio_length_samples)
        end = min(int(end_s * sampling_rate), audio_length_samples)
        if end > start:
            chunks.append({"start": start, "end": end})
    return chunks


def collapse_decoded_to_speech(
    decoded: np.ndarray,
    intervals: Iterable[tuple[float, float]],
    sampling_rate: int = WHISPER_SAMPLE_RATE,
) -> tuple[np.ndarray, list[dict]] | None:
    """Cut already-decoded audio down to the given speech intervals — the same
    mechanism faster-whisper applies internally for its silero VAD: silence never
    reaches the decoder, and restore_speech_timestamps() maps the results back onto
    the original timeline afterwards.

    Returns ``(collapsed_audio, speech_chunks)`` or ``None`` when no usable speech
    chunk remains (caller must then fall back to another VAD).
    """
    speech_chunks = speech_intervals_to_chunks(intervals, decoded.shape[0], sampling_rate)
    if not speech_chunks:
        return None

    audio = np.concatenate([decoded[c["start"] : c["end"]] for c in speech_chunks])
    return audio, speech_chunks


def group_intervals_by_language(
    intervals: Iterable[tuple[float, float]],
    languages: Iterable[str],
) -> list[tuple[str, list[tuple[float, float]]]]:
    """Group consecutive speech intervals that detected the same language into
    decode runs: each run is transcribed as one collapsed decode with its language
    pinned, so a multilingual file never forces one language onto all speech."""
    runs: list[tuple[str, list[tuple[float, float]]]] = []
    for interval, language in zip(intervals, languages, strict=True):
        if runs and runs[-1][0] == language:
            runs[-1][1].append(interval)
        else:
            runs.append((language, [interval]))
    return runs


def _max_end(turns: list[tuple[float, float]]) -> float:
    """Furthest point any turn reaches. Not ``turns[-1][1]``: turns are sorted by
    start, so a turn nested inside a longer one sorts *after* it and ends earlier.
    """
    return max(end for _, end in turns)


def _split_turns_by_span(
    turns: list[tuple[float, float]],
    max_run_s: float,
) -> Iterable[list[tuple[float, float]]]:
    """Split a run of turns into sub-runs whose wall-clock span (furthest end minus
    first start) stays within ``max_run_s``, breaking only at turn boundaries. A
    single turn longer than the cap becomes its own sub-run rather than being cut.

    Each cut falls on the *widest* silence gap between consecutive turns, then
    recurses on the two halves. Cutting at the longest pause keeps boundaries on
    natural sentence breaks: a greedy first-fit split can land a boundary in the
    middle of a sentence, where Whisper's decode of the following run drifts and
    drops the opening words. Splitting at the widest pause is stable regardless of
    the exact cap value.
    """
    span = _max_end(turns) - turns[0][0]
    if len(turns) == 1 or span <= max_run_s:
        yield turns
        return

    # Widest gap wins; ties break toward the centre so near-uniform gaps split
    # roughly in half (balanced recursion) instead of peeling one turn at a time.
    # Each gap is measured against the furthest end so far rather than the previous
    # turn's end, so overlapping turns yield a real (non-negative) gap.
    mid = len(turns) / 2
    reach = list(itertools.accumulate((end for _, end in turns), max))
    split_at = max(range(1, len(turns)), key=lambda i: (turns[i][0] - reach[i - 1], -abs(i - mid)))
    yield from _split_turns_by_span(turns[:split_at], max_run_s)
    yield from _split_turns_by_span(turns[split_at:], max_run_s)


def turns_to_language_runs(
    turns: list[tuple[float, float]],
    languages: list[str],
    pad_s: float = SPEECH_PAD_S,
    merge_gap_s: float = MERGE_GAP_S,
    max_run_s: float = MAX_RUN_S,
) -> list[tuple[str, list[tuple[float, float]]]]:
    """Group consecutive same-language turns into decode runs and pad/merge each
    run's turns into speech intervals.

    Runs are additionally capped at ``max_run_s`` of wall-clock span so a long
    stretch of continuous speech is decoded as several bounded clips instead of
    one very long block (which makes Whisper's long-form decode drift and skip
    windows). Splits fall on turn boundaries, so no word is cut.

    Padding is clamped at run boundaries to the midpoint between the adjacent
    turns: neighbouring runs (of different languages, or the same language split
    by the length cap) can be closer than the pad — or even overlap, in crosstalk
    — and without the clamp the same audio would be decoded twice.

    Overlapping turns of the same language are unioned before anything else. pyannote
    routinely emits a short turn nested inside a longer one, and since turns are sorted
    by start the nested one lands *last* in its group; every "last turn ends here"
    assumption below would then place the run boundary in the middle of the longer
    turn and clamp the rest of its speech away, silently deleting it from the decode.
    Unioning restores the sorted-non-overlapping invariant the split and the clamp
    rely on. Speaker labels are untouched by this: runs carry timing only, and
    merge_whipser_diarization attributes words from the original turns.
    """
    grouped = [
        (language, sub_run)
        for language, run_turns in group_intervals_by_language(turns, languages)
        for sub_run in _split_turns_by_span(pad_and_merge_intervals(run_turns, 0.0, 0.0), max_run_s)
    ]

    runs: list[tuple[str, list[tuple[float, float]]]] = []
    for i, (language, run_turns) in enumerate(grouped):
        # Turns of *different* languages can still overlap (crosstalk), so the
        # neighbours' bounds are taken from their furthest reach, not their last turn.
        lower = 0.0
        if i > 0:
            previous_end = _max_end(grouped[i - 1][1])
            lower = max(0.0, (previous_end + run_turns[0][0]) / 2)
        upper = math.inf
        if i + 1 < len(grouped):
            next_start = min(start for start, _ in grouped[i + 1][1])
            upper = (_max_end(run_turns) + next_start) / 2
        intervals = pad_and_merge_intervals(run_turns, pad_s, merge_gap_s, lower, upper)
        if intervals:
            runs.append((language, intervals))
    return runs


def restore_and_split_segments(
    fw_segments: Iterable,
    speech_chunks: list[dict],
    intervals: list[tuple[float, float]],
    original_duration_s: float,
    sampling_rate: int = WHISPER_SAMPLE_RATE,
) -> Iterable[Segment]:
    """Map collapsed-timeline segments back onto the original timeline and repair the
    two artefacts of decoding collapsed speech:

    - clamp every segment/word boundary to [0, original_duration_s], since the model's
      timestamp tokens in the final zero-padded 30s window can overshoot the real end
      (SpeechTimestampsMap adds silence offsets without clamping);
    - split any segment whose words straddle two speech intervals — with the collapsed
      decode a single segment can span the concatenation seam, restoring into a span
      that covers the removed silent gap and mixing two speakers' words.

    Yields repo-owned ``core.Segment`` with sequential ids. Lazy generator.
    """
    restored = restore_speech_timestamps(fw_segments, speech_chunks, sampling_rate)
    core_segments = Segment.from_faster_whisper_segments(restored)

    next_id = 0
    for seg in core_segments:
        seg.start = clamp(seg.start, 0.0, original_duration_s)
        seg.end = clamp(seg.end, 0.0, original_duration_s)

        if not seg.words:
            seg.id = next_id
            next_id += 1
            yield seg
            continue

        for word in seg.words:
            word.start = clamp(word.start, 0.0, original_duration_s)
            word.end = clamp(word.end, 0.0, original_duration_s)

        for piece in _split_segment_by_intervals(seg, intervals):
            piece.id = next_id
            next_id += 1
            yield piece


def _interval_index(intervals: list[tuple[float, float]], midpoint: float) -> int | None:
    for i, (start, end) in enumerate(intervals):
        if start - _SPLIT_TOLERANCE_S <= midpoint <= end + _SPLIT_TOLERANCE_S:
            return i
    return None


def _split_segment_by_intervals(seg: Segment, intervals: list[tuple[float, float]]) -> Iterable[Segment]:
    """Break ``seg`` where consecutive words fall in different speech intervals.

    A word belongs to the interval containing its midpoint; a word matching no
    interval stays with the current group so it never forces a spurious split.
    """
    groups: list[list[Word]] = []
    current: list[Word] = []
    current_idx: int | None = None

    for word in seg.words or []:
        idx = _interval_index(intervals, (word.start + word.end) / 2)
        if idx is None:
            idx = current_idx
        if current and idx != current_idx:
            groups.append(current)
            current = []
        current.append(word)
        current_idx = idx
    if current:
        groups.append(current)

    if len(groups) <= 1:
        yield seg
        return

    # whisper word tokens carry their leading spaces, so join reproduces the text exactly.
    for group in groups:
        yield seg.model_copy(
            update={
                "start": group[0].start,
                "end": group[-1].end,
                "text": "".join(word.word for word in group),
                "words": group,
            }
        )
