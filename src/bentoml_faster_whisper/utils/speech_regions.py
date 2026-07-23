import itertools
import math
from typing import Iterable, Protocol

import numpy as np
from faster_whisper.transcribe import restore_speech_timestamps

from bentoml_faster_whisper.utils.core import Segment, Word, clamp, positive_env
from bentoml_faster_whisper.utils.logger import get_logger

logger = get_logger(__name__)

WHISPER_SAMPLE_RATE = 16000
SPEECH_PAD_S = 0.3
MERGE_GAP_S = 1.0

MAX_RUN_S = positive_env("WHISPER_MAX_DECODE_RUN_S", 60.0, float)
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
    """
    return pad_and_merge_intervals(((s.start, s.end) for s in segments), pad_s, merge_gap_s)


def pad_and_merge_intervals(
    intervals: Iterable[tuple[float, float]],
    pad_s: float = SPEECH_PAD_S,
    merge_gap_s: float = MERGE_GAP_S,
    lower_bound_s: float = 0.0,
    upper_bound_s: float = math.inf,
) -> list[tuple[float, float]]:
    """Pad each (start, end) interval by ``pad_s``, clamp to bounds, and merge close gaps."""
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
    """Convert second-based speech intervals into faster-whisper speech-chunk dicts."""
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
    """Cut decoded audio down to speech intervals."""
    speech_chunks = speech_intervals_to_chunks(intervals, decoded.shape[0], sampling_rate)
    if not speech_chunks:
        return None

    audio = np.concatenate([decoded[c["start"] : c["end"]] for c in speech_chunks])
    return audio, speech_chunks


def group_intervals_by_language(
    intervals: Iterable[tuple[float, float]],
    languages: Iterable[str],
) -> list[tuple[str, list[tuple[float, float]]]]:
    """Group consecutive speech intervals that detected the same language into decode runs."""
    runs: list[tuple[str, list[tuple[float, float]]]] = []
    for interval, language in zip(intervals, languages, strict=True):
        if runs and runs[-1][0] == language:
            runs[-1][1].append(interval)
        else:
            runs.append((language, [interval]))
    return runs


def _max_end(turns: list[tuple[float, float]]) -> float:
    """Return furthest end timestamp across all turns."""
    return max(end for _, end in turns)


def _split_turns_by_span(
    turns: list[tuple[float, float]],
    max_run_s: float,
) -> Iterable[list[tuple[float, float]]]:
    """Split a run of turns into sub-runs whose wall-clock span stays within ``max_run_s``."""
    span = _max_end(turns) - turns[0][0]
    if len(turns) == 1 or span <= max_run_s:
        yield turns
        return

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
    """Group consecutive same-language turns into decode runs capped at max_run_s."""
    grouped = [
        (language, sub_run)
        for language, run_turns in group_intervals_by_language(turns, languages)
        for sub_run in _split_turns_by_span(pad_and_merge_intervals(run_turns, 0.0, 0.0), max_run_s)
    ]

    runs: list[tuple[str, list[tuple[float, float]]]] = []
    for i, (language, run_turns) in enumerate(grouped):
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
    """Map collapsed-timeline segments back onto original file timeline."""
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
    """Break segment where consecutive words fall in different speech intervals."""
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

    for group in groups:
        yield seg.model_copy(
            update={
                "start": group[0].start,
                "end": group[-1].end,
                "text": "".join(word.word for word in group),
                "words": group,
            }
        )
