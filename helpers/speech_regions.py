from typing import Iterable, Protocol

import numpy as np
from faster_whisper.audio import decode_audio
from faster_whisper.transcribe import restore_speech_timestamps

from core import Segment, Word

# Whisper models operate on 16 kHz mono audio; speech chunks are expressed in samples at this rate.
WHISPER_SAMPLE_RATE = 16000

# Padding around each speech turn so Whisper doesn't cut into the first/last phoneme,
# and gap size below which neighbouring turns are decoded as one clip (keeps decoding
# context across short pauses and avoids many tiny seek windows).
SPEECH_PAD_S = 0.3
MERGE_GAP_S = 1.0

# A restored word is snapped into its speech chunk, so its midpoint sits within that
# chunk's original-timeline interval; this only absorbs 2-decimal rounding and the
# duration clamp when matching a word back to its interval.
_SPLIT_TOLERANCE_S = 0.1


class _TimedSegment(Protocol):
    start: float
    end: float


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
    intervals = sorted((max(s.start - pad_s, 0.0), s.end + pad_s) for s in segments if s.end > s.start)

    merged: list[list[float]] = []
    for start, end in intervals:
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


def collapse_audio_to_speech(
    path: str,
    intervals: Iterable[tuple[float, float]],
    sampling_rate: int = WHISPER_SAMPLE_RATE,
) -> tuple[np.ndarray, list[dict], float] | None:
    """Decode ``path`` and cut it down to the given speech intervals — the same
    mechanism faster-whisper applies internally for its silero VAD: silence never
    reaches the decoder, and restore_speech_timestamps() maps the results back onto
    the original timeline afterwards.

    Returns ``(collapsed_audio, speech_chunks, original_duration_s)`` or ``None`` when
    no usable speech chunk remains (caller must then fall back to another VAD).
    """
    decoded = decode_audio(path, sampling_rate=sampling_rate)
    original_duration_s = decoded.shape[0] / sampling_rate
    speech_chunks = speech_intervals_to_chunks(intervals, decoded.shape[0], sampling_rate)
    if not speech_chunks:
        return None

    audio = np.concatenate([decoded[c["start"] : c["end"]] for c in speech_chunks])
    return audio, speech_chunks, original_duration_s


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
        seg.start = _clamp(seg.start, 0.0, original_duration_s)
        seg.end = _clamp(seg.end, 0.0, original_duration_s)

        if not seg.words:
            seg.id = next_id
            next_id += 1
            yield seg
            continue

        for word in seg.words:
            word.start = _clamp(word.start, 0.0, original_duration_s)
            word.end = _clamp(word.end, 0.0, original_duration_s)

        for piece in _split_segment_by_intervals(seg, intervals):
            piece.id = next_id
            next_id += 1
            yield piece


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


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

    for word in seg.words:
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
