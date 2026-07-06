from typing import Iterable, Protocol

# Padding around each speech turn so Whisper doesn't cut into the first/last phoneme,
# and gap size below which neighbouring turns are decoded as one clip (keeps decoding
# context across short pauses and avoids many tiny seek windows).
SPEECH_PAD_S = 0.3
MERGE_GAP_S = 1.0


class _TimedSegment(Protocol):
    start: float
    end: float


def diarization_to_clip_timestamps(
    segments: Iterable[_TimedSegment],
    pad_s: float = SPEECH_PAD_S,
    merge_gap_s: float = MERGE_GAP_S,
) -> list[float]:
    """Collapse (possibly overlapping) diarization speaker turns into a flat
    [start1, end1, start2, end2, ...] list for faster-whisper's ``clip_timestamps``.

    Returns an empty list when there are no speech turns — callers must fall back
    to another VAD then, because faster-whisper treats an empty clip list as
    "decode the whole file".
    """
    intervals = sorted((max(s.start - pad_s, 0.0), s.end + pad_s) for s in segments if s.end > s.start)

    merged: list[list[float]] = []
    for start, end in intervals:
        if merged and start - merged[-1][1] <= merge_gap_s:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return [timestamp for interval in merged for timestamp in interval]
