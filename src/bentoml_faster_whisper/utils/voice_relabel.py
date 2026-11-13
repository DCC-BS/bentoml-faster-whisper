"""Relabel whole segments whose voice matches another speaker of the same recording.

Pyannote's local segmentation misses fast turn changes between similar voices, so a
speaker's opening sentence is often attributed to the previous speaker. Each speaker's
voice profile is the mean embedding of their longest crosstalk-free turns in the same
file; a segment moves to another speaker only when that speaker's profile is clearly
closer. See docs/technical_architecture.md ("Voice Relabelling").
"""

from collections.abc import Callable, Iterable, Sequence
from typing import Protocol

import numpy as np

from bentoml_faster_whisper.utils.core import Segment
from bentoml_faster_whisper.utils.logger import get_logger

logger = get_logger(__name__)

Embed = Callable[[float, float], np.ndarray]

PROFILE_MIN_TURN_S = 4.0
PROFILE_MAX_TURNS = 25
PROFILE_TRIM_S = 0.3
MIN_SEGMENT_S = 0.6
MIN_MARGIN = 0.05


class _Turn(Protocol):
    start: float
    end: float
    speaker: str


def _unit(vector: np.ndarray) -> np.ndarray | None:
    vector = np.asarray(vector, dtype=np.float64).ravel()
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm == 0:
        return None
    return vector / norm


def _has_crosstalk(turn: _Turn, turns: Sequence[_Turn]) -> bool:
    return any(other.speaker != turn.speaker and other.start < turn.end and other.end > turn.start for other in turns)


def speaker_profiles(turns: Sequence[_Turn], embed: Embed) -> dict[str, np.ndarray]:
    """Mean voice embedding per speaker, from their longest crosstalk-free turns."""
    profiles: dict[str, np.ndarray] = {}
    for speaker in sorted({turn.speaker for turn in turns}):
        clean = [
            turn
            for turn in turns
            if turn.speaker == speaker
            and turn.end - turn.start >= PROFILE_MIN_TURN_S
            and not _has_crosstalk(turn, turns)
        ]
        clean.sort(key=lambda turn: turn.start - turn.end)
        vectors = [
            vector
            for turn in clean[:PROFILE_MAX_TURNS]
            if (vector := _unit(embed(turn.start + PROFILE_TRIM_S, turn.end - PROFILE_TRIM_S))) is not None
        ]
        if vectors and (profile := _unit(np.mean(vectors, axis=0))) is not None:
            profiles[speaker] = profile
    return profiles


def relabel_segments(
    segments: Iterable[Segment],
    profiles: dict[str, np.ndarray],
    embed: Embed,
) -> Iterable[Segment]:
    if len(profiles) < 2:
        yield from segments
        return

    relabelled = 0
    for segment in segments:
        current = segment.speaker
        if current in profiles and segment.end - segment.start >= MIN_SEGMENT_S:
            voice = _unit(embed(segment.start, segment.end))
            if voice is not None:
                similarity = {speaker: float(voice @ profile) for speaker, profile in profiles.items()}
                best = max(similarity, key=similarity.__getitem__)
                if best != current and similarity[best] - similarity[current] > MIN_MARGIN:
                    segment.speaker = best
                    for word in segment.words or []:
                        word.speaker = best
                    relabelled += 1
        yield segment

    logger.debug("Voice relabelling finished", relabelled_segments=relabelled, speakers=len(profiles))
