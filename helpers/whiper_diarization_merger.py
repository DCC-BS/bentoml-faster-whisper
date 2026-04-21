from typing import Any, Iterable, Optional

from core import Segment as WhisperSegment
from diarization_service import DiarizationSegment
from helpers.iter_with_peek import IterWithPeek


def _pack_segements_in_range(
    segments: IterWithPeek[DiarizationSegment],
    start_time: float,
    end_time: float,
) -> Iterable[DiarizationSegment]:
    while segments.has_next() and segments.peek().start <= end_time:
        next_segment = segments.peek()

        # skip the segment if it ends before the window
        if next_segment.end < start_time:
            next(segments)
            continue

        yield next_segment

        # break and don't consume the segment if the segment ends after the window
        if next_segment.end > end_time:
            break

        next(segments)


def _find_best_speaker(
    segments: Iterable[DiarizationSegment],
    start_time: float,
    end_time: float,
) -> Optional[str]:
    best_intersection = 0
    best_speaker = None

    for d in segments:
        intersection = min(d.end, end_time) - max(d.start, start_time)
        if intersection > best_intersection:
            best_intersection = intersection
            best_speaker = d.speaker

    return best_speaker


def _majority_speaker(words: list[Any], word_speakers: list[Optional[str]]) -> Optional[str]:
    """Return the speaker covering the most total word duration."""
    duration: dict[str, float] = {}
    for word, speaker in zip(words, word_speakers):
        if speaker is not None:
            duration[speaker] = duration.get(speaker, 0.0) + (word.end - word.start)
    return max(duration, key=duration.__getitem__) if duration else None


def merge_whipser_diarization(
    whisper_segments: Iterable[WhisperSegment],
    diarization_segments: Iterable[DiarizationSegment],
) -> Iterable[WhisperSegment]:
    diarization_segments_peekable = IterWithPeek(diarization_segments)

    for seg in whisper_segments:
        # Materialise candidates once — safe because _pack_segements_in_range
        # does not consume the last segment from the peekable when it extends
        # beyond the window, so it remains available for the next segment.
        candidates = list(_pack_segements_in_range(diarization_segments_peekable, seg.start, seg.end))

        if seg.words:
            word_candidates = IterWithPeek(iter(candidates))
            word_speakers: list[Optional[str]] = []

            for word in seg.words:
                current_word_candidates = _pack_segements_in_range(word_candidates, word.start, word.end)
                best_word_speaker = _find_best_speaker(current_word_candidates, word.start, word.end)
                word_speakers.append(best_word_speaker)
                if best_word_speaker:
                    word.speaker = best_word_speaker

            # Derive segment speaker from words so segment and words are consistent.
            majority = _majority_speaker(seg.words, word_speakers)
            if majority:
                seg.speaker = majority
        else:
            best_speaker = _find_best_speaker(iter(candidates), seg.start, seg.end)
            if best_speaker:
                seg.speaker = best_speaker

        yield seg
