import itertools
from typing import Iterable, Optional

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
    """
    Find the best matching speaker for a time segment.

    Returns:
        Tuple of (best_speaker, new_window_start)
    """
    best_intersection = 0
    best_speaker = None

    for d in segments:
        intersection = min(d.end, end_time) - max(d.start, start_time)
        if intersection > best_intersection:
            best_intersection = intersection
            best_speaker = d.speaker

    return best_speaker


def merge_whipser_diarization(
    whisper_segments: Iterable[WhisperSegment],
    diarization_segments: Iterable[DiarizationSegment],
) -> Iterable[WhisperSegment]:
    diarization_segments_peekable = IterWithPeek(diarization_segments)

    for seg in whisper_segments:
        candidates = _pack_segements_in_range(
            diarization_segments_peekable, seg.start, seg.end
        )

        # Create a copy of the candidates for the words
        seg_candidates, word_candidates_it = itertools.tee(candidates)
        word_candidates = IterWithPeek(word_candidates_it)

        # Find best speaker for segment
        best_speaker = _find_best_speaker(seg_candidates, seg.start, seg.end)

        # Assign speaker
        if best_speaker:
            seg.speaker = best_speaker

        # Process words with the same approach
        if seg.words is not None:  # Start from the same window position
            for word in seg.words:
                current_word_candidates = _pack_segements_in_range(
                    word_candidates, word.start, word.end
                )
                best_word_speaker = _find_best_speaker(
                    current_word_candidates,
                    word.start,
                    word.end,
                )

                # Assign speaker
                if best_word_speaker:
                    word.speaker = best_word_speaker

        yield seg
