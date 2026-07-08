from typing import Any, Iterable, Optional

from core import Segment as WhisperSegment
from diarization_service import DiarizationSegment
from helpers.iter_with_peek import IterWithPeek
from helpers.speech_regions import SPEECH_PAD_S


class _PeekWithMemory[T]:
    """IterWithPeek that also remembers the last item returned by ``next()``.

    The merge is a single streaming pass over the (sorted) diarization turns, so
    when an item snaps to the *previous* turn we can no longer reach it through
    the iterator — it has already been consumed. Retaining the last consumed turn
    lets the nearest-turn fallback look backwards without buffering the whole stream.
    """

    def __init__(self, it: Iterable[T]):
        self._it = IterWithPeek(it)
        self.last: Optional[T] = None

    def __iter__(self):
        return self

    def __next__(self) -> T:
        self.last = next(self._it)
        return self.last

    def has_next(self) -> bool:
        return self._it.has_next()

    def peek(self) -> T:
        return self._it.peek()


def _pack_segements_in_range(
    segments: _PeekWithMemory[DiarizationSegment],
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


def _nearest_speaker(
    start_time: float,
    end_time: float,
    neighbors: Iterable[Optional[DiarizationSegment]],
    tolerance: float = SPEECH_PAD_S,
) -> Optional[str]:
    """Snap an item with no diarization overlap to the closest turn within ``tolerance``.

    Transcription decode windows are padded by ``SPEECH_PAD_S`` around each speaker
    turn, so a short word/segment can land entirely inside that pad and overlap no
    turn at all. Distance is the gap between the item's ``[start, end]`` and the
    turn's ``[start, end]`` (0 when they touch/overlap). The closest turn wins, but
    only if it is within ``tolerance`` — beyond that we keep ``None`` rather than
    inventing a speaker across genuine silence.
    """
    best_distance: Optional[float] = None
    best_speaker: Optional[str] = None

    for turn in neighbors:
        if turn is None:
            continue

        if start_time > turn.end:
            distance = start_time - turn.end
        elif turn.start > end_time:
            distance = turn.start - end_time
        else:
            distance = 0.0

        if distance > tolerance:
            continue

        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_speaker = turn.speaker

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
    diarization_segments_peekable = _PeekWithMemory(diarization_segments)

    for seg in whisper_segments:
        # Materialise candidates once — safe because _pack_segements_in_range
        # does not consume the last segment from the peekable when it extends
        # beyond the window, so it remains available for the next segment.
        candidates = list(_pack_segements_in_range(diarization_segments_peekable, seg.start, seg.end))

        # Turns bordering this window, for the nearest-turn fallback: the last turn
        # consumed just before the window, and the next unconsumed turn after it.
        seg_prev = diarization_segments_peekable.last
        seg_next = diarization_segments_peekable.peek() if diarization_segments_peekable.has_next() else None

        if seg.words:
            word_candidates = _PeekWithMemory(iter(candidates))
            word_speakers: list[Optional[str]] = []

            for word in seg.words:
                current_word_candidates = _pack_segements_in_range(word_candidates, word.start, word.end)
                best_word_speaker = _find_best_speaker(current_word_candidates, word.start, word.end)

                if best_word_speaker is None:
                    # No overlap: snap to the nearest turn within SPEECH_PAD_S. Candidates
                    # bordering the word live at the word-stream head / last consumed item,
                    # while turns just outside the segment window come from seg_prev/seg_next.
                    word_next = word_candidates.peek() if word_candidates.has_next() else None
                    best_word_speaker = _nearest_speaker(
                        word.start,
                        word.end,
                        (word_candidates.last, word_next, seg_prev, seg_next),
                    )

                word_speakers.append(best_word_speaker)
                if best_word_speaker:
                    word.speaker = best_word_speaker

            # Derive segment speaker from words so segment and words are consistent.
            majority = _majority_speaker(seg.words, word_speakers)
            if majority:
                seg.speaker = majority
        else:
            best_speaker = _find_best_speaker(iter(candidates), seg.start, seg.end)
            if best_speaker is None:
                best_speaker = _nearest_speaker(seg.start, seg.end, (*candidates, seg_prev, seg_next))
            if best_speaker:
                seg.speaker = best_speaker

        yield seg
