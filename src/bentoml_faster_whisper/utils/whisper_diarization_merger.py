import copy
from typing import Iterable, Optional

from bentoml_faster_whisper.services.diarization_service import DiarizationSegment
from bentoml_faster_whisper.utils.core import Segment as WhisperSegment
from bentoml_faster_whisper.utils.iter_with_peek import IterWithPeek
from bentoml_faster_whisper.utils.speech_regions import SPEECH_PAD_S


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
) -> tuple[Optional[str], float]:
    """Speaker of the turn with the largest overlap, plus that overlap in seconds."""
    best_intersection = 0.0
    best_speaker = None

    for d in segments:
        intersection = min(d.end, end_time) - max(d.start, start_time)
        if intersection > best_intersection:
            best_intersection = intersection
            best_speaker = d.speaker

    return best_speaker, best_intersection


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


def _majority_speaker(words: list, word_speakers: list[Optional[str]]) -> Optional[str]:
    """Return the speaker covering the most total word duration."""
    duration: dict[str, float] = {}
    for word, speaker in zip(words, word_speakers):
        if speaker is not None:
            duration[speaker] = duration.get(speaker, 0.0) + (word.end - word.start)
    return max(duration, key=duration.__getitem__) if duration else None


# A word only justifies cutting a segment when at least this fraction of its
# duration lies inside its assigned diarization turn. Word timestamps jitter by
# ~100-300ms around pyannote's turn borders, so a border word can flip to the
# neighbouring turn by raw max-overlap; an unconfident word keeps its speaker
# label but never starts a new segment.
_SPLIT_MIN_OVERLAP_FRACTION = 0.5

# A split must not produce a piece shorter than this. Where pyannote emits
# 0.1-0.2s turn fragments (crosstalk, simultaneous greetings), word-level
# speaker changes are noise: a sub-threshold group folds back into its
# neighbour instead of becoming its own one-word segment.
_MIN_SPLIT_PIECE_S = 0.5


def _split_segment_by_speaker(
    seg: WhisperSegment,
    split_speakers: list[Optional[str]],
    word_speakers: list[Optional[str]],
) -> Iterable[WhisperSegment]:
    """Split ``seg`` where consecutive words were confidently assigned different
    speakers.

    Whisper's own segmentation knows nothing about speakers: in a fast exchange
    whose turns were merged into one decode window, a single segment can span
    several pyannote turns. Flattening it to one majority speaker would erase
    the alternation, so the segment is cut at word-level speaker changes — but
    only at credible ones: ``split_speakers`` carries a speaker only for words
    solidly inside their turn (``_SPLIT_MIN_OVERLAP_FRACTION``), border and
    padding words never force a split, and a piece shorter than
    ``_MIN_SPLIT_PIECE_S`` folds back into its neighbour.
    """
    groups: list[tuple[Optional[str], list]] = []
    for word, speaker in zip(seg.words or [], split_speakers):
        if groups and (speaker is None or groups[-1][0] is None or speaker == groups[-1][0]):
            group_speaker, group_words = groups[-1]
            group_words.append(word)
            if group_speaker is None:
                groups[-1] = (speaker, group_words)
        else:
            groups.append((speaker, [word]))

    def duration(words: list) -> float:
        return words[-1].end - words[0].start

    # Fold sub-threshold pieces into their neighbour (previous when possible),
    # then re-merge neighbours that ended up with the same speaker.
    index = 0
    while len(groups) > 1 and index < len(groups):
        if duration(groups[index][1]) >= _MIN_SPLIT_PIECE_S:
            index += 1
            continue
        if index > 0:
            groups[index - 1][1].extend(groups[index][1])
            del groups[index]
            index = max(index - 1, 0)
        else:
            speaker = groups[1][0]
            groups[1] = (speaker, groups[0][1] + groups[1][1])
            del groups[0]
    merged: list[tuple[Optional[str], list]] = []
    for speaker, words in groups:
        if merged and merged[-1][0] == speaker:
            merged[-1][1].extend(words)
        else:
            merged.append((speaker, words))
    groups = merged

    if len(groups) <= 1:
        # No credible speaker change: keep the segment whole and label it like
        # the pre-split implementation did, by word-duration majority.
        majority = _majority_speaker(seg.words or [], word_speakers)
        if majority:
            seg.speaker = majority
        yield seg
        return

    for speaker, words in groups:
        # whisper word tokens carry their leading spaces, so join reproduces the text exactly.
        piece = copy.copy(seg)
        piece.start = words[0].start
        piece.end = words[-1].end
        piece.text = "".join(word.word for word in words)
        piece.words = words
        piece.speaker = speaker
        yield piece


def merge_whipser_diarization(
    whisper_segments: Iterable[WhisperSegment],
    diarization_segments: Iterable[DiarizationSegment],
) -> Iterable[WhisperSegment]:
    diarization_segments_peekable = _PeekWithMemory(diarization_segments)
    next_id = 0

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
            split_speakers: list[Optional[str]] = []

            for word in seg.words:
                current_word_candidates = _pack_segements_in_range(word_candidates, word.start, word.end)
                best_word_speaker, overlap = _find_best_speaker(current_word_candidates, word.start, word.end)
                # Only a word solidly inside its turn may later cut the segment;
                # border words keep their label but never define a boundary.
                confident = overlap >= _SPLIT_MIN_OVERLAP_FRACTION * (word.end - word.start)

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
                split_speakers.append(best_word_speaker if confident else None)
                if best_word_speaker:
                    word.speaker = best_word_speaker

            # Cut the segment at confident speaker changes so a fast exchange decoded
            # as one window keeps its alternation; segment and word speakers stay consistent.
            for piece in _split_segment_by_speaker(seg, split_speakers, word_speakers):
                piece.id = next_id
                next_id += 1
                yield piece
        else:
            best_speaker, _ = _find_best_speaker(iter(candidates), seg.start, seg.end)
            if best_speaker is None:
                best_speaker = _nearest_speaker(seg.start, seg.end, (*candidates, seg_prev, seg_next))
            if best_speaker:
                seg.speaker = best_speaker

            seg.id = next_id
            next_id += 1
            yield seg
