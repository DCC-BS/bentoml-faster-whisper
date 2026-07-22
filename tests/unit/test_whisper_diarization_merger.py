import pytest
from pyannote.core import Segment

from bentoml_faster_whisper.services.diarization_service import DiarizationSegment
from bentoml_faster_whisper.utils.whisper_diarization_merger import merge_whisper_diarization


class DummyWord:
    def __init__(self, start, end, word: str = ""):
        self.start = start
        self.end = end
        self.word = word
        self.speaker: str | None = None


class DummyWhisperSegment:
    def __init__(self, start: float, end: float, words: list[DummyWord] | None = None, text: str = ""):
        self.start = start
        self.end = end
        self.words = words
        self.text = text
        self.speaker: str | None = None


def test_merge_whisper_diarization():
    whisper_segments = [
        DummyWhisperSegment(start=0, end=5, words=[]),
        DummyWhisperSegment(start=6, end=10, words=[]),
    ]
    diarization_segments = [
        DiarizationSegment(segment=Segment(0, 5), speaker="A"),
        DiarizationSegment(segment=Segment(6, 10), speaker="B"),
    ]

    result = list(merge_whisper_diarization(whisper_segments, diarization_segments))  # type: ignore

    assert result[0].speaker == "A"
    assert result[1].speaker == "B"


def test_merge_whisper_diarization_with_words():
    whisper_segments = [
        DummyWhisperSegment(start=0, end=5, words=[DummyWord(start=1, end=2), DummyWord(start=3, end=4)]),
        DummyWhisperSegment(
            start=6,
            end=10,
            words=[DummyWord(start=7, end=8), DummyWord(start=9, end=10)],
        ),
    ]
    diarization_segments = [
        DiarizationSegment(segment=Segment(0, 5), speaker="A"),
        DiarizationSegment(segment=Segment(6, 10), speaker="B"),
    ]

    result = list(merge_whisper_diarization(whisper_segments, diarization_segments))  # type: ignore

    assert result[0].speaker == "A"
    assert result[1].speaker == "B"
    assert result[0].words is not None
    assert result[0].words[0].speaker == "A"
    assert result[0].words[1].speaker == "A"
    assert result[1].words is not None
    assert result[1].words[0].speaker == "B"
    assert result[1].words[1].speaker == "B"


def test_diarization_segments_consumed_incrementally():
    """Test that diarization segments are consumed incrementally as needed."""
    whisper_segments = [
        DummyWhisperSegment(start=0, end=5),
        DummyWhisperSegment(start=6, end=10),
        DummyWhisperSegment(start=12, end=15),
    ]

    # Create diarization segments with some gaps
    diarization_segments = [
        DiarizationSegment(segment=Segment(0, 3), speaker="A"),
        DiarizationSegment(segment=Segment(3, 5), speaker="B"),
        DiarizationSegment(segment=Segment(6, 8), speaker="A"),
        DiarizationSegment(segment=Segment(8, 10), speaker="B"),
        DiarizationSegment(segment=Segment(12, 15), speaker="C"),
    ]

    # Use a generator with a counter to track consumption
    consumed_count = 0

    def segment_generator():
        nonlocal consumed_count
        for segment in diarization_segments:
            consumed_count += 1
            yield segment

    # Process the first segment
    list(merge_whisper_diarization([whisper_segments[0]], segment_generator()))  # type: ignore
    # Should have consumed first 2 segments (0-3, 3-5) and peeked at the third
    assert consumed_count == 3

    # Reset counter for the second test
    consumed_count = 0

    # Process first and second whisper segments
    list(merge_whisper_diarization(whisper_segments[:2], segment_generator()))  # type: ignore
    # Should have consumed first 4 segments (0-3, 3-5, 6-8, 8-10) and peeked at the fifth
    assert consumed_count == 5

    # Reset counter for the full test
    consumed_count = 0

    result = list(merge_whisper_diarization(whisper_segments, segment_generator()))
    # Should have consumed all 5 segments
    assert consumed_count == 5

    # Verify the speakers were assigned correctly
    assert result[0].speaker in ("A", "B")  # First segment overlaps with both A and B
    assert result[1].speaker in ("A", "B")  # Second segment overlaps with both A and B
    assert result[2].speaker == "C"  # Third segment overlaps only with C


def test_segment_spanning_speaker_change_is_split():
    """A segment whose words belong to two speakers is cut at the change, one
    speaker per emitted segment — a majority vote would erase the second voice."""
    whisper_segments = [
        DummyWhisperSegment(
            start=3,
            end=7,
            words=[DummyWord(start=3, end=5, word=" hello"), DummyWord(start=5, end=7, word=" there")],
        )
    ]
    diarization_segments = [
        DiarizationSegment(segment=Segment(0, 5), speaker="A"),
        DiarizationSegment(segment=Segment(5, 10), speaker="B"),
    ]

    result = list(merge_whisper_diarization(whisper_segments, diarization_segments))  # type: ignore

    assert [(s.speaker, s.start, s.end, s.text) for s in result] == [
        ("A", 3, 5, " hello"),
        ("B", 5, 7, " there"),
    ]
    assert [s.id for s in result] == [0, 1]
    assert result[0].words is not None and result[0].words[0].speaker == "A"
    assert result[1].words is not None and result[1].words[0].speaker == "B"


def test_fast_exchange_splits_into_alternating_speakers():
    """Fast two-speaker alternation inside one merged decode window (the 95-99s
    teams_konferenz case): every alternation becomes its own segment."""
    whisper_segments = [
        DummyWhisperSegment(
            start=0,
            end=6,
            words=[
                DummyWord(start=0, end=2, word=" oui"),
                DummyWord(start=2, end=4, word=" non"),
                DummyWord(start=4, end=6, word=" si"),
            ],
        )
    ]
    diarization_segments = [
        DiarizationSegment(segment=Segment(0, 2), speaker="A"),
        DiarizationSegment(segment=Segment(2, 4), speaker="B"),
        DiarizationSegment(segment=Segment(4, 6), speaker="A"),
    ]

    result = list(merge_whisper_diarization(whisper_segments, diarization_segments))  # type: ignore

    assert [(s.speaker, s.text) for s in result] == [("A", " oui"), ("B", " non"), ("A", " si")]


def test_border_word_straddling_turn_boundary_does_not_split():
    """A word whose timestamps straddle a turn border (majority of it in neither
    turn) keeps its max-overlap label but must not cut the segment — word
    timestamps jitter ~100-300ms around pyannote borders, and cutting there
    produced spurious one-word segments with the wrong speaker ("Hallo" /
    "zusammen" at 110.8s in teams_konferenz)."""
    whisper_segments = [
        DummyWhisperSegment(
            start=110.3,
            end=111.2,
            words=[
                DummyWord(start=110.33, end=110.77, word=" Hallo"),  # inside A's turn
                DummyWord(start=110.77, end=111.19, word=" zusammen"),  # straddles the A|B border
            ],
        )
    ]
    diarization_segments = [
        DiarizationSegment(segment=Segment(110.3, 110.9), speaker="A"),
        DiarizationSegment(segment=Segment(110.9, 111.4), speaker="B"),
    ]

    result = list(merge_whisper_diarization(whisper_segments, diarization_segments))  # type: ignore

    assert len(result) == 1, f"border jitter must not split: {[(s.speaker, s.text) for s in result]}"
    assert result[0].speaker == "A"  # majority by word duration


def test_words_without_speaker_do_not_split_segment():
    """Unassigned words (in silence between turns of the same speaker) stay with
    the running group instead of cutting the segment."""
    whisper_segments = [
        DummyWhisperSegment(
            start=0,
            end=5,
            words=[
                DummyWord(start=0, end=2, word=" a"),
                DummyWord(start=3.0, end=3.1, word=" mid"),  # >SPEECH_PAD_S from both turns: no speaker
                DummyWord(start=4, end=5, word=" b"),
            ],
        )
    ]
    diarization_segments = [
        DiarizationSegment(segment=Segment(0, 2), speaker="A"),
        DiarizationSegment(segment=Segment(4, 6), speaker="A"),
    ]

    result = list(merge_whisper_diarization(whisper_segments, diarization_segments))  # type: ignore

    assert len(result) == 1
    assert result[0].speaker == "A"
    assert result[0].words is not None and result[0].words[1].speaker is None


def test_segment_in_pre_pad_snaps_to_upcoming_turn():
    """A segment entirely in the pre-pad (before the turn starts) snaps to that turn."""
    # Turn A starts at 1.0; decode window padding places a segment in [0.7, 0.95],
    # which has zero overlap with A but is only 0.05s away.
    whisper_segments = [DummyWhisperSegment(start=0.7, end=0.95, words=[])]
    diarization_segments = [DiarizationSegment(segment=Segment(1, 5), speaker="A")]

    result = list(merge_whisper_diarization(whisper_segments, diarization_segments))  # type: ignore

    assert result[0].speaker == "A"


def test_segment_in_post_pad_snaps_to_previous_turn():
    """A segment entirely in the post-pad (after the turn ends) snaps to that turn."""
    # Turn A ends at 5.0; segment sits in [5.05, 5.25], 0.05s after A, no overlap.
    whisper_segments = [DummyWhisperSegment(start=5.05, end=5.25, words=[])]
    diarization_segments = [DiarizationSegment(segment=Segment(1, 5), speaker="A")]

    result = list(merge_whisper_diarization(whisper_segments, diarization_segments))  # type: ignore

    assert result[0].speaker == "A"


def test_segment_between_two_turns_snaps_to_closer():
    """A segment between two turns, both within tolerance, snaps to the closer one."""
    # A ends at 1.0, B starts at 1.4. Segment [1.1, 1.15]: 0.1s from A, 0.25s from B.
    whisper_segments = [DummyWhisperSegment(start=1.1, end=1.15, words=[])]
    diarization_segments = [
        DiarizationSegment(segment=Segment(0, 1), speaker="A"),
        DiarizationSegment(segment=Segment(1.4, 3), speaker="B"),
    ]

    result = list(merge_whisper_diarization(whisper_segments, diarization_segments))  # type: ignore

    assert result[0].speaker == "A"


def test_segment_far_from_any_turn_stays_none():
    """A segment farther than SPEECH_PAD_S from any turn keeps speaker=None."""
    # Turn A ends at 1.0; segment at [2.0, 2.1] is 1.0s away — beyond the 0.3s pad.
    whisper_segments = [DummyWhisperSegment(start=2.0, end=2.1, words=[])]
    diarization_segments = [DiarizationSegment(segment=Segment(0, 1), speaker="A")]

    result = list(merge_whisper_diarization(whisper_segments, diarization_segments))  # type: ignore

    assert result[0].speaker is None


def test_words_in_pad_regions_snap_to_turn():
    """Words in the pre-pad and post-pad of a turn snap to that turn's speaker."""
    # Turn A covers [1, 5]; decode window is padded to [0.7, 5.3].
    # w0 is pre-pad, w1 overlaps A, w2 is post-pad.
    whisper_segments = [
        DummyWhisperSegment(
            start=0.7,
            end=5.3,
            words=[
                DummyWord(start=0.7, end=0.95),  # pre-pad, 0.05s before A
                DummyWord(start=1.5, end=2.0),  # inside A
                DummyWord(start=5.05, end=5.25),  # post-pad, 0.05s after A
            ],
        )
    ]
    diarization_segments = [DiarizationSegment(segment=Segment(1, 5), speaker="A")]

    result = list(merge_whisper_diarization(whisper_segments, diarization_segments))  # type: ignore

    assert result[0].words is not None
    assert result[0].words[0].speaker == "A"
    assert result[0].words[1].speaker == "A"
    assert result[0].words[2].speaker == "A"
    assert result[0].speaker == "A"


def test_word_between_two_turns_snaps_to_closer():
    """A word with no overlap between two turns snaps to the closer turn."""
    # A covers [0, 1], B covers [1.4, 5]. The middle word [1.1, 1.15] has no overlap:
    # 0.1s from A vs 0.25s from B, so it snaps to A.
    whisper_segments = [
        DummyWhisperSegment(
            start=0,
            end=5,
            words=[
                DummyWord(start=0.2, end=0.8),  # A
                DummyWord(start=1.1, end=1.15),  # gap, closer to A
                DummyWord(start=2.0, end=4.0),  # B
            ],
        )
    ]
    diarization_segments = [
        DiarizationSegment(segment=Segment(0, 1), speaker="A"),
        DiarizationSegment(segment=Segment(1.4, 5), speaker="B"),
    ]

    result = list(merge_whisper_diarization(whisper_segments, diarization_segments))  # type: ignore

    words = [word for seg in result for word in (seg.words or [])]
    assert [word.speaker for word in words] == ["A", "A", "B"]  # middle word snapped to closer turn
    # the A->B change also splits the segment
    assert [seg.speaker for seg in result] == ["A", "B"]


def test_word_far_from_any_turn_stays_none():
    """A word farther than SPEECH_PAD_S from any turn keeps speaker=None."""
    # A covers [0, 1], B covers [4, 5]. The middle word [2.0, 2.1] is >0.3s from both.
    whisper_segments = [
        DummyWhisperSegment(
            start=0,
            end=5,
            words=[
                DummyWord(start=0.2, end=0.8),  # A
                DummyWord(start=2.0, end=2.1),  # silence, far from both
                DummyWord(start=4.2, end=4.8),  # B
            ],
        )
    ]
    diarization_segments = [
        DiarizationSegment(segment=Segment(0, 1), speaker="A"),
        DiarizationSegment(segment=Segment(4, 5), speaker="B"),
    ]

    result = list(merge_whisper_diarization(whisper_segments, diarization_segments))  # type: ignore

    words = [word for seg in result for word in (seg.words or [])]
    assert [word.speaker for word in words] == ["A", None, "B"]  # middle word too far to snap
    # the unassigned word stays with A's group; the split happens at the B word
    assert [seg.speaker for seg in result] == ["A", "B"]


if __name__ == "__main__":
    pytest.main()
