import pytest
from pyannote.core import Segment

from diarization_service import DiarizationSegment
from whiper_diarization_merger import merge_whipser_diarization


class DummyWord:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.speaker: str | None = None


class DummyWhisperSegment:
    def __init__(self, start: float, end: float, words: list[DummyWord] | None = None):
        self.start = start
        self.end = end
        self.words = words
        self.speaker: str | None = None


def test_merge_whipser_diarization():
    whisper_segments = [
        DummyWhisperSegment(start=0, end=5, words=[]),
        DummyWhisperSegment(start=6, end=10, words=[]),
    ]
    diarization_segments = [
        DiarizationSegment(segment=Segment(0, 5), speaker="A", label="A"),
        DiarizationSegment(segment=Segment(6, 10), speaker="B", label="B"),
    ]

    result = list(merge_whipser_diarization(whisper_segments, diarization_segments))

    assert result[0].speaker == "A"
    assert result[1].speaker == "B"


def test_merge_whipser_diarization_with_words():
    whisper_segments = [
        DummyWhisperSegment(
            start=0, end=5, words=[DummyWord(start=1, end=2), DummyWord(start=3, end=4)]
        ),
        DummyWhisperSegment(
            start=6,
            end=10,
            words=[DummyWord(start=7, end=8), DummyWord(start=9, end=10)],
        ),
    ]
    diarization_segments = [
        DiarizationSegment(segment=Segment(0, 5), label="label1", speaker="A"),
        DiarizationSegment(segment=Segment(6, 10), label="label2", speaker="B"),
    ]

    result = list(merge_whipser_diarization(whisper_segments, diarization_segments))

    assert result[0].speaker == "A"
    assert result[1].speaker == "B"
    assert result[0].words[0].speaker == "A"
    assert result[0].words[1].speaker == "A"
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
        DiarizationSegment(segment=Segment(0, 3), speaker="A", label="A"),
        DiarizationSegment(segment=Segment(3, 5), speaker="B", label="B"),
        DiarizationSegment(segment=Segment(6, 8), speaker="A", label="A"),
        DiarizationSegment(segment=Segment(8, 10), speaker="B", label="B"),
        DiarizationSegment(segment=Segment(12, 15), speaker="C", label="C"),
    ]

    # Use a generator with a counter to track consumption
    consumed_count = 0

    def segment_generator():
        nonlocal consumed_count
        for segment in diarization_segments:
            consumed_count += 1
            yield segment

    # Process the first segment
    list(merge_whipser_diarization([whisper_segments[0]], segment_generator()))
    # Should have consumed first 2 segments (0-3, 3-5) and peeked at the third
    assert consumed_count == 3

    # Reset counter for the second test
    consumed_count = 0

    # Process first and second whisper segments
    list(merge_whipser_diarization(whisper_segments[:2], segment_generator()))
    # Should have consumed first 4 segments (0-3, 3-5, 6-8, 8-10) and peeked at the fifth
    assert consumed_count == 5

    # Reset counter for the full test
    consumed_count = 0

    # Process all segments
    result = list(merge_whipser_diarization(whisper_segments, segment_generator()))
    # Should have consumed all 5 segments
    assert consumed_count == 5

    # Verify the speakers were assigned correctly
    assert result[0].speaker in ("A", "B")  # First segment overlaps with both A and B
    assert result[1].speaker in ("A", "B")  # Second segment overlaps with both A and B
    assert result[2].speaker == "C"  # Third segment overlaps only with C


if __name__ == "__main__":
    pytest.main()
