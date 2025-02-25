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

    result = merge_whipser_diarization(whisper_segments, diarization_segments)

    assert result[0].speaker == "A"
    assert result[1].speaker == "B"
    assert result[0].words[0].speaker == "A"
    assert result[0].words[1].speaker == "A"
    assert result[1].words[0].speaker == "B"
    assert result[1].words[1].speaker == "B"


if __name__ == "__main__":
    pytest.main()
