from unittest.mock import MagicMock, patch

import pytest
from pyannote.core import Segment

from diarization_service import DiarizationSegment, DiarizationService


def _make_service_with_mock_pipeline(turns: list[tuple]) -> DiarizationService:
    """Build a DiarizationService with a mocked pipeline that returns the given (turn, speaker) pairs."""
    mock_output = MagicMock()
    mock_output.speaker_diarization = turns

    mock_pipeline = MagicMock()
    mock_pipeline.return_value = mock_output

    sut = DiarizationService()
    sut.pipeline = mock_pipeline
    return sut


def test_diarize_file_not_found():
    sut = _make_service_with_mock_pipeline([])
    with pytest.raises(FileNotFoundError):
        list(sut.diarize("/nonexistent/path/audio.wav"))


def test_diarize_invalid_num_speaker_zero():
    sut = _make_service_with_mock_pipeline([])
    with pytest.raises(ValueError):
        list(sut.diarize(__file__, num_speaker=0))


def test_diarize_invalid_num_speaker_negative():
    sut = _make_service_with_mock_pipeline([])
    with pytest.raises(ValueError):
        list(sut.diarize(__file__, num_speaker=-1))


def test_diarize_returns_segments(tmp_path):
    turns = [
        (Segment(0.0, 3.0), "SPEAKER_00"),
        (Segment(3.5, 7.0), "SPEAKER_01"),
    ]
    sut = _make_service_with_mock_pipeline(turns)

    audio_file = tmp_path / "audio.wav"
    audio_file.touch()

    with patch("torchaudio.load", return_value=(MagicMock(), 16000)):
        result = list(sut.diarize(str(audio_file)))

    assert len(result) == 2
    assert result[0].start == 0.0
    assert result[0].end == 3.0
    assert result[0].speaker == "SPEAKER_00"
    assert result[1].start == 3.5
    assert result[1].end == 7.0
    assert result[1].speaker == "SPEAKER_01"


def test_diarize_passes_num_speakers_to_pipeline(tmp_path):
    sut = _make_service_with_mock_pipeline([])
    audio_file = tmp_path / "audio.wav"
    audio_file.touch()

    with patch("torchaudio.load", return_value=(MagicMock(), 16000)):
        list(sut.diarize(str(audio_file), num_speaker=2))

    sut.pipeline.assert_called_once()  # type: ignore[union-attr]
    _, kwargs = sut.pipeline.call_args  # type: ignore[union-attr]
    assert kwargs["num_speakers"] == 2


def test_diarize_passes_none_num_speakers_by_default(tmp_path):
    sut = _make_service_with_mock_pipeline([])
    audio_file = tmp_path / "audio.wav"
    audio_file.touch()

    with patch("torchaudio.load", return_value=(MagicMock(), 16000)):
        list(sut.diarize(str(audio_file)))

    sut.pipeline.assert_called_once()  # type: ignore[union-attr]
    _, kwargs = sut.pipeline.call_args  # type: ignore[union-attr]
    assert kwargs["num_speakers"] is None


def test_diarization_segment_label_equals_speaker():
    seg = DiarizationSegment(Segment(1.0, 2.0), label="SPEAKER_00", speaker="SPEAKER_00")
    assert seg.label == seg.speaker
    assert seg.start == 1.0
    assert seg.end == 2.0
