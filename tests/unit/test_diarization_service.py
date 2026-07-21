import unittest.mock
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pyannote.core import Segment

from bentoml_faster_whisper.services.diarization_service import DiarizationSegment, DiarizationService


def _make_service_with_mock_pipeline(turns: list[tuple]) -> tuple[DiarizationService, MagicMock]:
    mock_output = MagicMock()
    mock_output.speaker_diarization = turns

    mock_pipeline = MagicMock()
    mock_pipeline.return_value = mock_output

    sut = DiarizationService()
    sut.pipeline = mock_pipeline
    return sut, mock_pipeline


def test_diarize_file_not_found():
    sut, _ = _make_service_with_mock_pipeline([])
    with pytest.raises(FileNotFoundError):
        list(sut.diarize("/nonexistent/path/audio.wav"))


def test_diarize_invalid_num_speaker_zero():
    sut, _ = _make_service_with_mock_pipeline([])
    with pytest.raises(ValueError):
        list(sut.diarize(__file__, num_speaker=0))


def test_diarize_invalid_num_speaker_negative():
    sut, _ = _make_service_with_mock_pipeline([])
    with pytest.raises(ValueError):
        list(sut.diarize(__file__, num_speaker=-1))


def test_diarize_returns_segments(tmp_path):
    turns = [
        (Segment(0.0, 3.0), "SPEAKER_00"),
        (Segment(3.5, 7.0), "SPEAKER_01"),
    ]
    sut, _ = _make_service_with_mock_pipeline(turns)

    audio_file = tmp_path / "audio.wav"
    audio_file.touch()

    result = list(sut.diarize(str(audio_file)))

    assert len(result) == 2
    assert result[0].start == pytest.approx(0.0)
    assert result[0].end == pytest.approx(3.0)
    assert result[0].speaker == "SPEAKER_00"
    assert result[1].start == pytest.approx(3.5)
    assert result[1].end == pytest.approx(7.0)
    assert result[1].speaker == "SPEAKER_01"


def test_diarize_passes_num_speakers_to_pipeline(tmp_path):
    sut, mock_pipeline = _make_service_with_mock_pipeline([])
    audio_file = tmp_path / "audio.wav"
    audio_file.touch()

    list(sut.diarize(str(audio_file), num_speaker=2))

    mock_pipeline.assert_called_once()
    _, kwargs = mock_pipeline.call_args
    assert kwargs["num_speakers"] == 2


def test_diarize_passes_none_num_speakers_by_default(tmp_path):
    sut, mock_pipeline = _make_service_with_mock_pipeline([])
    audio_file = tmp_path / "audio.wav"
    audio_file.touch()

    list(sut.diarize(str(audio_file)))

    mock_pipeline.assert_called_once()
    _, kwargs = mock_pipeline.call_args
    assert kwargs["num_speakers"] is None


def test_diarize_passes_wav_path_directly(tmp_path):
    """WAV files are passed directly to the pipeline — no conversion, no waveform in RAM."""
    sut, mock_pipeline = _make_service_with_mock_pipeline([])
    audio_file = tmp_path / "audio.wav"
    audio_file.touch()

    list(sut.diarize(str(audio_file)))

    mock_pipeline.assert_called_once()
    args, _ = mock_pipeline.call_args
    assert args[0] == str(audio_file)


def test_diarize_converts_mp3_to_wav(tmp_path):
    """MP3 files are converted to a temp WAV before passing to the pipeline."""
    sut, mock_pipeline = _make_service_with_mock_pipeline([])
    audio_file = tmp_path / "audio.mp3"
    audio_file.touch()

    with unittest.mock.patch("subprocess.run") as mock_run:
        # Make ffmpeg appear to succeed and create a temp WAV
        def fake_ffmpeg(*args, **kwargs):
            Path(args[0][-1]).touch()

        mock_run.side_effect = fake_ffmpeg
        list(sut.diarize(str(audio_file)))

    mock_pipeline.assert_called_once()
    args, _ = mock_pipeline.call_args
    assert args[0].endswith(".wav")
    assert args[0] != str(audio_file)


def test_diarization_segment_exposes_speaker_and_bounds():
    seg = DiarizationSegment(Segment(1.0, 2.0), speaker="SPEAKER_00")
    assert seg.speaker == "SPEAKER_00"
    assert seg.start == pytest.approx(1.0)
    assert seg.end == pytest.approx(2.0)
