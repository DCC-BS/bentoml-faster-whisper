from unittest.mock import MagicMock, patch
import av
import pytest
from bentoml.exceptions import InvalidArgument

from bentoml_faster_whisper.models.enums import ResponseFormat
from bentoml_faster_whisper.models.transcription_request import TranscriptionRequest
from bentoml_faster_whisper.models.translation_request import TranslationRequest
from bentoml_faster_whisper.services.faster_whisper_handler import (
    FasterWhisperHandler,
    _audio_decode_errors_as_invalid,
)


def test_audio_decode_errors_as_invalid_catches_ffmpeg_error():
    with pytest.raises(InvalidArgument, match="Failed to decode audio file"):
        with _audio_decode_errors_as_invalid():
            raise av.error.InvalidDataError(1, "Corrupt audio data")


def test_audio_decode_errors_as_invalid_does_not_catch_runtime_error():
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        with _audio_decode_errors_as_invalid():
            raise RuntimeError("CUDA out of memory")


def test_prepare_audio_segments_propagates_transcribe_runtime_error():
    model_manager = MagicMock()
    whisper_mock = MagicMock()
    whisper_mock.transcribe.side_effect = RuntimeError("CUDA out of memory during transcribe")
    model_manager.get.return_value = whisper_mock

    handler = FasterWhisperHandler(model_manager=model_manager, diarization=MagicMock())
    request = TranscriptionRequest.model_validate(
        {"file": "/tmp/dummy.mp3", "response_format": ResponseFormat.JSON, "diarization": False}
    )

    with patch("bentoml_faster_whisper.services.faster_whisper_handler.decode_audio") as mock_decode:
        mock_decode.return_value = MagicMock(shape=(16000,))
        with pytest.raises(RuntimeError, match="CUDA out of memory during transcribe"):
            handler.prepare_audio_segments(request)


def test_translate_audio_propagates_transcribe_runtime_error():
    model_manager = MagicMock()
    whisper_mock = MagicMock()
    whisper_mock.transcribe.side_effect = RuntimeError("CUDA out of memory during translate")
    model_manager.get.return_value = whisper_mock

    handler = FasterWhisperHandler(model_manager=model_manager, diarization=MagicMock())
    request = TranslationRequest.model_validate({"file": "/tmp/dummy.mp3", "response_format": ResponseFormat.JSON})

    with patch("bentoml_faster_whisper.services.faster_whisper_handler.decode_audio") as mock_decode:
        mock_decode.return_value = MagicMock(shape=(16000,))
        with pytest.raises(RuntimeError, match="CUDA out of memory during translate"):
            handler.translate_audio(request)


def test_prepare_audio_segments_maps_decode_audio_ffmpeg_error_to_invalid_argument():
    model_manager = MagicMock()
    handler = FasterWhisperHandler(model_manager=model_manager, diarization=MagicMock())
    request = TranscriptionRequest.model_validate(
        {"file": "/tmp/corrupt.mp3", "response_format": ResponseFormat.JSON, "diarization": False}
    )

    with patch("bentoml_faster_whisper.services.faster_whisper_handler.decode_audio") as mock_decode:
        mock_decode.side_effect = av.error.InvalidDataError(1, "Corrupt audio stream")
        with pytest.raises(InvalidArgument, match="Failed to decode audio file"):
            handler.prepare_audio_segments(request)
