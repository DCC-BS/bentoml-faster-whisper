from fastapi import HTTPException
import pytest
from pydantic import ValidationError

from api_models.TranscriptionRequest import TranscriptionRequest
from config import faster_whisper_config
from service import FasterWhisper


def test_get_models_standard_case():
    # given
    faster_whisper_service = FasterWhisper()

    # when
    models = faster_whisper_service.get_models()

    # then
    assert models is not None
    assert [m.id for m in models.data] == [faster_whisper_config.default_model_name]


def test_get_model_served_case():
    faster_whisper_service = FasterWhisper()

    model = faster_whisper_service.get_model(faster_whisper_config.default_model_name)

    assert model.id == faster_whisper_config.default_model_name


def test_model_not_found():
    # given
    unknown_model_name = "unknown-model-v1"
    faster_whisper_service = FasterWhisper()

    # when / then
    with pytest.raises(HTTPException):
        faster_whisper_service.get_model(unknown_model_name)


def test_served_model_accepted():
    request = TranscriptionRequest.model_validate(
        {"file": "audio.mp3", "model": faster_whisper_config.default_model_name}
    )
    assert request.model == faster_whisper_config.default_model_name


def test_non_served_model_rejected():
    with pytest.raises(ValidationError):
        TranscriptionRequest.model_validate({"file": "audio.mp3", "model": "whisper-1"})
