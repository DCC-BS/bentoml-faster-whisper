from fastapi import HTTPException
import pytest
from service import FasterWhisper


def test_get_models_standard_case():
    # given
    faster_whisper_service = FasterWhisper()

    # when
    models = faster_whisper_service.get_models()

    # then
    assert models is not None


def test_model_not_found():
    # given
    unknown_model_name = "unknown-model-v1"
    faster_whisper_service = FasterWhisper()

    # when / then
    with pytest.raises(HTTPException):
        faster_whisper_service.get_model(unknown_model_name)
