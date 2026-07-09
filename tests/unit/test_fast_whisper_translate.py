from pathlib import Path

import pytest

from api_models.enums import ResponseFormat
from api_models.TranslationRequest import TranslationRequest

pytestmark = pytest.mark.model


def _extend_params(**params):
    return TranslationRequest.model_validate(params).model_dump()


def test_translate_standard_case(faster_whisper_service):
    # given
    file = Path("./tests/assets/example_audio_german.mp3")

    # when
    transcription = faster_whisper_service.translate(**_extend_params(file=file, response_format=ResponseFormat.JSON))

    # then
    assert transcription is not None
