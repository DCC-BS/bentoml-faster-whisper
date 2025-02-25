from pathlib import Path

import pytest
from api_models.enums import ResponseFormat
from service import FasterWhisper
from api_models.input_models import TranslationRequest


@pytest.fixture(scope="module")
def faster_whisper_service():
    """Create a single FasterWhisper instance for all tests in this module."""
    return FasterWhisper()


def _extend_params(**params):
    return TranslationRequest.model_validate(params).model_dump()


def test_translate_standard_case(faster_whisper_service):
    # given
    file = Path("./tests/assets/example_audio_german.mp3")

    # when
    transcription = faster_whisper_service.translate(
        **_extend_params(file=file, response_format=ResponseFormat.JSON)
    )

    # then
    assert transcription is not None
