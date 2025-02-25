import json
from pathlib import Path

from fastapi import HTTPException
import pytest
from bentoml.exceptions import InvalidArgument
import torch

from api_models.enums import TimestampGranularity, ResponseFormat
from api_models.input_models import TranscriptionRequest
from service import FasterWhisper


@pytest.fixture(scope="module")
def faster_whisper_service():
    """Create a single FasterWhisper instance for all tests in this module."""
    return FasterWhisper()


def _extend_params(**params):
    if "timestamp_granularities" in params:
        params = params | {
            "timestamp_granularities[]": params.get("timestamp_granularities") or []
        }

    return TranscriptionRequest.model_validate(params).model_dump()


def test_transcribe_standard_case(faster_whisper_service):
    # given
    file = Path("./tests/assets/example_audio.mp3")

    # when
    transcription = faster_whisper_service.transcribe(
        **_extend_params(file=file, response_format=ResponseFormat.JSON)
    )

    # then
    assert transcription is not None


@pytest.mark.parametrize(
    "temperature", [[0.3, 0.6], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], 0.0]
)
def test_transcribe_temperature(faster_whisper_service, temperature):
    # given
    file = Path("./tests/assets/example_audio.mp3")

    # when
    transcription = faster_whisper_service.transcribe(
        **_extend_params(
            file=file, temperature=temperature, response_format=ResponseFormat.JSON
        )
    )

    # then
    assert transcription is not None


@pytest.mark.parametrize(
    "response_format, timestamp_granularities",
    [
        (ResponseFormat.JSON, []),
        (ResponseFormat.VERBOSE_JSON, [TimestampGranularity.WORD]),
        (ResponseFormat.SRT, []),
        (ResponseFormat.TEXT, []),
        (ResponseFormat.VTT, []),
    ],
)
def test_transcribe_response_format(
    faster_whisper_service, response_format, timestamp_granularities
):
    # given
    file = Path("./tests/assets/example_audio.mp3")

    params = _extend_params(
        file=file,
        response_format=response_format,
        timestamp_granularities=timestamp_granularities,
    )

    # when
    transcription = faster_whisper_service.transcribe(**params)

    # then
    assert transcription is not None


def test_response_format_verbose_timestamp_granularities_segment(
    faster_whisper_service,
):
    # given
    file = Path("./tests/assets/example_audio.mp3")
    response_format = ResponseFormat.VERBOSE_JSON
    timestamp_granularities = [TimestampGranularity.SEGMENT]

    # when/then
    with pytest.raises(InvalidArgument):
        faster_whisper_service.transcribe(
            **_extend_params(
                file=file,
                response_format=response_format,
                timestamp_granularities=timestamp_granularities,
            )
        )


def test_response_format_verbose_timestamp_granularities_word(faster_whisper_service):
    # given
    file = Path("./tests/assets/example_audio.mp3")
    response_format = ResponseFormat.VERBOSE_JSON
    timestamp_granularities = [TimestampGranularity.WORD]

    # when
    transcription = faster_whisper_service.transcribe(
        **_extend_params(
            file=file,
            response_format=response_format,
            timestamp_granularities=timestamp_granularities,
        )
    )

    # then
    assert json.loads(transcription)["words"] is not None


@pytest.mark.asyncio
async def test_transcribe_streaming(faster_whisper_service):
    # given
    file = Path("./tests/assets/example_audio.mp3")
    chunks = []

    # when
    async for chunk in faster_whisper_service.streaming_transcribe(
        **_extend_params(file=file, response_format=ResponseFormat.JSON)
    ):
        chunks.append(chunk)

    # then
    assert chunks is not None


def test_transcribe_task(faster_whisper_service):
    # given
    file = Path("./tests/assets/example_audio.mp3")

    # when
    transcription = faster_whisper_service.task_transcribe(
        **_extend_params(file=file, response_format=ResponseFormat.JSON)
    )

    # then
    assert transcription is not None
