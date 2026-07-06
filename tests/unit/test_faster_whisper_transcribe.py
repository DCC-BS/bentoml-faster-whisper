import json
from pathlib import Path

import pytest
from bentoml.exceptions import InvalidArgument

from api_models.enums import ResponseFormat, TimestampGranularity
from api_models.TranscriptionRequest import TranscriptionRequest
from service import FasterWhisper


@pytest.fixture(scope="module")
def faster_whisper_service():
    """Create a single FasterWhisper instance for all tests in this module."""
    return FasterWhisper()


def _extend_params(**params):
    # diarization off by default: these tests target the Whisper path, not the pyannote pipeline.
    params.setdefault("diarization", False)
    return TranscriptionRequest.model_validate(params).model_dump()


def test_transcribe_standard_case(faster_whisper_service):
    # given
    file = Path("./tests/assets/example_audio.mp3")

    # when
    transcription = faster_whisper_service.transcribe(**_extend_params(file=file, response_format=ResponseFormat.JSON))

    # then
    assert transcription is not None


@pytest.mark.parametrize("temperature", [[0.3, 0.6], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], 0.0])
def test_transcribe_temperature(faster_whisper_service, temperature):
    # given
    file = Path("./tests/assets/example_audio.mp3")

    # when
    transcription = faster_whisper_service.transcribe(
        **_extend_params(file=file, temperature=temperature, response_format=ResponseFormat.JSON)
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
def test_transcribe_response_format(faster_whisper_service, response_format, timestamp_granularities):
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


def test_transcribe_streaming(faster_whisper_service):
    # given
    file = Path("./tests/assets/example_audio.mp3")
    chunks = []

    # when
    for chunk in faster_whisper_service.streaming_transcribe(
        **_extend_params(file=file, response_format=ResponseFormat.JSON)
    ):
        chunks.append(chunk)

    # then
    assert chunks, "streaming must emit at least one chunk"
    for chunk in chunks:
        # NDJSON contract: bare, newline-delimited JSON payloads with no SSE framing.
        assert chunk.endswith("\n"), f"chunk must be newline-delimited, got {chunk!r}"
        assert not chunk.startswith("data:"), f"chunk must not carry SSE 'data:' framing, got {chunk!r}"
        payload = json.loads(chunk)  # each line must be valid, self-contained JSON
        assert "text" in payload


def test_transcribe_task(faster_whisper_service):
    # given
    file = Path("./tests/assets/example_audio.mp3")

    # when
    transcription = faster_whisper_service.task_transcribe(
        **_extend_params(file=file, response_format=ResponseFormat.JSON)
    )

    # then
    assert transcription is not None
