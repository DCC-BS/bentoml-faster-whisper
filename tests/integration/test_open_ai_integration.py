from pathlib import Path

import pytest
from openai import OpenAI
from openai.types.audio import Transcription

from api_models.enums import ResponseFormat, TimestampGranularity

# to run these test start the server with the following command:
# uv run bentoml serve service:FasterWhisper -p 8003


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
def test_transcribe_endpoint(response_format, timestamp_granularities):
    # given

    file = Path(__file__).resolve().parent.parent / "assets" / "example_audio.mp3"

    openai = OpenAI(
        base_url="http://localhost:50001/v1",
        api_key="none",
    )

    # when
    transcription: Transcription = openai.audio.transcriptions.create(
        file=file,
        response_format=response_format,
        model="large-v2",
        timestamp_granularities=timestamp_granularities,
    )

    result = ""
    match response_format:
        case ResponseFormat.JSON:
            result = transcription.text
        case ResponseFormat.VERBOSE_JSON:
            result = transcription.text
        case _:
            result = transcription

    # then
    assert "I am just a sample audio text." in result
