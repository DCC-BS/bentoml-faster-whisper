import logging
from typing import Annotated, Generator, Iterable, Literal

from bentoml.validators import ContentType
from faster_whisper.transcribe import TranscriptionInfo
from pydantic import BaseModel, ConfigDict, Field

from bentoml_faster_whisper.models.enums import ResponseFormat
from bentoml_faster_whisper.models.transcription_json_diarized_response import (
    TranscriptionJsonDiarizedResponse,
)
from bentoml_faster_whisper.models.transcription_json_response import TranscriptionJsonResponse
from bentoml_faster_whisper.models.transcription_verbose_json_response import TranscriptionVerboseJsonResponse
from bentoml_faster_whisper.utils.core import Segment, segments_to_srt, segments_to_text, segments_to_vtt

logger = logging.getLogger(__name__)

WhisperResponse = (
    Annotated[str, ContentType("application/json")]
    | TranscriptionJsonResponse
    | TranscriptionJsonDiarizedResponse
    | TranscriptionVerboseJsonResponse
)


def content_type_for_format(response_format: ResponseFormat) -> str:
    """HTTP Content-Type for a rendered response body. The endpoints declare a single
    ``application/json`` return type (the JSON formats dominate), so text/vtt/srt would
    otherwise be served mislabelled as JSON; the endpoint overrides the header per request."""
    return {
        ResponseFormat.TEXT: "text/plain; charset=utf-8",
        ResponseFormat.VTT: "text/vtt; charset=utf-8",
        ResponseFormat.SRT: "application/x-subrip; charset=utf-8",
    }.get(response_format, "application/json")


class ModelObject(BaseModel):
    id: str
    created: int
    object_: Literal["model"] = Field(serialization_alias="object")
    owned_by: str
    language: list[str] = Field(default_factory=list)
    """List of ISO 639-3 supported by the model. It's possible that the list will be empty. This field is not a part of the OpenAI API spec and is added for convenience."""  # noqa: E501

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {
                    "id": "large-v2",
                    "created": 1668556800,
                    "object": "model",
                    "owned_by": "Systran",
                },
            ]
        },
    )


class ModelListResponse(BaseModel):
    data: list[ModelObject]
    object: Literal["list"] = "list"


def segments_to_response(
    segments: Iterable[Segment],
    transcription_info: TranscriptionInfo,
    response_format: ResponseFormat,
) -> "WhisperResponse":
    segments = list(segments)
    if response_format == ResponseFormat.TEXT:
        return segments_to_text(segments)
    elif response_format == ResponseFormat.JSON:
        return TranscriptionJsonResponse.from_segments(segments).model_dump_json()
    elif response_format == ResponseFormat.JSON_DIARIZED:
        return TranscriptionJsonDiarizedResponse.from_segments(segments).model_dump_json()
    elif response_format == ResponseFormat.VERBOSE_JSON:
        return TranscriptionVerboseJsonResponse.from_segments(segments, transcription_info).model_dump_json()
    elif response_format == ResponseFormat.VTT:
        return "".join(segments_to_vtt(segment, i) for i, segment in enumerate(segments))
    elif response_format == ResponseFormat.SRT:
        return "".join(segments_to_srt(segment, i) for i, segment in enumerate(segments))
    else:
        raise ValueError(f"Unknown response format: {response_format}")


def segments_to_streaming_response(
    segments: Iterable[Segment],
    transcription_info: TranscriptionInfo,
    response_format: ResponseFormat,
) -> Generator[str, None, None]:
    """Stream one newline-delimited chunk per segment (NDJSON-style).

    Each chunk is the bare payload for the requested format followed by a
    single ``\\n`` so consumers can split the stream on line boundaries. No
    SSE (``data: ``) framing is applied; callers that need SSE must add it.
    """

    def segment_responses() -> Generator[str, None, None]:
        for i, segment in enumerate(segments):
            if response_format == ResponseFormat.TEXT:
                data = segment.text
            elif response_format == ResponseFormat.JSON:
                data = TranscriptionJsonResponse.from_segments([segment]).model_dump_json()
            elif response_format == ResponseFormat.JSON_DIARIZED:
                data = TranscriptionJsonDiarizedResponse.from_segments([segment]).model_dump_json()
            elif response_format == ResponseFormat.VERBOSE_JSON:
                data = TranscriptionVerboseJsonResponse.from_segment(segment, transcription_info).model_dump_json()
            elif response_format == ResponseFormat.VTT:
                data = segments_to_vtt(segment, i)
            elif response_format == ResponseFormat.SRT:
                data = segments_to_srt(segment, i)
            else:
                raise ValueError(f"Unknown response format: {response_format}")
            yield f"{data}\n"

    return segment_responses()
