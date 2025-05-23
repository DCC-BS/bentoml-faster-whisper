import logging
from typing import Annotated, Generator, Iterable, Literal

from bentoml.validators import ContentType
from faster_whisper.transcribe import TranscriptionInfo
from pydantic import BaseModel, ConfigDict, Field

from api_models.enums import ResponseFormat
from api_models.TranscriptionJsonDiariexedResponse import (
    TranscriptionJsonDiariexedResponse,
)
from api_models.TranscriptionJsonResponse import TranscriptionJsonResponse
from api_models.TranscriptionVerboseJsonResponse import TranscriptionVerboseJsonResponse
from core import Segment, segments_to_srt, segments_to_text, segments_to_vtt

logger = logging.getLogger(__name__)

WhisperResponse = (
    Annotated[str, ContentType("application/json")]
    | TranscriptionJsonResponse
    | TranscriptionJsonDiariexedResponse
    | TranscriptionVerboseJsonResponse
)


class ModelObject(BaseModel):
    id: str
    """The model identifier, which can be referenced in the API endpoints."""
    created: int
    """The Unix timestamp (in seconds) when the model was created."""
    object_: Literal["model"] = Field(serialization_alias="object")
    """The object type, which is always "model"."""
    owned_by: str
    """The organization that owns the model."""
    language: list[str] = Field(default_factory=list)
    """List of ISO 639-3 supported by the model. It's possible that the list will be empty. This field is not a part of the OpenAI API spec and is added for convenience."""  # noqa: E501

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {
                    "id": "Systran/faster-whisper-large-v3",
                    "created": 1700732060,
                    "object": "model",
                    "owned_by": "Systran",
                },
                {
                    "id": "Systran/faster-distil-whisper-large-v3",
                    "created": 1711378296,
                    "object": "model",
                    "owned_by": "Systran",
                },
                {
                    "id": "bofenghuang/whisper-large-v2-cv11-french-ct2",
                    "created": 1687968011,
                    "object": "model",
                    "owned_by": "bofenghuang",
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
    if response_format == ResponseFormat.TEXT:  # noqa: RET503
        return segments_to_text(segments)
    elif response_format == ResponseFormat.JSON:
        return TranscriptionJsonResponse.from_segments(segments).model_dump_json()
    elif response_format == ResponseFormat.JSON_DIARZED:
        return TranscriptionJsonDiariexedResponse.from_segments(segments).model_dump_json()
    elif response_format == ResponseFormat.VERBOSE_JSON:
        return TranscriptionVerboseJsonResponse.from_segments(segments, transcription_info).model_dump_json()
    elif response_format == ResponseFormat.VTT:
        return "".join(segments_to_vtt(segment, i) for i, segment in enumerate(segments))
    elif response_format == ResponseFormat.SRT:
        return "".join(segments_to_srt(segment, i) for i, segment in enumerate(segments))


def format_as_sse(data: str) -> str:
    return f"data: {data}\n\n"


def segments_to_streaming_response(
    segments: Iterable[Segment],
    transcription_info: TranscriptionInfo,
    response_format: ResponseFormat,
) -> Generator["WhisperResponse", None, None]:
    def segment_responses() -> Generator[str, None, None]:
        for i, segment in enumerate(segments):
            if response_format == ResponseFormat.TEXT:
                data = segment.text
            elif response_format == ResponseFormat.JSON:
                data = TranscriptionJsonResponse.from_segments([segment]).model_dump_json()
            elif response_format == ResponseFormat.JSON_DIARZED:
                data = TranscriptionJsonDiariexedResponse.from_segments([segment]).model_dump_json()
            elif response_format == ResponseFormat.VERBOSE_JSON:
                data = TranscriptionVerboseJsonResponse.from_segment(segment, transcription_info).model_dump_json()
            elif response_format == ResponseFormat.VTT:
                data = segments_to_vtt(segment, i)
            elif response_format == ResponseFormat.SRT:
                data = segments_to_srt(segment, i)
            else:
                raise ValueError(f"Unknown response format: {response_format}")
            yield format_as_sse(data)

    return segment_responses()
