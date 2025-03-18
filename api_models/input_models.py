from typing import Annotated

from annotated_types import Ge, Le
from bentoml.exceptions import InvalidArgument
from huggingface_hub import ModelInfo
from pydantic import BaseModel, BeforeValidator, Field

from api_models.enums import ResponseFormat, TimestampGranularity
from api_models.output_models import ModelObject, logger
from config import faster_whisper_config

ModelName = Annotated[
    str,
    Field(
        description="The ID of the model. You can get a list of available models by calling `/v1/models`.",
        examples=[
            "Systran/faster-distil-whisper-large-v2",
            "bofenghuang/whisper-large-v2-cv11-french-ct2",
        ],
    ),
]


def _convert_timestamp_granularities(
    timestamp_granularities: str | list[TimestampGranularity],
) -> list[TimestampGranularity]:
    if isinstance(timestamp_granularities, list):
        return timestamp_granularities

    timestamps = timestamp_granularities.split(",")
    return [TimestampGranularity(t.strip()) for t in timestamps]


TimestampGranularities = Annotated[list[TimestampGranularity], BeforeValidator(_convert_timestamp_granularities)]


def _convert_temperature(
    temperature: str | int | float | list[float],
) -> list[float]:
    if isinstance(temperature, list):
        return temperature
    elif isinstance(temperature, (float, int)):
        return [float(temperature)]

    temperatures: list[str] = temperature.split(",")

    return [float(t.strip()) for t in temperatures]


BoundedTemperature = Annotated[
    float,
    Ge(faster_whisper_config.min_temperature),
    Le(faster_whisper_config.max_temperature),
]

ValidatedTemperature = Annotated[
    BoundedTemperature | list[BoundedTemperature],
    BeforeValidator(_convert_temperature),
]


def _process_empty_response_format(response_format: str | ResponseFormat | bytes | None) -> ResponseFormat:
    if isinstance(response_format, bytes):
        try:
            response_format = response_format.decode("utf-8")
        except UnicodeDecodeError:
            logger.warning(
                f"Cannot decode bytes response_format: {response_format}. Using default: {faster_whisper_config.default_response_format}"
            )
            return ResponseFormat(faster_whisper_config.default_response_format)

    if response_format == "" or response_format is None:
        return ResponseFormat(faster_whisper_config.default_response_format)

    if isinstance(response_format, ResponseFormat):
        return response_format

    try:
        return ResponseFormat(response_format)
    except ValueError:
        logger.warning(
            f"Invalid response format: {response_format}. Using default: {faster_whisper_config.default_response_format}"
        )
        return ResponseFormat(faster_whisper_config.default_response_format)


ValidatedResponseFormat = Annotated[ResponseFormat, BeforeValidator(_process_empty_response_format)]


def hf_model_info_to_model_object(model: ModelInfo) -> ModelObject:
    assert model.created_at is not None
    assert model.card_data is not None
    assert model.card_data.language is None or isinstance(model.card_data.language, str | list)
    if model.card_data.language is None:
        language = []
    elif isinstance(model.card_data.language, str):
        language = [model.card_data.language]
    else:
        language = model.card_data.language
    transformed_model = ModelObject(
        id=model.id,
        created=int(model.created_at.timestamp()),
        object_="model",
        owned_by=model.id.split("/")[0],
        language=language,
    )
    return transformed_model


def validate_timestamp_granularities(response_format, timestamp_granularities, diarization: bool | None):
    if (
        timestamp_granularities != faster_whisper_config.default_timestamp_granularities
        and response_format != ResponseFormat.VERBOSE_JSON
    ):
        logger.warning(
            "It only makes sense to provide `timestamp_granularities[]` when `response_format` is set to "
            "`verbose_json`. See https://platform.openai.com/docs/api-reference/audio/createTranscription#audio"
            "-createtranscription-timestamp_granularities."
            # noqa: E501
        )

    if "word" not in timestamp_granularities and response_format == ResponseFormat.VERBOSE_JSON:
        raise InvalidArgument(
            f"timestamp_granularities must contain 'word' when response_format "
            f"is set to {ResponseFormat.VERBOSE_JSON}"
        )

    if response_format == ResponseFormat.JSON_DIARZED and not diarization:
        raise InvalidArgument(
            f"response_format must be set to {ResponseFormat.JSON_DIARZED} when diarization is enabled"
        )

    if "word" in timestamp_granularities and response_format == ResponseFormat.JSON_DIARZED:
        raise InvalidArgument(
            f"timestamp_granularities must not contain 'word' when response_format "
            f"is set to {ResponseFormat.JSON_DIARZED}"
        )


class ValidatedVadOptions(BaseModel):
    threshold: Annotated[float, "Speech threshold", Ge(0.0), Le(1.0)] = 0.5
    neg_threshold: Annotated[float, "Silence threshold", Ge(0.0), Le(1.0)] = 0.15
    min_speech_duration_ms: Annotated[int, "Minimum speech duration in milliseconds", Ge(0), Le(1000)] = 0
    max_speech_duration_s: Annotated[float, "Maximum speech duration in seconds", Ge(0.5)] = 999_999
    min_silence_duration_ms: Annotated[int, "Minimum silence duration in milliseconds", Ge(100), Le(10_000)] = 2000
    speech_pad_ms: Annotated[int, "Speech padding in milliseconds", Ge(10), Le(1000)] = 400
