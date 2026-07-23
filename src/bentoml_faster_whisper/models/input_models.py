from typing import Annotated

from annotated_types import Ge, Le
from bentoml.exceptions import InvalidArgument
from pydantic import AfterValidator, BaseModel, BeforeValidator, Field

from bentoml_faster_whisper.config import faster_whisper_config
from bentoml_faster_whisper.models.enums import ResponseFormat, TimestampGranularity
from bentoml_faster_whisper.models.output_models import logger


def _validate_served_model(model: str) -> str:
    served = faster_whisper_config.default_model_name
    if model != served:
        raise ValueError(f"Only '{served}' is served by this API; got '{model}'.")
    return model


ModelName = Annotated[
    str,
    AfterValidator(_validate_served_model),
    Field(
        description=f"Whisper model to use. Only '{faster_whisper_config.default_model_name}' is served by this API.",
        examples=[faster_whisper_config.default_model_name],
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

    # Do NOT require 'word' for verbose_json: the OpenAI spec makes
    # timestamp_granularities optional and defaults it to ["segment"], so a plain
    # `response_format=verbose_json` request (no granularities) must be accepted and
    # return segment timestamps only. Rejecting it broke standard OpenAI SDK clients.

    if response_format == ResponseFormat.JSON_DIARIZED and not diarization:
        raise InvalidArgument(
            f"diarization must be enabled when response_format is set to {ResponseFormat.JSON_DIARIZED}"
        )

    if "word" in timestamp_granularities and response_format == ResponseFormat.JSON_DIARIZED:
        raise InvalidArgument(
            f"timestamp_granularities must not contain 'word' when response_format "
            f"is set to {ResponseFormat.JSON_DIARIZED}"
        )


class ValidatedVadOptions(BaseModel):
    threshold: Annotated[float, "Speech threshold", Ge(0.0), Le(1.0)] = 0.5
    neg_threshold: Annotated[float, "Silence threshold", Ge(0.0), Le(1.0)] = 0.15
    min_speech_duration_ms: Annotated[int, "Minimum speech duration in milliseconds", Ge(0), Le(1000)] = 0
    max_speech_duration_s: Annotated[float, "Maximum speech duration in seconds", Ge(0.5)] = 999_999
    min_silence_duration_ms: Annotated[int, "Minimum silence duration in milliseconds", Ge(100), Le(10_000)] = 2000
    speech_pad_ms: Annotated[int, "Speech padding in milliseconds", Ge(10), Le(1000)] = 400
