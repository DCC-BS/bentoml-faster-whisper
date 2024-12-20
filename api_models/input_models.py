from pathlib import Path
from typing import Annotated, List, Optional, Union

from annotated_types import Ge, Le, MaxLen
from bentoml.exceptions import InvalidArgument
from huggingface_hub import ModelInfo
from pydantic import BaseModel, BeforeValidator, Field, confloat
from bentoml.validators import ContentType
from api_models.enums import Language, ResponseFormat, TimestampGranularity
from api_models.output_models import ModelObject, logger
from config import faster_whisper_config

ModelName = Annotated[
    str,
    Field(
        description="The ID of the model. You can get a list of available models by calling `/v1/models`.",
        examples=[
            "Systran/faster-distil-whisper-large-v3",
            "bofenghuang/whisper-large-v2-cv11-french-ct2",
        ],
    ),
]


def _convert_timestamp_granularities(
    timestamp_granularities: str | List[TimestampGranularity],
) -> List[TimestampGranularity]:
    if isinstance(timestamp_granularities, List):
        return timestamp_granularities

    timestamps = timestamp_granularities.split(",")
    return [TimestampGranularity(t.strip()) for t in timestamps]


TimestampGranularities = Annotated[
    List[TimestampGranularity], BeforeValidator(_convert_timestamp_granularities)
]


def _convert_temperature(
    temperature: Union[str, int, float, List[float]],
) -> List[float]:
    if isinstance(temperature, List):
        return temperature
    elif isinstance(temperature, (float, int)):
        return [float(temperature)]

    temperatures = temperature.split(",")

    return [float(t.strip()) for t in temperatures]


BoundedTemperature = confloat(
    ge=faster_whisper_config.min_temperature, le=faster_whisper_config.max_temperature
)

ValidatedTemperature = Annotated[
    Union[BoundedTemperature, List[BoundedTemperature]],
    BeforeValidator(_convert_temperature),
]


def _process_empty_response_format(response_format: Optional[str]) -> Optional[str]:
    if response_format == "":
        return faster_whisper_config.default_response_format
    return response_format


ValidatedResponseFormat = Annotated[
    ResponseFormat, BeforeValidator(_process_empty_response_format)
]


def _process_empty_language(language: Optional[str]) -> Optional[str]:
    if language == "":
        return faster_whisper_config.default_language
    return language


# None needed to return None from Validator
ValidatedLanguage = Annotated[Language | None, BeforeValidator(_process_empty_language)]


def hf_model_info_to_model_object(model: ModelInfo) -> ModelObject:
    assert model.created_at is not None
    assert model.card_data is not None
    assert model.card_data.language is None or isinstance(
        model.card_data.language, str | list
    )
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


def validate_timestamp_granularities(response_format, timestamp_granularities):
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

    if (
        "word" not in timestamp_granularities
        and response_format == ResponseFormat.VERBOSE_JSON
    ):
        raise InvalidArgument(
            f"timestamp_granularities must contain 'word' when response_format "
            f"is set to {ResponseFormat.VERBOSE_JSON}"
        )


class ValidatedVadOptions(BaseModel):
    onset: Annotated[float, "Speech threshold", Ge(0.0), Le(1.0)] = 0.5
    offset: Annotated[float, "Silence threshold", Ge(0.0), Le(1.0)] = 0.15
    min_speech_duration_ms: Annotated[
        int, "Minimum speech duration in milliseconds", Ge(0), Le(1000)
    ] = 0
    max_speech_duration_s: Annotated[
        float, "Maximum speech duration in seconds", Ge(0.5)
    ] = 999_999
    min_silence_duration_ms: Annotated[
        int, "Minimum silence duration in milliseconds", Ge(100), Le(10_000)
    ] = 2000
    speech_pad_ms: Annotated[int, "Speech padding in milliseconds", Ge(10), Le(1000)] = 400


# class BatchTranscriptionRequest(BaseModel):
#     file: Path
#     model: Optional[str] = faster_whisper_config.default_model_name
#     language: Optional[str] = faster_whisper_config.default_language
#     prompt: Optional[str] = faster_whisper_config.default_prompt
#     response_format: Optional[ResponseFormat] = (
#         faster_whisper_config.default_response_format
#     )
#     temperature: Optional[Union[float, List[float]]] = (
#         faster_whisper_config.default_temperature
#     )
#     timestamp_granularities: Optional[TimestampGranularities] = (
#         faster_whisper_config.default_timestamp_granularities
#     )
#     best_of: Optional[Annotated[int, Ge(1), Le(10)]] = faster_whisper_config.best_of
#     vad_filter: Optional[bool] = faster_whisper_config.vad_filter
#     vad_options: Optional[ValidatedVadOptions] = faster_whisper_config.vad_parameters
#     condition_on_previous_text: Optional[bool] = (
#         faster_whisper_config.condition_on_previous_text
#     )
#     repetition_penalty: Optional[Annotated[float, Ge(0.5), Le(2.0)]] = (
#         faster_whisper_config.repetition_penalty
#     )
#     length_penalty: Optional[Annotated[float, Ge(0.5), Le(2.0)]] = (
#         faster_whisper_config.length_penalty
#     )
#     no_repeat_ngram_size: Optional[Annotated[int, Ge(0), Le(5)]] = (
#         faster_whisper_config.no_repeat_ngram_size
#     )
#     hotwords: Optional[Annotated[str, MaxLen(500)]] = faster_whisper_config.hotwords
#     beam_size: Optional[Annotated[int, Ge(1), Le(16)]] = faster_whisper_config.beam_size
#     patience: Optional[Annotated[float, Ge(0.0), Le(1.0)]] = (
#         faster_whisper_config.patience
#     )
#     compression_ratio_threshold: Optional[Annotated[float, Ge(0.0), Le(4.0)]] = (
#         faster_whisper_config.compression_ratio_threshold
#     )
#     log_prob_threshold: Optional[Annotated[float, Ge(-10.0)]] = (
#         faster_whisper_config.log_prob_threshold
#     )
#     prompt_reset_on_temperature: Optional[Annotated[float, Ge(0.0), Le(2.0)]] = (
#         faster_whisper_config.prompt_reset_on_temperature
#     )

class TranscriptionRequest(BaseModel):
    file: Annotated[Path, ContentType("audio/mpeg")]
    model: Optional[ModelName] = Field(
        default=faster_whisper_config.default_model_name,
        description="Whisper model to load",
    )
    language: Optional[ValidatedLanguage] = Field(
        default=faster_whisper_config.default_language,
        description='The language spoken in the audio. It should be a language code such as "en" or "fr". If '
        "not set, the language will be detected in the first 30 seconds of audio.",
    )
    prompt: Optional[str] = Field(
        default=faster_whisper_config.default_prompt,
        description="Optional text string or iterable of token ids to provide as a prompt for the first window.",
    )
    response_format: Optional[ValidatedResponseFormat] = Field(
        default=faster_whisper_config.default_response_format,
        description="The format of the output, in one of these options: `json`, `text`, `srt`, `verbose_json`, "
        "or `vtt`.",
    )
    temperature: Optional[ValidatedTemperature] = Field(
        default=faster_whisper_config.default_temperature,
        description="Temperature value, which can either be a single float or a list of floats. "
        f"Valid Range: Between {faster_whisper_config.min_temperature} and "
        f"{faster_whisper_config.max_temperature}",
    )
    timestamp_granularities: Optional[TimestampGranularities] = Field(
        default=faster_whisper_config.default_timestamp_granularities,
        alias="timestamp_granularities[]",
        description="The timestamp granularities to populate for this transcription. response_format must be "
        "set verbose_json to use timestamp granularities.",
    )
    best_of: Optional[Annotated[int, Ge(1), Le(10)]] = Field(
        default=faster_whisper_config.best_of,
        desription="Number of candidates when sampling with non-zero temperature.",
    )
    vad_filter: Optional[bool] = Field(
        default=faster_whisper_config.vad_filter,
        description="Enable the voice activity detection (VAD) to filter out parts of the audio without speech. This step is using the Silero VAD model https://github.com/snakers4/silero-vad.",
    )
    vad_parameters: Optional[ValidatedVadOptions] = Field(
        default=faster_whisper_config.vad_parameters,
        description="Dictionary of Silero VAD parameters or VadOptions class (see available parameters and default values in the class `VadOptions`).",
    )
    condition_on_previous_text: Optional[bool] = Field(
        default=faster_whisper_config.condition_on_previous_text,
        description="If True, the previous output of the model is provided as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.",
    )
    repetition_penalty: Optional[Annotated[float, Ge(0.5), Le(2.0)]] = Field(
        default=faster_whisper_config.repetition_penalty,
        description="Penalty applied to the score of previously generated tokens (set > 1 to penalize).",
    )
    length_penalty: Optional[Annotated[float, Ge(0.5), Le(2.0)]] = Field(
        default=faster_whisper_config.length_penalty,
        description="Exponential length penalty constant.",
    )
    no_repeat_ngram_size: Optional[Annotated[int, Ge(0), Le(5)]] = Field(
        default=faster_whisper_config.no_repeat_ngram_size,
        description="Prevent repetitions of ngrams with this size (set 0 to disable).",
    )
    hotwords: Optional[Annotated[str, MaxLen(500)]] = Field(
        default=faster_whisper_config.hotwords,
        description="Hotwords/hint phrases to provide the model with. Has no effect if prefix is not None.",
    )
    beam_size: Optional[Annotated[int, Ge(1), Le(16)]] = Field(
        default=faster_whisper_config.beam_size,
        description="Beam size to use for decoding.",
    )
    patience: Optional[Annotated[float, Ge(0.0), Le(1.0)]] = Field(
        default=faster_whisper_config.patience,
        description="Beam search patience factor.",
    )
    compression_ratio_threshold: Optional[
        Annotated[float, Ge(0.0), Le(4.0)]
    ] = Field(
        default=faster_whisper_config.compression_ratio_threshold,
        description="If the gzip compression ratio is above this value, treat as failed.",
    )
    log_prob_threshold: Optional[Annotated[float, Ge(-10.0)]] = Field(
        default=faster_whisper_config.log_prob_threshold,
        description="If the average log probability over sampled tokens is below this value, treat as failed.",
    )
    prompt_reset_on_temperature: Optional[
        Annotated[float, Ge(0.0), Le(2.0)]
    ] = Field(
        default=faster_whisper_config.prompt_reset_on_temperature,
        description="Resets prompt if temperature is above this value. Arg has effect only if condition_on_previous_text is True.",
    )
