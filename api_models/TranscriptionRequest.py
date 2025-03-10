from pathlib import Path
from typing import Annotated, Optional

from annotated_types import Ge, Le, MaxLen
from bentoml.exceptions import InvalidArgument
from bentoml.validators import ContentType
from pydantic import AliasChoices, BaseModel, BeforeValidator, ConfigDict, Field

from api_models.enums import Language
from api_models.input_models import (
    ModelName,
    TimestampGranularities,
    ValidatedResponseFormat,
    ValidatedTemperature,
    ValidatedVadOptions,
)
from config import faster_whisper_config


def _process_empty_language(language: Optional[str]) -> Optional[str]:
    if language == "":
        return faster_whisper_config.default_language
    return language


# None needed to return None from Validator
ValidatedLanguage = Annotated[Language | None, BeforeValidator(_process_empty_language)]


class TranscriptionRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    @classmethod
    def from_dict(cls, d: dict) -> "TranscriptionRequest":
        try:
            return TranscriptionRequest.model_validate(d)
        except TypeError as e:
            raise InvalidArgument(str(e))

    file: Annotated[Path, ContentType("audio/mpeg")]
    model: ModelName = Field(
        default=faster_whisper_config.default_model_name,
        description="Whisper model to load",
    )
    language: ValidatedLanguage = Field(
        default=faster_whisper_config.default_language,
        description='The language spoken in the audio. It should be a language code such as "en" or "fr". If '
        "not set, the language will be detected in the first 30 secondss of audio.",
    )
    prompt: str = Field(
        default=faster_whisper_config.default_prompt,
        description="Optional text string or iterable of token ids to provide as a prompt for the first window.",
    )
    response_format: "ValidatedResponseFormat" = Field(
        default=faster_whisper_config.default_response_format,
        description="The format of the output, in one of these options: `json`, `text`, `srt`, `verbose_json`, "
        "or `vtt`.",
    )
    temperature: ValidatedTemperature = Field(
        default=faster_whisper_config.default_temperature,
        description="Temperature value, which can either be a single float or a list of floats. "
        f"Valid Range: Between {faster_whisper_config.min_temperature} and "
        f"{faster_whisper_config.max_temperature}",
    )
    timestamp_granularities: TimestampGranularities = Field(
        default=faster_whisper_config.default_timestamp_granularities,
        validation_alias=AliasChoices("timestamp_granularities", "timestamp_granularities[]"),
        description="The timestamp granularities to populate for this transcription. response_format must be "
        "set verbose_json to use timestamp granularities.",
    )
    best_of: Annotated[int, Ge(1), Le(10)] = Field(
        default=faster_whisper_config.best_of,
        description="Number of candidates when sampling with non-zero temperature.",
    )
    vad_filter: bool = Field(
        default=faster_whisper_config.vad_filter,
        description="Enable the voice activity detection (VAD) to filter out parts of the audio without speech. This step is using the Silero VAD model https://github.com/snakers4/silero-vad.",
    )
    vad_parameters: ValidatedVadOptions = Field(
        default=ValidatedVadOptions.model_validate(faster_whisper_config.vad_parameters),
        description="Dictionary of Silero VAD parameters or VadOptions class (see available parameters and default values in the class `VadOptions`).",
    )
    condition_on_previous_text: bool = Field(
        default=faster_whisper_config.condition_on_previous_text,
        description="If True, the previous output of the model is provided as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.",
    )
    repetition_penalty: Annotated[float, Ge(0.5), Le(2.0)] = Field(
        default=faster_whisper_config.repetition_penalty,
        description="Penalty applied to the score of previously generated tokens (set > 1 to penalize).",
    )
    length_penalty: Annotated[float, Ge(0.5), Le(2.0)] = Field(
        default=faster_whisper_config.length_penalty,
        description="Exponential length penalty constant.",
    )
    no_repeat_ngram_size: Annotated[int, Ge(0), Le(5)] = Field(
        default=faster_whisper_config.no_repeat_ngram_size,
        description="Prevent repetitions of ngrams with this size (set 0 to disable).",
    )
    hotwords: Annotated[str, MaxLen(500)] = Field(
        default=faster_whisper_config.hotwords,
        description="Hotwords/hint phrases to provide the model with. Has no effect if prefix is not None.",
    )
    beam_size: Annotated[int, Ge(1), Le(16)] = Field(
        default=faster_whisper_config.beam_size,
        description="Beam size to use for decoding.",
    )
    patience: Annotated[float, Ge(0.0), Le(1.0)] = Field(
        default=faster_whisper_config.patience,
        description="Beam search patience factor.",
    )
    compression_ratio_threshold: Annotated[float, Ge(0.0), Le(4.0)] = Field(
        default=faster_whisper_config.compression_ratio_threshold,
        description="If the gzip compression ratio is above this value, treat as failed.",
    )
    log_prob_threshold: Annotated[float, Ge(-10.0)] = Field(
        default=faster_whisper_config.log_prob_threshold,
        description="If the average log probability over sampled tokens is below this value, treat as failed.",
    )
    prompt_reset_on_temperature: Annotated[float, Ge(0.0), Le(2.0)] = Field(
        default=faster_whisper_config.prompt_reset_on_temperature,
        description="Resets prompt if temperature is above this value. Arg has effect only if condition_on_previous_text is True.",
    )
    diarization: bool = Field(
        default=False,
        description="If True, the model will attempt to separate speakers in the audio.",
    )
    diarization_speaker_count: Annotated[Optional[int], Ge(2)] = Field(
        default=None,
        description="The number of speakers to separate in the audio. This argument is only used if diarization is True.",
    )
    progress_id: Optional[str] = Field(
        default=None,
        description="A unique identifier for reporting the progress of a task.",
    )
