import enum
import os

import torch
from pydantic import BaseModel, Field

from api_models.enums import Language, ResponseFormat, TimestampGranularity


class Device(enum.StrEnum):
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"


class Quantization(enum.StrEnum):
    INT8 = "int8"
    INT8_FLOAT16 = "int8_float16"
    INT8_BFLOAT16 = "int8_bfloat16"
    INT8_FLOAT32 = "int8_float32"
    INT16 = "int16"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"
    DEFAULT = "default"


class WhisperModelConfig(BaseModel):
    """See https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py#L599."""

    inference_device: Device = Device.CUDA if torch.cuda.is_available() else Device.AUTO
    device_index: int | list[int] = 0
    compute_type: Quantization = Quantization.FLOAT16 if torch.cuda.is_available() else Quantization.DEFAULT
    cpu_threads: int = 0
    num_workers: int = 1
    ttl: int = Field(default=300, ge=-1)
    """
    Time in seconds until the model is unloaded if it is not being used.
    -1: Never unload the model.
    0: Unload the model immediately after usage.
    """


class FasterWhisperConfig(BaseModel):
    default_model_name: str = "large-v2"
    default_prompt: str = ""
    default_language: Language | None = None
    default_response_format: ResponseFormat = ResponseFormat.JSON
    default_temperature: list[float] = [
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
    ]
    default_timestamp_granularities: list[TimestampGranularity] = [TimestampGranularity.SEGMENT]

    min_temperature: float = 0.0
    max_temperature: float = 2.0

    best_of: int = 10
    vad_filter: bool = True
    vad_parameters: dict[str, float | int] = dict(
        threshold=0.5,
        neg_threshold=0.15,
        min_speech_duration_ms=0,
        max_speech_duration_s=999_999,
        min_silence_duration_ms=2000,
        speech_pad_ms=400,
    )
    diarization: bool = True
    condition_on_previous_text: bool = True
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    hotwords: str = ""
    beam_size: int = 5
    patience: float = 1.0
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    prompt_reset_on_temperature: float = 0.5


class LanguageIdConfig(BaseModel):
    """Tunables for the turn-level language identification in the multi-language
    path (diarization enabled, no ``language`` given); consumed by
    ``helpers/language_id.py``.

    Every field can be overridden by an environment variable named ``LID_<FIELD>``
    (e.g. ``LID_SWITCH_PENALTY=3.0``), typically set via the ``.env`` file. An
    invalid value fails at import with a validation error rather than being
    silently replaced by the default.
    """

    min_turn_s: float = Field(default=2.0, gt=0.0)
    """Turns shorter than this get no own detection pass (Whisper's language ID is
    unreliable on very short clips); they are resolved from context instead."""

    batch_size: int = Field(default=8, ge=1)
    """Encoder batch for language-detection windows; bounds peak GPU memory
    alongside the decode model."""

    inventory_mass_share: float = Field(default=0.15, gt=0.0, le=1.0)
    """Share of the total probability-weighted speech time a language needs to
    enter the auto-detected language inventory."""

    min_language_mass_s: float = Field(default=15.0, gt=0.0)
    """Absolute probability-weighted seconds that also admit a language into the
    inventory, regardless of share — keeps a minority language alive in long files."""

    switch_penalty: float = Field(default=2.0, ge=0.0)
    """Viterbi cost (log-probability units) of switching language between adjacent
    turns; higher values smooth harder over isolated misdetections."""

    evidence_cap_s: float = Field(default=10.0, gt=0.0)
    """Cap on a turn's emission weight in seconds, so a single confidently
    misdetected long turn can still be outvoted by its context."""

    @classmethod
    def from_env(cls, prefix: str = "LID_") -> "LanguageIdConfig":
        overrides = {
            name: os.environ[f"{prefix}{name.upper()}"]
            for name in cls.model_fields
            if f"{prefix}{name.upper()}" in os.environ
        }
        return cls.model_validate(overrides)


faster_whisper_config = FasterWhisperConfig()
language_id_config = LanguageIdConfig.from_env()
