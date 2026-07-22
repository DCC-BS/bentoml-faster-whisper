import enum
import os

import torch
from dcc_backend_common.config import AbstractAppConfig
from pydantic import BaseModel, Field

from bentoml_faster_whisper.models.enums import Language, ResponseFormat, TimestampGranularity


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
    # int8_float16 on CUDA: ~50% less VRAM and faster decoder token generation with no quality
    # regression on the curated German eval (WER 0.460 -> 0.455, CER 0.159 -> 0.157). CPU builds
    # keep DEFAULT since int8_float16 is a CUDA compute type.
    compute_type: Quantization = Quantization.INT8_FLOAT16 if torch.cuda.is_available() else Quantization.DEFAULT
    cpu_threads: int = 0
    num_workers: int = 1


class FasterWhisperConfig(BaseModel):
    default_model_name: str = Field(default_factory=lambda: os.getenv("DEFAULT_WHISPER_MODEL", "large-v2"))
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
    vad_parameters: dict[str, float | int] = Field(
        default_factory=lambda: dict(
            threshold=0.5,
            neg_threshold=0.15,
            min_speech_duration_ms=0,
            max_speech_duration_s=999_999,
            min_silence_duration_ms=2000,
            speech_pad_ms=400,
        )
    )
    diarization: bool = True
    # False by default: on the curated German eval this improved every metric (WER 0.460 -> 0.453,
    # CER 0.159 -> 0.154, BLEU 45.7 -> 46.1) by avoiding the hallucination cascades that
    # previous-text conditioning can trigger across 30s windows. Still per-request overridable.
    condition_on_previous_text: bool = False
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
    ``utils/language_id.py``.

    Every field can be overridden by an environment variable named ``LID_<FIELD>``
    (e.g. ``LID_SWITCH_PENALTY=3.0``), typically set via the ``.env`` file. An
    invalid value fails at import with a validation error rather than being
    silently replaced by the default.
    """

    min_turn_s: float = Field(default=1.0, gt=0.0)
    batch_size: int = Field(default=8, ge=1)
    inventory_mass_share: float = Field(default=0.15, gt=0.0, le=1.0)
    min_language_mass_s: float = Field(default=15.0, gt=0.0)
    switch_penalty: float = Field(default=2.0, ge=0.0)
    evidence_cap_s: float = Field(default=10.0, gt=0.0)

    @classmethod
    def from_env(cls, prefix: str = "LID_") -> "LanguageIdConfig":
        overrides = {
            name: os.environ[f"{prefix}{name.upper()}"]
            for name in cls.model_fields
            if f"{prefix}{name.upper()}" in os.environ
        }
        return cls.model_validate(overrides)


class AppConfig(AbstractAppConfig):
    whisper_model: WhisperModelConfig = Field(default_factory=WhisperModelConfig)
    faster_whisper: FasterWhisperConfig = Field(default_factory=FasterWhisperConfig)
    language_id: LanguageIdConfig = Field(default_factory=LanguageIdConfig.from_env)

    @classmethod
    def from_env(cls) -> "AppConfig":
        # The per-field default_factory declarations above already build each sub-config
        # (LanguageIdConfig.from_env included), so cls() reads the environment fully.
        return cls()


_config: AppConfig | None = None


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = AppConfig.from_env()
    return _config


# Module-level aliases for the two sub-configs read at IMPORT time to build pydantic model
# schemas (Field defaults / Annotated constraints in models/*, language_id util). Those are
# frozen at class-definition time and can't be DI-injected. Runtime consumers (services,
# container) take config via the DI container / get_config() instead.
faster_whisper_config = get_config().faster_whisper
language_id_config = get_config().language_id
