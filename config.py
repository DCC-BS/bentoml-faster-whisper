import enum
from typing import Dict, List, Union

import torch
from pydantic import BaseModel, Field
from faster_whisper.vad import VadOptions
from api_models.enums import ResponseFormat, TimestampGranularity


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
    compute_type: Quantization = (
        Quantization.FLOAT16 if torch.cuda.is_available() else Quantization.DEFAULT
    )
    cpu_threads: int = 0
    num_workers: int = 1
    ttl: int = Field(default=300, ge=-1)
    """
    Time in seconds until the model is unloaded if it is not being used.
    -1: Never unload the model.
    0: Unload the model immediately after usage.
    """


class FasterWhisperConfig(BaseModel):
    default_model_name: str = "large-v3"
    default_prompt: str = None
    default_language: str = None
    default_response_format: ResponseFormat = ResponseFormat.JSON
    default_temperature: List[float] = [
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
    ]
    default_timestamp_granularities: List[TimestampGranularity] = [
        TimestampGranularity.SEGMENT
    ]

    min_temperature: float = 0.0
    max_temperature: float = 2.0

    best_of: int = 10
    vad_filter: bool = True
    vad_parameters: Dict[str, Union[float, int]] = {
        "onset": 0.5,
        "offset": 0.15,
        "min_speech_duration_ms": 0,
        "max_speech_duration_s": 999_999,
        "min_silence_duration_ms": 2000,
        "speech_pad_ms": 400
    }
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


faster_whisper_config = FasterWhisperConfig()
