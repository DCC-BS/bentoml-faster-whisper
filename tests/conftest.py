import pytest
import torch

from bentoml_faster_whisper.config import WhisperModelConfig, faster_whisper_config
from bentoml_faster_whisper.service import FasterWhisper
from bentoml_faster_whisper.services.diarization_service import DiarizationService
from bentoml_faster_whisper.services.faster_whisper_handler import FasterWhisperHandler
from bentoml_faster_whisper.services.model_manager import WhisperModelProvider

_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


# pyannote model checkpoints contain non-tensor objects incompatible with
# torch 2.6+ default of weights_only=True
torch.load = _patched_torch_load  # ty: ignore[invalid-assignment]


@pytest.fixture(scope="session")
def model_manager():
    """One resident large-v2 copy for the whole suite. The provider never unloads,
    so there are no non-daemon unload timers to block interpreter shutdown."""
    return WhisperModelProvider(WhisperModelConfig(), faster_whisper_config.default_model_name)


@pytest.fixture(scope="session")
def handler(model_manager) -> FasterWhisperHandler:
    return FasterWhisperHandler(model_manager=model_manager, diarization=DiarizationService())


@pytest.fixture(scope="session")
def faster_whisper_service(handler):
    # bentoml.service() rebinds FasterWhisper to a Service[FasterWhisper] value, so it cannot
    # be used as a return annotation here.
    service = FasterWhisper()
    service.handler = handler
    return service
