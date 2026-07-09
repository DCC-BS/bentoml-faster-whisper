import pytest
import torch

from config import WhisperModelConfig
from handlers.fast_whipser_handler import FasterWhisperHandler
from model_manager import WhisperModelManager
from service import FasterWhisper

_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


# pyannote model checkpoints contain non-tensor objects incompatible with
# torch 2.6+ default of weights_only=True
torch.load = _patched_torch_load  # ty: ignore[invalid-assignment]


@pytest.fixture(scope="session")
def model_manager():
    """One large-v2 copy for the whole suite.

    ttl=-1 keeps the model resident and, crucially, stops _decrement_ref from arming a
    threading.Timer. Those timers are non-daemon, so the default ttl=300 makes the
    interpreter block for five minutes in threading._shutdown() after the last test.
    """
    manager = WhisperModelManager(WhisperModelConfig(ttl=-1))
    yield manager
    for model_name, model in list(manager.loaded_models.items()):
        # load_model() registers the entry before the weights are read, so a failed or
        # never-entered load leaves model=None, which unload() rejects.
        if model.model is not None:
            manager.unload_model(model_name)


@pytest.fixture(scope="session")
def handler(model_manager) -> FasterWhisperHandler:
    handler = FasterWhisperHandler()
    handler.model_manager = model_manager
    return handler


@pytest.fixture(scope="session")
def faster_whisper_service(handler):
    # bentoml.service() rebinds FasterWhisper to a Service[FasterWhisper] value, so it cannot
    # be used as a return annotation here.
    service = FasterWhisper()
    service.handler = handler
    return service
