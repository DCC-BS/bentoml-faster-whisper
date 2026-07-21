from bentoml_faster_whisper.container import Container
from bentoml_faster_whisper.services.diarization_service import DiarizationService
from bentoml_faster_whisper.services.faster_whisper_handler import FasterWhisperHandler
from bentoml_faster_whisper.services.model_manager import WhisperModelProvider
from bentoml_faster_whisper.services.progress_handler import ProgressHandler


def test_container_provides_singletons():
    container = Container()

    cfg = container.config()
    assert cfg is not None

    model_mgr = container.model_manager()
    assert isinstance(model_mgr, WhisperModelProvider)
    assert container.model_manager() is model_mgr

    diar_svc = container.diarization_service()
    assert isinstance(diar_svc, DiarizationService)
    assert container.diarization_service() is diar_svc

    fw_handler = container.faster_whisper_handler()
    assert isinstance(fw_handler, FasterWhisperHandler)
    assert container.faster_whisper_handler() is fw_handler

    prog_handler = container.progress_handler()
    assert isinstance(prog_handler, ProgressHandler)
    assert container.progress_handler() is prog_handler
