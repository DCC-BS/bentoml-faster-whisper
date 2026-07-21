from dependency_injector import containers, providers

from bentoml_faster_whisper.config import get_config
from bentoml_faster_whisper.services.diarization_service import DiarizationService
from bentoml_faster_whisper.services.faster_whisper_handler import FasterWhisperHandler
from bentoml_faster_whisper.services.model_manager import WhisperModelProvider
from bentoml_faster_whisper.services.progress_handler import ProgressHandler


class Container(containers.DeclarativeContainer):
    """Dependency injection container for bentoml-faster-whisper."""

    wiring_config = containers.WiringConfiguration(modules=["bentoml_faster_whisper.service"])

    config = providers.Singleton(get_config)

    model_manager = providers.Singleton(
        WhisperModelProvider,
        whisper_config=config.provided.whisper_model,
    )

    diarization_service = providers.Singleton(DiarizationService)

    faster_whisper_handler = providers.Singleton(
        FasterWhisperHandler,
        model_manager=model_manager,
        diarization=diarization_service,
    )

    progress_handler = providers.Singleton(ProgressHandler)
