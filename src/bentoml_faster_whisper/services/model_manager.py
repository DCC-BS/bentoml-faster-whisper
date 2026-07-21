from __future__ import annotations

import logging
import threading
import time

from faster_whisper import WhisperModel

from bentoml_faster_whisper.config import WhisperModelConfig, faster_whisper_config
from bentoml_faster_whisper.utils import metrics

logger = logging.getLogger(__name__)


class WhisperModelProvider:
    """Loads the single served Whisper model once and keeps it resident.

    The model id is fixed to ``faster_whisper_config.default_model_name`` (the only
    model this API serves); device / compute / thread settings come from
    ``WhisperModelConfig``. The first ``get()`` loads the weights under a lock; every
    later ``get()`` returns the same instance. The model is never unloaded, so callers
    may let the lazy ``transcribe()`` generators outlive the calling method without any
    ref-count guard.
    """

    def __init__(self, whisper_config: WhisperModelConfig) -> None:
        self.whisper_config = whisper_config
        self.model_id = faster_whisper_config.default_model_name
        self._lock = threading.Lock()
        self._model: WhisperModel | None = None

    def get(self) -> WhisperModel:
        # Double-checked locking: once loaded, the hot path returns without taking the lock.
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self._load()
        return self._model

    def _load(self) -> WhisperModel:
        logger.debug("Loading model %s", self.model_id)
        start = time.perf_counter()
        model = WhisperModel(
            self.model_id,
            device=self.whisper_config.inference_device,
            device_index=self.whisper_config.device_index,
            compute_type=self.whisper_config.compute_type,
            cpu_threads=self.whisper_config.cpu_threads,
            num_workers=self.whisper_config.num_workers,
        )
        load_duration = time.perf_counter() - start
        metrics.model_load_duration().observe(load_duration)
        metrics.model_loads_total().inc()
        metrics.models_loaded().inc()
        logger.info("Model %s loaded in %.2fs", self.model_id, load_duration)
        return model
