"""Regression tests for the transcription pipeline's memory and concurrency behaviour.

Runs the real Whisper model on the bundled audio assets (no network).
"""

import threading
import time
from pathlib import Path

import pytest

from api_models.enums import ResponseFormat
from api_models.TranscriptionRequest import TranscriptionRequest
from config import WhisperModelConfig
from diarization_service import DiarizationService
from handlers.progress_handler import ProgressHandler
from helpers.utils import get_audio_duration
from model_manager import SelfDisposingModel, WhisperModelManager

SHORT_AUDIO = Path("./tests/assets/example_audio.mp3")
LONG_AUDIO = Path("./tests/assets/long_example_audio.mp3")


@pytest.fixture(scope="module")
def service(faster_whisper_service):
    return faster_whisper_service


def _request(file: Path, **overrides) -> TranscriptionRequest:
    # diarization off: these tests target Whisper resource handling, not the pyannote pipeline.
    params = {"file": file, "response_format": ResponseFormat.JSON, "diarization": False, **overrides}
    return TranscriptionRequest.model_validate(params)


@pytest.mark.model
def test_model_ref_released_after_full_transcription(handler):
    request = _request(SHORT_AUDIO)

    result = handler.transcribe_audio(request)

    assert result is not None
    sdm = handler.model_manager.loaded_models[request.model]
    assert sdm.ref_count == 0, "model ref must return to 0 after a completed transcription"
    assert sdm.model is not None, "model must stay loaded (ttl!=0) for reuse"


@pytest.mark.model
def test_model_ref_held_until_generator_consumed(handler):
    request = _request(LONG_AUDIO)

    segments, _info = handler.prepare_audio_segments(request)
    sdm = handler.model_manager.loaded_models[request.model]

    # The generator is lazy, so the model must stay held while segments remain to be produced.
    next(segments)
    assert sdm.ref_count == 1, "model ref must be held while the generator is alive"

    for _ in segments:
        pass
    assert sdm.ref_count == 0


@pytest.mark.model
def test_generator_close_releases_model_ref(handler):
    request = _request(LONG_AUDIO)

    segments, _info = handler.prepare_audio_segments(request)
    sdm = handler.model_manager.loaded_models[request.model]
    assert sdm.ref_count == 1

    next(segments)
    segments.close()  # simulate client disconnect mid-stream

    assert sdm.ref_count == 0, "closing the generator must release the held model ref"


@pytest.mark.model
def test_streaming_endpoint_releases_ref_after_full_stream(service):
    request = _request(LONG_AUDIO)
    request_params = request.model_dump()

    chunks = list(service.streaming_transcribe(**request_params))

    assert chunks
    assert all(chunk.endswith("\n") and not chunk.startswith("data:") for chunk in chunks)
    sdm = service.handler.model_manager.loaded_models[request.model]
    assert sdm.ref_count == 0


@pytest.mark.model
def test_concurrent_transcriptions_release_model_ref(handler):
    errors: list[BaseException] = []
    results: list[object] = []
    lock = threading.Lock()

    def run():
        try:
            res = handler.transcribe_audio(_request(SHORT_AUDIO))
            with lock:
                results.append(res)
        except BaseException as e:  # noqa: BLE001 - surface any worker failure
            with lock:
                errors.append(e)

    threads = [threading.Thread(target=run) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent transcriptions raised: {errors}"
    assert len(results) == 6
    sdm = handler.model_manager.loaded_models[_request(SHORT_AUDIO).model]
    assert sdm.ref_count == 0, "ref count must net to 0 after concurrent load"


def test_self_disposing_model_concurrent_ref_counting():
    """The ref-counting context manager is correct under heavy concurrent use."""
    loads: list[int] = []

    def load_fn() -> object:
        loads.append(1)
        return object()

    model = SelfDisposingModel("dummy", load_fn=load_fn, ttl=-1)

    def use():
        with model:
            time.sleep(0.005)

    threads = [threading.Thread(target=use) for _ in range(25)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert model.ref_count == 0
    assert len(loads) == 1, "model must be loaded exactly once despite concurrent access"


def test_unload_model_does_not_deadlock(monkeypatch):
    """unload_model() holds WhisperModelManager._lock while SelfDisposingModel.unload() calls
    back into _handle_model_unload(), which acquires the same lock. A non-reentrant Lock hangs
    the calling thread forever. The TTL path never hit this: it unloads from a Timer thread
    that holds no lock.
    """
    manager = WhisperModelManager(WhisperModelConfig(ttl=-1))
    monkeypatch.setattr(manager, "_load_fn", lambda model_id: object())

    with manager.load_model("dummy"):
        pass

    unloaded = threading.Event()

    def unload():
        manager.unload_model("dummy")
        unloaded.set()

    # daemon so a regression hangs this test rather than the whole interpreter.
    threading.Thread(target=unload, daemon=True).start()

    assert unloaded.wait(timeout=10), "unload_model() deadlocked re-acquiring the manager lock"
    assert "dummy" not in manager.loaded_models


def test_progress_handler_instances_are_isolated():
    a = ProgressHandler()
    b = ProgressHandler()

    a.add_progress("task-a")

    assert "task-a" in a.progress_dict
    assert "task-a" not in b.progress_dict, "progress state must not leak across instances"


@pytest.mark.model
def test_task_transcribe_removes_progress_entry(service):
    progress_id = "resource-mgmt-test"
    params = _request(SHORT_AUDIO, progress_id=progress_id).model_dump()

    service.task_transcribe(**params)

    assert progress_id not in service.progress_handler.progress_dict, (
        "progress entry must be removed after the task completes"
    )


def test_task_transcribe_removes_progress_entry_on_prepare_failure(service, monkeypatch):
    progress_id = "resource-mgmt-prepare-failure"

    def boom(*args, **kwargs):
        raise RuntimeError("prepare_audio_segments failed")

    monkeypatch.setattr(service.handler, "prepare_audio_segments", boom)

    params = _request(SHORT_AUDIO, progress_id=progress_id).model_dump()

    with pytest.raises(RuntimeError, match="prepare_audio_segments failed"):
        service.task_transcribe(**params)

    assert progress_id not in service.progress_handler.progress_dict, (
        "progress entry must be removed when prepare_audio_segments raises"
    )


def test_diarization_batch_sizes_from_env(monkeypatch):
    monkeypatch.setenv("DIARIZATION_SEGMENTATION_BATCH_SIZE", "2")
    monkeypatch.setenv("DIARIZATION_EMBEDDING_BATCH_SIZE", "1")

    svc = DiarizationService()

    assert svc._segmentation_batch_size == 2
    assert svc._embedding_batch_size == 1


def test_diarization_batch_sizes_default():
    svc = DiarizationService()

    assert svc._segmentation_batch_size == 4
    assert svc._embedding_batch_size == 4


def test_get_audio_duration_matches_ffprobe():
    duration = get_audio_duration(LONG_AUDIO)

    # long_example_audio.mp3 is ~96.4s.
    assert duration == pytest.approx(96.4, abs=1.0)


def test_get_audio_duration_missing_file_returns_zero():
    assert get_audio_duration(Path("/does/not/exist.mp3")) == 0.0
