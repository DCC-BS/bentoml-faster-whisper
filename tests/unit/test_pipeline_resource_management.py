"""Regression tests for the transcription pipeline's memory and concurrency behaviour.

Runs the real Whisper model on the bundled audio assets (no network).
"""

import threading
import time
from pathlib import Path

import pytest

from api_models.enums import ResponseFormat
from api_models.TranscriptionRequest import TranscriptionRequest
from diarization_service import DiarizationService
from handlers.fast_whipser_handler import FasterWhisperHandler
from handlers.progress_handler import ProgressHandler
from helpers.utils import get_audio_duration
from model_manager import SelfDisposingModel
from service import FasterWhisper

SHORT_AUDIO = Path("./tests/assets/example_audio.mp3")
LONG_AUDIO = Path("./tests/assets/long_example_audio.mp3")


@pytest.fixture(scope="module")
def handler() -> FasterWhisperHandler:
    return FasterWhisperHandler()


@pytest.fixture(scope="module")
def service():
    return FasterWhisper()


def _request(file: Path, **overrides) -> TranscriptionRequest:
    params = {"file": file, "response_format": ResponseFormat.JSON, **overrides}
    return TranscriptionRequest.model_validate(params)


def test_model_ref_released_after_full_transcription(handler):
    request = _request(SHORT_AUDIO)

    result = handler.transcribe_audio(request)

    assert result is not None
    sdm = handler.model_manager.loaded_models[request.model]
    assert sdm.ref_count == 0, "model ref must return to 0 after a completed transcription"
    assert sdm.model is not None, "model must stay loaded (ttl>0) for reuse"


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


def test_generator_close_releases_model_ref(handler):
    request = _request(LONG_AUDIO)

    segments, _info = handler.prepare_audio_segments(request)
    sdm = handler.model_manager.loaded_models[request.model]
    assert sdm.ref_count == 1

    next(segments)
    segments.close()  # simulate client disconnect mid-stream

    assert sdm.ref_count == 0, "closing the generator must release the held model ref"


def test_streaming_endpoint_releases_ref_after_full_stream(service):
    request = _request(LONG_AUDIO)
    request_params = request.model_dump()

    chunks = list(service.streaming_transcribe(**request_params))

    assert chunks
    assert all(chunk.endswith("\n") and not chunk.startswith("data:") for chunk in chunks)
    sdm = service.handler.model_manager.loaded_models[request.model]
    assert sdm.ref_count == 0


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


def test_progress_handler_instances_are_isolated():
    a = ProgressHandler()
    b = ProgressHandler()

    a.add_progress("task-a")

    assert "task-a" in a.progress_dict
    assert "task-a" not in b.progress_dict, "progress state must not leak across instances"


def test_task_transcribe_removes_progress_entry(service):
    progress_id = "resource-mgmt-test"
    params = _request(SHORT_AUDIO, progress_id=progress_id).model_dump()

    service.task_transcribe(**params)

    assert progress_id not in service.progress_handler.progress_dict, (
        "progress entry must be removed after the task completes"
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
