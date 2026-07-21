"""Regression tests for the transcription pipeline's memory and concurrency behaviour.

Runs the real Whisper model on the bundled audio assets (no network).
"""

import threading
import time
from pathlib import Path

import pytest

from bentoml_faster_whisper.config import WhisperModelConfig
from bentoml_faster_whisper.models.enums import ResponseFormat
from bentoml_faster_whisper.models.transcription_request import TranscriptionRequest
from bentoml_faster_whisper.services.diarization_service import DiarizationService
from bentoml_faster_whisper.services.model_manager import WhisperModelProvider
from bentoml_faster_whisper.services.progress_handler import ProgressHandler
from bentoml_faster_whisper.utils.core import get_audio_duration

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
def test_streaming_endpoint_completes_full_stream(service):
    request = _request(LONG_AUDIO)
    request_params = request.model_dump()

    chunks = list(service.streaming_transcribe(**request_params))

    assert chunks
    assert all(chunk.endswith("\n") and not chunk.startswith("data:") for chunk in chunks)


@pytest.mark.model
def test_concurrent_transcriptions_succeed(handler):
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


def test_provider_loads_model_once_under_concurrency(monkeypatch):
    """The provider loads the model exactly once and hands the same instance to every
    caller, even when many threads race on the first get()."""
    provider = WhisperModelProvider(WhisperModelConfig())
    loads: list[int] = []

    def fake_load() -> object:
        loads.append(1)
        time.sleep(0.005)  # widen the race window
        return object()

    monkeypatch.setattr(provider, "_load", fake_load)

    results: list[object] = []
    lock = threading.Lock()

    def use():
        model = provider.get()
        with lock:
            results.append(model)

    threads = [threading.Thread(target=use) for _ in range(25)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(loads) == 1, "model must be loaded exactly once despite concurrent get()"
    assert results and all(m is results[0] for m in results), "every caller gets the same instance"


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
