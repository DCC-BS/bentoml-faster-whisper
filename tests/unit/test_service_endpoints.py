"""Endpoint-level tests that drive the service methods with a stub handler, so no
Whisper model is loaded. They cover behaviour that lives in service.py itself:
streaming applying the same cleaning as the other endpoints, and the task
progress bar surviving a zero-duration transcription_info.
"""

from types import SimpleNamespace
from typing import Any, Optional

import pytest

from bentoml_faster_whisper.models.progress_response import ProgressResponse
from bentoml_faster_whisper.models.transcription_request import TranscriptionRequest
from bentoml_faster_whisper.service import FasterWhisper
from bentoml_faster_whisper.utils.core import Segment


def _segment(**overrides: Any) -> Segment:
    defaults: dict[str, Any] = dict(
        id=0,
        seek=0,
        start=0.0,
        end=1.0,
        text=" Hello world.",
        tokens=[],
        temperature=0.0,
        avg_logprob=-0.3,
        compression_ratio=1.1,
        no_speech_prob=0.05,
        words=None,
    )
    return Segment(**{**defaults, **overrides})


def _info(language: str = "de", duration: float = 10.0, log_prob_threshold: Optional[float] = -1.0):
    return SimpleNamespace(
        language=language,
        duration=duration,
        transcription_options=SimpleNamespace(log_prob_threshold=log_prob_threshold),
    )


class _StubHandler:
    """Stands in for FasterWhisperHandler: returns canned segments without a model."""

    def __init__(self, segments: list[Segment], info: Any):
        self._segments = segments
        self._info = info

    def prepare_audio_segments(self, request, diarization_progress_callback=None, decode_progress_callback=None):
        def gen():
            yield from self._segments

        return gen(), self._info


class _RecordingProgress:
    def __init__(self) -> None:
        self.updates: list[ProgressResponse] = []

    def add_progress(self, id: str) -> None:
        pass

    def update_progress(self, id: str, progress: ProgressResponse) -> None:
        self.updates.append(progress)

    def remove_progress(self, id: str) -> None:
        pass


def _service(handler: _StubHandler) -> Any:
    service = FasterWhisper()
    service.handler = handler  # type: ignore
    return service


def test_streaming_drops_hallucinations_like_other_endpoints():
    segments = [
        _segment(text=" Hello world."),
        _segment(id=1, text=" Untertitel der Amara.org-Community"),  # known de hallucination
        _segment(id=2, text=" Bye.", no_speech_prob=0.95, avg_logprob=-2.0),  # silence
    ]
    service = _service(_StubHandler(segments, _info(language="de")))
    request = TranscriptionRequest.from_dict(
        {"file": "/tmp/example.mp3", "diarization": False, "response_format": "text"}
    )

    output = "".join(service.streaming_transcribe(**request.model_dump()))

    assert "Hello world." in output
    assert "Amara.org" not in output
    assert "Bye." not in output


def test_task_progress_survives_zero_duration():
    progress = _RecordingProgress()
    service = _service(_StubHandler([_segment(end=1.0)], _info(duration=0.0)))
    service.progress_handler = progress
    request = TranscriptionRequest.from_dict(
        {"file": "/tmp/example.mp3", "diarization": False, "progress_id": "task-1"}
    )

    # Would raise ZeroDivisionError without the guard.
    result = service.task_transcribe(**request.model_dump())

    assert isinstance(result, str)
    assert progress.updates, "expected at least one progress update"
    assert progress.updates[-1].progress == pytest.approx(0.3)
