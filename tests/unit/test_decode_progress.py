"""Progress reporting during the decode phase.

Decode runs concurrently and is fully materialised before ``prepare_audio_segments``
returns, so consuming the returned segments can no longer drive a progress bar: it
finishes in microseconds after all the work is done. The decode phase must therefore
report its own progress as runs complete.
"""

import dataclasses
import threading
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from faster_whisper.transcribe import Segment as FWSegment

from bentoml_faster_whisper.models.progress_response import ProgressResponse
from bentoml_faster_whisper.models.transcription_request import TranscriptionRequest
from bentoml_faster_whisper.service import FasterWhisper
from bentoml_faster_whisper.services.faster_whisper_handler import FasterWhisperHandler
from bentoml_faster_whisper.utils.speech_regions import WHISPER_SAMPLE_RATE, turns_to_language_runs


@dataclasses.dataclass
class _Info:
    language: str = "de"
    language_probability: float = 0.9
    all_language_probs: Any = None
    duration: float = 0.0


class _FakeWhisper:
    """Returns one segment per run, after a barrier so all runs are in flight at once."""

    def __init__(self, parties: int, concurrent: bool) -> None:
        self._barrier = threading.Barrier(parties) if concurrent else None

    def transcribe(self, audio, language=None, vad_filter=False, **options):
        if self._barrier is not None:
            self._barrier.wait(timeout=5)
        segment = FWSegment(
            id=0,
            seek=0,
            start=0.0,
            end=len(audio) / WHISPER_SAMPLE_RATE,
            text=f" run-{language}",
            tokens=[],
            avg_logprob=-0.3,
            compression_ratio=1.1,
            no_speech_prob=0.05,
            words=None,
            temperature=0.0,
        )
        return iter([segment]), _Info(language=language or "de")


def _handler(num_workers: int) -> FasterWhisperHandler:
    model_manager = SimpleNamespace(whisper_config=SimpleNamespace(num_workers=num_workers))
    return FasterWhisperHandler(model_manager=model_manager, diarization=SimpleNamespace())  # type: ignore


def _turns(count: int, span_s: float = 30.0) -> list[tuple[float, float]]:
    return [(i * span_s, i * span_s + span_s / 2) for i in range(count)]


@pytest.mark.parametrize("num_workers", [1, 4])
def test_decode_runs_report_progress_as_they_complete(num_workers: int):
    turns = _turns(8)
    total_s = turns[-1][1] + 5.0
    decoded = np.zeros(int(total_s * WHISPER_SAMPLE_RATE), dtype=np.float32)
    # Consecutive same-language turns collapse into fewer, span-capped runs.
    run_count = len(turns_to_language_runs(turns, ["de"] * len(turns)))
    assert run_count > 1
    whisper = _FakeWhisper(parties=min(run_count, num_workers), concurrent=num_workers > 1)

    fractions: list[float] = []
    segments, info = _handler(num_workers)._decode_language_runs(
        whisper,  # type: ignore
        decoded,
        turns,
        ["de"] * len(turns),
        total_s,
        decode_options={},
        tag_language=False,
        progress_callback=fractions.append,
    )

    # Reported before anything consumes the segments: the decode work is already done here.
    assert len(fractions) == run_count, "expected one progress report per completed decode run"
    assert fractions == sorted(fractions), "progress must not go backwards"
    assert all(0.0 < f <= 1.0 for f in fractions)
    assert fractions[-1] == pytest.approx(1.0)

    # Out-of-order completion must not reorder the timeline or the ids assigned from it.
    produced = list(segments)
    assert len(produced) == run_count
    assert [seg.start for seg in produced] == sorted(seg.start for seg in produced)
    assert [seg.id for seg in produced] == list(range(run_count))
    assert info.duration == pytest.approx(total_s)


class _ProgressReportingHandler:
    """Stub handler that reports decode progress the way the real one now does."""

    def __init__(self, segments: list, info: Any) -> None:
        self._segments = segments
        self._info = info

    def prepare_audio_segments(self, request, diarization_progress_callback=None, decode_progress_callback=None):
        if diarization_progress_callback is not None:
            diarization_progress_callback(1.0)
        if decode_progress_callback is not None:
            decode_progress_callback(0.5)
            decode_progress_callback(1.0)

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


def test_task_progress_advances_past_diarization_during_decode():
    info = SimpleNamespace(
        language="de", duration=100.0, transcription_options=SimpleNamespace(log_prob_threshold=-1.0)
    )
    service = FasterWhisper()
    service.handler = _ProgressReportingHandler([], info)  # type: ignore
    progress = _RecordingProgress()
    service.progress_handler = progress  # type: ignore
    request = TranscriptionRequest.from_dict({"file": "/tmp/example.mp3", "diarization": True, "progress_id": "task-1"})

    service.task_transcribe(**request.model_dump())

    reported = [update.progress for update in progress.updates]
    assert reported == sorted(reported)
    assert any(0.3 < value < 1.0 for value in reported), f"decode phase reported no progress: {reported}"
    assert reported[-1] == pytest.approx(1.0)
