"""Regression: decode-run ends must not produce outro hallucinations.

OWUIOberflaeche.wav has no outro phrases spoken; production transcribed "Bis zum nächsten
Mal." / "Tschüss." in the 164.9-167.1 s pause. See docs/technical_architecture.md
("Run-Tail Hallucinations"). Diarization is replayed from a fixture so the run layout is
deterministic.
"""

import functools
import json
from pathlib import Path

import pytest
from pyannote.core import Segment as PyannoteSegment

from bentoml_faster_whisper.models.enums import ResponseFormat
from bentoml_faster_whisper.models.transcription_request import TranscriptionRequest
from bentoml_faster_whisper.services import faster_whisper_handler
from bentoml_faster_whisper.services.diarization_service import DiarizationSegment
from bentoml_faster_whisper.utils.speech_regions import turns_to_language_runs

pytestmark = pytest.mark.model

ASSETS = Path(__file__).resolve().parent.parent / "assets"
AUDIO = ASSETS / "OWUIOberflaeche.wav"
TURNS = ASSETS / "owui_turns.json"

OUTRO_PHRASES = [
    "Bis zum nächsten Mal",
    "Tschüss",
    "Das war's",
    "Zuschauen",
]

MIN_PAUSE_S = 1.0


def _replay_turns() -> list[DiarizationSegment]:
    raw = json.loads(TURNS.read_text())
    return [DiarizationSegment(PyannoteSegment(start, end), speaker) for start, end, speaker in raw]


def _pauses(turns: list[DiarizationSegment]) -> list[tuple[float, float]]:
    spans = sorted((t.start, t.end) for t in turns)
    pauses = []
    reach = spans[0][1]
    for start, end in spans[1:]:
        if start - reach >= MIN_PAUSE_S:
            pauses.append((reach, start))
        reach = max(reach, end)
    return pauses


# 30 s puts a run end into the 164.9 s pause, the layout seen in production.
@pytest.fixture(scope="module", params=[60.0, 30.0], ids=lambda cap: f"max_run_{cap:.0f}s")
def diarized_segments(request, handler) -> list[dict]:
    transcription_request = TranscriptionRequest.model_validate(
        {
            "file": AUDIO,
            "diarization": True,
            "response_format": ResponseFormat.VERBOSE_JSON,
        }
    )
    turns = _replay_turns()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(handler.diarization, "diarize", lambda *args, **kwargs: iter(turns))
        monkeypatch.setattr(
            faster_whisper_handler,
            "turns_to_language_runs",
            functools.partial(turns_to_language_runs, max_run_s=request.param),
        )
        return json.loads(handler.transcribe_audio(transcription_request))["segments"]


def _describe(segment: dict) -> str:
    return (
        f"[{segment['start']:.2f}-{segment['end']:.2f}] {segment['text']!r} "
        f"(no_speech_prob={segment['no_speech_prob']:.3f}, avg_logprob={segment['avg_logprob']:.3f})"
    )


def test_no_outro_hallucinations(diarized_segments):
    hallucinated = [
        _describe(segment)
        for segment in diarized_segments
        if any(phrase.lower() in segment["text"].lower() for phrase in OUTRO_PHRASES)
    ]
    assert not hallucinated, "outro phrases never spoken in the recording were transcribed:\n" + "\n".join(hallucinated)


def test_no_segment_inside_a_speech_pause(diarized_segments):
    pauses = _pauses(_replay_turns())
    assert any(start < 165.0 and end > 167.0 for start, end in pauses), "fixture no longer has the 164.9 s pause"

    in_pause = [
        f"{_describe(segment)} inside pause {start:.2f}-{end:.2f}"
        for segment in diarized_segments
        for start, end in pauses
        if segment["start"] >= start and segment["end"] <= end
    ]
    assert not in_pause, "segments emitted where pyannote heard no speech:\n" + "\n".join(in_pause)


def test_production_gap_has_no_text(diarized_segments):
    between = [
        _describe(segment) for segment in diarized_segments if segment["start"] >= 164.85 and segment["end"] <= 166.8
    ]
    assert not between, "text transcribed in the silent 164.9-166.8 s pause:\n" + "\n".join(between)
