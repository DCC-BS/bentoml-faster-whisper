"""Regression: short turns decoded in a language their speaker isn't using.

Known limitation, pinned as strict xfail; see docs/technical_architecture.md ("Known
Limitation: Short Turns in the Wrong Language"). Internal recordings, skipped when absent.
"""

import json
from pathlib import Path

import pytest
from pyannote.core import Segment as PyannoteSegment

from bentoml_faster_whisper.models.enums import ResponseFormat
from bentoml_faster_whisper.models.transcription_request import TranscriptionRequest
from bentoml_faster_whisper.services.diarization_service import DiarizationSegment

pytestmark = [
    pytest.mark.model,
    pytest.mark.xfail(
        reason="known limitation: Whisper LID misdetects short turns; no penalty/min-turn setting fixes it cleanly",
        strict=True,
    ),
]

INTERNAL = Path(__file__).resolve().parent.parent / "assets" / "internal"

# (recording, start_s, end_s, spoken language, words the correct transcript contains)
CASES = [
    ("teams_konferenz.mp4", 1256.8, 1258.3, "de", ["begrüss"]),
    ("lichtenstein.mp3", 14.3, 21.9, "es", ["oficina", "pasaporte"]),
]


@pytest.fixture(scope="module")
def transcripts(handler):
    cache: dict[str, list[dict]] = {}

    def transcribe(name: str) -> list[dict]:
        if name not in cache:
            audio = INTERNAL / name
            turns_path = INTERNAL / f"{name.replace('.', '_')}_turns.json"
            if not audio.exists() or not turns_path.exists():
                pytest.skip(f"internal asset {audio.name} or its turns fixture not present")
            turns = [
                DiarizationSegment(PyannoteSegment(start, end), speaker)
                for start, end, speaker in json.loads(turns_path.read_text())
            ]
            request = TranscriptionRequest.model_validate(
                {"file": audio, "diarization": True, "response_format": ResponseFormat.VERBOSE_JSON}
            )
            with pytest.MonkeyPatch.context() as monkeypatch:
                monkeypatch.setattr(handler.diarization, "diarize", lambda *args, **kwargs: iter(turns))
                cache[name] = json.loads(handler.transcribe_audio(request))["segments"]
        return cache[name]

    return transcribe


def _in_window(segments: list[dict], start: float, end: float) -> list[dict]:
    return [segment for segment in segments if segment["end"] > start and segment["start"] < end]


@pytest.mark.parametrize(
    ("name", "start", "end", "language", "canaries"), CASES, ids=[f"{c[0]}@{c[1]:.0f}s" for c in CASES]
)
def test_turn_decoded_in_spoken_language(transcripts, name, start, end, language, canaries):
    window = _in_window(transcripts(name), start, end)
    assert window, f"{name} {start}-{end}s: no segment at all over real speech"
    wrong = [(round(s["start"], 2), s["language"], s["text"]) for s in window if s["language"] != language]
    assert not wrong, f"{name} {start}-{end}s: speech in {language!r} decoded in another language: {wrong}"


@pytest.mark.parametrize(
    ("name", "start", "end", "language", "canaries"), CASES, ids=[f"{c[0]}@{c[1]:.0f}s" for c in CASES]
)
def test_turn_text_is_intelligible(transcripts, name, start, end, language, canaries):
    text = " ".join(s["text"] for s in _in_window(transcripts(name), start, end)).lower()
    missing = [word for word in canaries if word not in text]
    assert not missing, f"{name} {start}-{end}s: expected {missing} in the transcript, got {text!r}"
