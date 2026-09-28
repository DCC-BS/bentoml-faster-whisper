"""Regression: segments labelled by ear as hallucination must not come back, and
segments labelled as real speech must still be transcribed (coverage only, not text).

Labels: tests/assets/reviewed_hallucinations.json, plus tests/assets/internal/ (gitignored,
skipped when absent). Diarization is replayed so the run layout is deterministic.
"""

import json
from pathlib import Path

import pytest
from pyannote.core import Segment as PyannoteSegment

from bentoml_faster_whisper.models.enums import ResponseFormat
from bentoml_faster_whisper.models.transcription_request import TranscriptionRequest
from bentoml_faster_whisper.services.diarization_service import DiarizationSegment
from tests.fuzzy_match import text_similarity

pytestmark = pytest.mark.model

ASSETS = Path(__file__).resolve().parent.parent / "assets"
INTERNAL = ASSETS / "internal"

AUDIO = {
    "OWUIOberflaeche.wav": (ASSETS / "OWUIOberflaeche.wav", ASSETS / "owui_turns.json"),
    "Regionaljournal_Basel_Baselland_radio_AUDI20260710_NR_0011_1004cf68be404710b30a996ac4f1ff93.mp3": (
        ASSETS / "Regionaljournal_Basel_Baselland_radio_AUDI20260710_NR_0011_1004cf68be404710b30a996ac4f1ff93.mp3",
        ASSETS / "regionaljournal_turns.json",
    ),
    "brain_dump.mp3": (INTERNAL / "brain_dump.mp3", INTERNAL / "brain_dump_mp3_turns.json"),
    "lichtenstein.mp3": (INTERNAL / "lichtenstein.mp3", INTERNAL / "lichtenstein_mp3_turns.json"),
    "teams_konferenz.mp3": (INTERNAL / "teams_konferenz.mp3", INTERNAL / "teams_konferenz_mp3_turns.json"),
    "teams_konferenz.mp4": (INTERNAL / "teams_konferenz.mp4", INTERNAL / "teams_konferenz_mp4_turns.json"),
    "Telefonat.m4a": (INTERNAL / "Telefonat.m4a", INTERNAL / "Telefonat_m4a_turns.json"),
}

HALLUCINATION_WINDOW_S = 1.5
HALLUCINATION_MIN_SIMILARITY = 0.8

REAL_SPEECH_TOLERANCE_S = 0.3


def _labels() -> list[dict]:
    labels = []
    for path in (ASSETS / "reviewed_hallucinations.json", INTERNAL / "reviewed_hallucinations.json"):
        if path.exists():
            labels.extend(json.loads(path.read_text(encoding="utf-8")))
    return [label for label in labels if label["verdict"] in ("hallucination", "real")]


def _label_id(label: dict) -> str:
    return f"{label['verdict']}-{Path(label['file']).stem[:24]}@{label['start']:.2f}"


LABELS = _labels()

KNOWN_MISSES = {("teams_konferenz.mp4", 3150.04): "indistinguishable from real low-confidence run-end speech"}


def _hallucination_params() -> list:
    params = []
    for label in LABELS:
        if label["verdict"] != "hallucination":
            continue
        reason = KNOWN_MISSES.get((label["file"], label["start"]))
        marks = [pytest.mark.xfail(reason=reason, strict=True)] if reason else []
        params.append(pytest.param(label, id=_label_id(label), marks=marks))
    return params


@pytest.fixture(scope="module")
def transcripts(handler):
    cache: dict[str, list[dict]] = {}

    def transcribe(file: str) -> list[dict]:
        if file not in cache:
            audio, turns_path = AUDIO[file]
            if not audio.exists() or not turns_path.exists():
                pytest.skip(f"asset {audio.name} or its turns fixture not present")
            turns = [
                DiarizationSegment(PyannoteSegment(start, end), speaker)
                for start, end, speaker in json.loads(turns_path.read_text())
            ]
            request = TranscriptionRequest.model_validate(
                {"file": audio, "diarization": True, "response_format": ResponseFormat.VERBOSE_JSON}
            )
            with pytest.MonkeyPatch.context() as monkeypatch:
                monkeypatch.setattr(handler.diarization, "diarize", lambda *args, **kwargs: iter(turns))
                cache[file] = json.loads(handler.transcribe_audio(request))["segments"]
        return cache[file]

    return transcribe


@pytest.mark.parametrize("label", _hallucination_params())
def test_confirmed_hallucination_is_not_transcribed(transcripts, label):
    segments = transcripts(label["file"])
    back = [
        segment
        for segment in segments
        if abs(segment["start"] - label["start"]) <= HALLUCINATION_WINDOW_S
        and text_similarity(segment["text"], label["text"]) >= HALLUCINATION_MIN_SIMILARITY
    ]
    assert not back, (
        f"{label['file']} @ {label['start']:.2f}s: hallucination {label['text']!r} (confirmed by ear) is transcribed "
        f"again: {[(round(s['start'], 2), s['text']) for s in back]}"
    )


@pytest.mark.parametrize("label", [label for label in LABELS if label["verdict"] == "real"], ids=_label_id)
def test_confirmed_real_speech_is_transcribed(transcripts, label):
    segments = transcripts(label["file"])
    lo, hi = label["start"] - REAL_SPEECH_TOLERANCE_S, label["end"] + REAL_SPEECH_TOLERANCE_S
    covering = [segment for segment in segments if segment["end"] > lo and segment["start"] < hi]
    assert covering, (
        f"{label['file']} @ {label['start']:.2f}-{label['end']:.2f}s: real speech {label['text']!r} (confirmed by ear) "
        "is missing from the transcript"
    )
