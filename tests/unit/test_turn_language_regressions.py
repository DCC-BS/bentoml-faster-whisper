"""Regression: short turns must not be decoded in a language their speaker isn't using.

Found in the manual hallucination review (2026-09-23): two stretches of real speech
come out as gibberish because their turn was decoded in the wrong language.

- teams_konferenz.mp4, ~1256.9-1258.2 s: SPEAKER_07 speaks German before and after
  ("... wenn es eines geben würde?" / "Nein, im Gegenteil."). Whisper LID on this
  1.3 s turn alone says fr (0.71) and the smoothing keeps it, so the question
  "Was würdest du begrüssen?" is decoded as "Parce que le resto peut être brûlissant."
  Forced to German the same audio decodes as "Das würde es doch begrüssen."
- lichtenstein.mp3, ~14.3-21.9 s: the interpreter says one Spanish sentence
  ("Bienvenidos a la oficina de extranjería y pasaportes del principado de
  Liechtenstein."), which pyannote splits into two turns. LID on the second half,
  dense with proper nouns, says de (0.64), so it is decoded as German gibberish
  ("Ich laufe die Diener der S-Bahn-Karriere im Passaport ...").

Diarization is replayed from the turns recorded for the review (both recordings are
internal and gitignored, so the cases skip without them). Real Whisper decode and the
real turn-level LID + Viterbi path (no ``language`` given), hence ``model``.
"""

import json
from pathlib import Path

import pytest
from pyannote.core import Segment as PyannoteSegment

from bentoml_faster_whisper.models.enums import ResponseFormat
from bentoml_faster_whisper.models.transcription_request import TranscriptionRequest
from bentoml_faster_whisper.services.diarization_service import DiarizationSegment

pytestmark = pytest.mark.model

INTERNAL = Path(__file__).resolve().parent.parent / "assets" / "internal"

# (recording, start_s, end_s, expected language, canary words). The window covers the
# misdetected speech; every segment overlapping it must carry the expected language and
# the text must contain each canary (lower-cased substring, spelling-tolerant stems).
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
