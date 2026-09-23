"""Regression: the end of a decode run must not produce outro hallucinations.

OWUIOberflaeche.wav is a 3 min German screen-cast narration (single speaker) with a
few multi-second pauses. Since diarized decodes are split into runs of at most
``WHISPER_MAX_DECODE_RUN_S`` (the long-form seek drift fix), every run boundary is a
place where Whisper's last window of that run holds only a sub-second tail of
padding/silence, zero-padded to 30 s. Whisper fills it with YouTube-outro phrases
("Das war's für heute.", "Bis zum nächsten Mal.", "Tschüss.", "Vielen Dank fürs
Zuschauen."), none of which are spoken anywhere in the recording. They come out with
``no_speech_prob`` ~0.6 and ``avg_logprob`` ~-0.75, so the dual-condition silence
filter (``> 0.9`` AND ``< -1.0``) lets them through.

Observed in production at 164.9-167.1 s ("Bis zum nächsten Mal." / "Tschüss."
between "... Open Case Law geben." and "Das heisst, hier ..."). With the default
60 s run cap and the replayed pyannote turns below, a run ends at 41.7 s and the same
hallucination appears there; a 30 s cap moves a run boundary into the 164.9 s pause
and reproduces the production transcript.

Diarization is replayed from a committed fixture (real pyannote output for this file)
so the run layout is deterministic; the Whisper decode is real, hence ``model``.
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

# Outro phrases Whisper emits on a near-empty trailing window. None is spoken in the
# recording (checked by ear), so any occurrence is a hallucination.
OUTRO_PHRASES = [
    "Bis zum nächsten Mal",
    "Tschüss",
    "Das war's",
    "Zuschauen",
]

# Pauses between pyannote turns that are silent by ear. Whisper may place a word a
# little into a pause (turn edges are padded), but a segment that lies entirely inside
# one has no speech under it.
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


# 60 s is the production default and ends a run at 41.7 s; 30 s additionally ends a
# run inside the 164.9-167.1 s pause, the layout seen in production.
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
    """The exact production symptom: nothing between '... geben.' (164.8 s) and 'Das heisst' (~166.8 s)."""
    between = [
        _describe(segment) for segment in diarized_segments if segment["start"] >= 164.85 and segment["end"] <= 166.8
    ]
    assert not between, "text transcribed in the silent 164.9-166.8 s pause:\n" + "\n".join(between)
