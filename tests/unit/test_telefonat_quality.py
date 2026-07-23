"""Fuzzy-match the diarized transcription of Telefonat.m4a (two-speaker Swiss German
phone call) against an ElevenLabs reference transcript.

Runs the real Whisper model but replays speaker turns derived from the reference
transcript instead of pyannote, so the decode → collapse → restore → merge → clean
path is scored without a GPU-heavy diarization pipeline or HF token. The same
metrics run against real pyannote in tests/integration/test_telefonat_quality.py.
"""

import json
from pathlib import Path

import pytest

from bentoml_faster_whisper.models.enums import ResponseFormat
from bentoml_faster_whisper.models.transcription_request import TranscriptionRequest
from tests.fuzzy_match import (
    load_reference,
    match_speakers,
    reference_turns,
    reference_words,
    text_similarity,
)

ASSETS = Path(__file__).resolve().parent.parent / "assets"
INTERNAL_ASSETS = ASSETS / "internal"
AUDIO = INTERNAL_ASSETS / "Telefonat.m4a"
REFERENCE_PATH = INTERNAL_ASSETS / "telefonat_transcript.json"
# Both assets are gitignored (internal recording); load lazily so a checkout
# without them still collects this module instead of crashing the whole pytest
# session — pytest.mark.skipif only skips execution, not the module-level import.
REFERENCE = load_reference(REFERENCE_PATH) if REFERENCE_PATH.exists() else None

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(REFERENCE is None, reason=f"internal asset {REFERENCE_PATH} not present"),
]

# Thresholds sit well below the currently measured values (see the integration
# twin for the real-pyannote numbers) so decode jitter doesn't flake the build,
# but far above what a broken pipeline produces.
MIN_TEXT_SIMILARITY = 0.75
MIN_SPEAKER_AGREEMENT = 0.80
MIN_WORD_COVERAGE = 0.85


@pytest.fixture(scope="module")
def diarized_segments(handler):
    reference = REFERENCE
    assert reference is not None  # guaranteed by the module skipif above
    request = TranscriptionRequest.model_validate(
        {"file": AUDIO, "diarization": True, "response_format": ResponseFormat.JSON_DIARIZED}
    )
    # handler is session-scoped and shared, so the stub must not outlive this transcription.
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(handler.diarization, "diarize", lambda *args, **kwargs: iter(reference_turns(reference)))
        return json.loads(handler.transcribe_audio(request))["segments"]


def test_transcript_fuzzy_matches_reference(diarized_segments):
    assert REFERENCE is not None  # guaranteed by the module skipif above
    text = " ".join(segment["text"] for segment in diarized_segments)

    similarity = text_similarity(REFERENCE["text"], text)

    assert similarity >= MIN_TEXT_SIMILARITY, f"text similarity {similarity:.3f}, transcript: {text!r}"


def test_speakers_fuzzy_match_reference(diarized_segments):
    assert REFERENCE is not None  # guaranteed by the module skipif above
    match = match_speakers(
        reference_words(REFERENCE),
        [(segment["start"], segment["end"], segment["speaker"]) for segment in diarized_segments],
    )

    assert match.coverage >= MIN_WORD_COVERAGE, f"only {match.coverage:.1%} of reference words covered"
    assert match.agreement >= MIN_SPEAKER_AGREEMENT, f"speaker agreement {match.agreement:.1%}"
