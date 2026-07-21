"""Full-pipeline quality test: real pyannote diarization + Whisper on Telefonat.m4a,
fuzzy-matched against an ElevenLabs reference transcript (text, speakers, timestamps).

The unit twin (tests/unit/test_telefonat_quality.py) replays reference-derived
speaker turns instead of pyannote; this one needs the pyannote pipeline (HF download)
and a capable GPU, hence the integration marker.
"""

import json
from pathlib import Path

import pytest

from bentoml_faster_whisper.models.enums import ResponseFormat
from bentoml_faster_whisper.models.transcription_request import TranscriptionRequest
from bentoml_faster_whisper.service import FasterWhisper
from tests.fuzzy_match import (
    load_reference,
    match_speakers,
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
    pytest.mark.integration,
    pytest.mark.skipif(REFERENCE is None, reason=f"internal asset {REFERENCE_PATH} not present"),
]

MIN_TEXT_SIMILARITY = 0.75
MIN_SPEAKER_AGREEMENT = 0.80
MIN_WORD_COVERAGE = 0.85


@pytest.fixture(scope="module")
def diarized_segments() -> list[dict]:
    service = FasterWhisper()
    params = TranscriptionRequest.model_validate(
        {"file": AUDIO, "diarization": True, "response_format": ResponseFormat.JSON_DIARZED}
    ).model_dump()
    return json.loads(service.transcribe(**params))["segments"]


def test_transcript_fuzzy_matches_reference(diarized_segments):
    assert REFERENCE is not None  # guaranteed by the module skipif above
    text = " ".join(segment["text"] for segment in diarized_segments)

    similarity = text_similarity(REFERENCE["text"], text)

    assert similarity >= MIN_TEXT_SIMILARITY, f"text similarity {similarity:.3f}, transcript: {text!r}"


def test_diarization_fuzzy_matches_reference(diarized_segments):
    assert REFERENCE is not None  # guaranteed by the module skipif above
    assert len({segment["speaker"] for segment in diarized_segments}) == 2, "the call has exactly two speakers"

    match = match_speakers(
        reference_words(REFERENCE),
        [(segment["start"], segment["end"], segment["speaker"]) for segment in diarized_segments],
    )

    assert match.coverage >= MIN_WORD_COVERAGE, f"only {match.coverage:.1%} of reference words covered"
    assert match.agreement >= MIN_SPEAKER_AGREEMENT, f"speaker agreement {match.agreement:.1%}"
