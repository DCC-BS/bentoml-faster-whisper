"""Full-pipeline twin of tests/unit/test_missing_segments_regression.py: real
pyannote diarization + Whisper on the committed Regionaljournal broadcast.

The unit twin replays committed pyannote turns so it is deterministic and needs
no HF token; this one runs the actual diarization pipeline end-to-end, hence the
integration marker (needs an HF download and a capable GPU). Both assert the same
thing: the long single-language diarized decode must not drop whole speech regions
(see the unit module for the root cause — Whisper long-form drift on an over-long
collapsed block).
"""

import json
from pathlib import Path

import pytest

from api_models.enums import ResponseFormat
from api_models.TranscriptionRequest import TranscriptionRequest
from service import FasterWhisper

pytestmark = pytest.mark.integration

ASSETS = Path(__file__).resolve().parent.parent / "assets"
AUDIO = ASSETS / "Regionaljournal_Basel_Baselland_radio_AUDI20260710_NR_0011_1004cf68be404710b30a996ac4f1ff93.mp3"

# Words that appear only in the passages Whisper drops when the whole file is
# decoded as one contiguous block (around 150-167 s).
CANARY_WORDS = ["Besorgnis", "fraglich", "vorprogrammiert"]

CONTINUOUS_SPEECH_WINDOW = (145.0, 172.0)
MAX_ALLOWED_GAP_S = 4.0


@pytest.fixture(scope="module")
def diarized_segments() -> list[dict]:
    service = FasterWhisper()
    params = TranscriptionRequest.model_validate(
        {
            "file": AUDIO,
            "language": "de",
            "diarization": True,
            "vad_filter": False,
            "response_format": ResponseFormat.JSON_DIARZED,
            "timestamp_granularities": ["segment"],
        }
    ).model_dump()
    return json.loads(service.transcribe(**params))["segments"]


@pytest.mark.parametrize("word", CANARY_WORDS)
def test_dropped_speech_region_is_transcribed(diarized_segments, word):
    transcript = " ".join(segment["text"] for segment in diarized_segments)
    assert word in transcript, (
        f"canary word {word!r} missing — a speech region around 150-167 s was dropped during the collapsed decode"
    )


def test_no_multi_second_gap_in_continuous_speech(diarized_segments):
    lo, hi = CONTINUOUS_SPEECH_WINDOW
    in_window = sorted((s["start"], s["end"]) for s in diarized_segments if s["end"] > lo and s["start"] < hi)
    assert in_window, "no segments at all in a window pyannote marked as continuous speech"

    max_gap = max((nxt - prev for (_, prev), (nxt, _) in zip(in_window, in_window[1:])), default=0.0)

    assert max_gap <= MAX_ALLOWED_GAP_S, (
        f"{max_gap:.1f}s gap between consecutive segments in a continuous-speech window "
        f"(max allowed {MAX_ALLOWED_GAP_S}s) — a speech region was dropped"
    )
