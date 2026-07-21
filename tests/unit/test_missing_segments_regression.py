"""Regression: a long single-language diarized file must not drop whole speech
regions during the collapsed decode.

The committed Regionaljournal broadcast (~25 min Swiss-German radio) collapses,
via pyannote's speech turns, into a handful of very long contiguous speech
intervals (the longest ~17 min). Handing such a block to a single
``whisper.transcribe()`` call triggers Whisper's long-form seek drift: whole
30-second windows get skipped and their segments never appear in the output.

Concretely, three passages around 150-167 s vanished from the response even
though pyannote reported continuous speech there and the same audio transcribes
cleanly when decoded as a shorter clip. The canary words below occur *only* in
those dropped passages, so their absence is an unambiguous signal that the drift
happened.

Diarization is replayed from a committed fixture (real pyannote output for this
file) so the test is deterministic and needs neither a GPU-heavy pipeline nor an
HF token — but it still runs the real Whisper decode, hence the ``model`` marker.
"""

import json
from pathlib import Path

import pytest
from pyannote.core import Segment as PyannoteSegment

from bentoml_faster_whisper.models.enums import ResponseFormat
from bentoml_faster_whisper.models.transcription_request import TranscriptionRequest
from bentoml_faster_whisper.services.diarization_service import DiarizationSegment

pytestmark = pytest.mark.model

ASSETS = Path(__file__).resolve().parent.parent / "assets"
AUDIO = ASSETS / "Regionaljournal_Basel_Baselland_radio_AUDI20260710_NR_0011_1004cf68be404710b30a996ac4f1ff93.mp3"
TURNS = ASSETS / "regionaljournal_turns.json"

# Words that appear only in the passages Whisper drops when the whole file is
# decoded as one contiguous block (150-167 s). Each transcribes reliably when the
# region is decoded on its own, so a missing canary means a dropped speech region.
CANARY_WORDS = ["Besorgnis", "fraglich", "vorprogrammiert"]

# The pyannote turns overlapping this window are continuous speech with no gap
# wider than ~0.5 s, so the emitted transcript must not leave a multi-second hole.
CONTINUOUS_SPEECH_WINDOW = (145.0, 172.0)
MAX_ALLOWED_GAP_S = 4.0


def _replay_turns() -> list[DiarizationSegment]:
    raw = json.loads(TURNS.read_text())
    return [DiarizationSegment(PyannoteSegment(start, end), speaker) for start, end, speaker in raw]


@pytest.fixture(scope="module")
def diarized_segments(handler):
    request = TranscriptionRequest.model_validate(
        {
            "file": AUDIO,
            "language": "de",
            "diarization": True,
            "vad_filter": False,
            "response_format": ResponseFormat.JSON_DIARZED,
            "timestamp_granularities": ["segment"],
        }
    )
    turns = _replay_turns()
    # handler is session-scoped and shared, so the stub must not outlive this transcription.
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(handler.diarization, "diarize", lambda *args, **kwargs: iter(turns))
        return json.loads(handler.transcribe_audio(request))["segments"]


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

    max_gap = 0.0
    for (_, prev_end), (next_start, _) in zip(in_window, in_window[1:]):
        max_gap = max(max_gap, next_start - prev_end)

    assert max_gap <= MAX_ALLOWED_GAP_S, (
        f"{max_gap:.1f}s gap between consecutive segments in a continuous-speech window "
        f"(max allowed {MAX_ALLOWED_GAP_S}s) — a speech region was dropped"
    )
