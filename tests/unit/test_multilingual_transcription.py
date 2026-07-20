"""Per-speech-region language detection: a diarized request without an explicit
language must transcribe each speech region in its own detected language.

multilingual_de_en.mp3 is head23.mp3 (German, ~23s), 2s of silence, then
jfk_real_speech.wav (English, ~11s). Speaker turns are replayed instead of
running pyannote; the 2s gap keeps the two regions from merging into one.
"""

import json
from pathlib import Path

import pytest

from api_models.enums import ResponseFormat
from api_models.TranscriptionRequest import TranscriptionRequest
from handlers.fast_whipser_handler import FasterWhisperHandler

AUDIO = Path("./tests/assets/multilingual_de_en.mp3")
LANGUAGE_BOUNDARY_S = 24.0  # inside the silence gap between the German and English halves

pytestmark = pytest.mark.model


class _FakeTurn:
    def __init__(self, start: float, end: float, speaker: str):
        self.start = start
        self.end = end
        self.speaker = speaker


_TURNS = [_FakeTurn(0.3, 22.9, "SPEAKER_00"), _FakeTurn(25.5, 35.5, "SPEAKER_01")]


@pytest.fixture(scope="module")
def diarizing_handler(handler):
    # handler is session-scoped and shared, so the stub has to be undone before the next module.
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(handler.diarization, "diarize", lambda *args, **kwargs: iter(_TURNS))
    yield handler
    monkeypatch.undo()


def _transcribe(handler: FasterWhisperHandler, **overrides) -> dict:
    request = TranscriptionRequest.model_validate({"file": AUDIO, "diarization": True, **overrides})
    response = handler.transcribe_audio(request)
    assert isinstance(response, str)  # json response formats serialize to a JSON string
    return json.loads(response)


@pytest.fixture(scope="module")
def verbose_response(diarizing_handler) -> dict:
    return _transcribe(
        diarizing_handler,
        response_format=ResponseFormat.VERBOSE_JSON,
        timestamp_granularities=["word"],
    )


def test_each_region_transcribed_in_its_language(verbose_response):
    german = [s for s in verbose_response["segments"] if s["end"] <= LANGUAGE_BOUNDARY_S]
    english = [s for s in verbose_response["segments"] if s["start"] >= LANGUAGE_BOUNDARY_S]

    assert german and english, f"expected segments on both sides of the gap: {verbose_response['segments']}"
    assert all(s["language"] == "de" for s in german)
    assert all(s["language"] == "en" for s in english)
    assert "mein name ist" in " ".join(s["text"] for s in german).lower()
    assert "country" in " ".join(s["text"] for s in english).lower()


def test_top_level_language_is_majority_by_speech_time(verbose_response):
    # ~23s German vs ~11s English
    assert verbose_response["language"] == "de"


def test_json_diarized_carries_segment_language(diarizing_handler):
    response = _transcribe(diarizing_handler, response_format=ResponseFormat.JSON_DIARZED)

    languages = {segment["language"] for segment in response["segments"]}
    assert languages == {"de", "en"}


def test_explicit_language_disables_per_segment_detection(diarizing_handler):
    response = _transcribe(diarizing_handler, language="de", response_format=ResponseFormat.JSON_DIARZED)

    assert response["segments"], "explicit-language transcription must still produce segments"
    assert all(segment["language"] is None for segment in response["segments"])


def test_lid_fallback_detects_one_language_over_all_speech(diarizing_handler, monkeypatch):
    """When no turn is long enough for per-turn LID, the handler falls back to a single
    detect_language() over the collapsed speech and tags every segment with that language.

    This is the only path that consumes the full-file collapsed audio, so it guards the
    collapse plumbing in prepare_audio_segments / _transcribe_language_runs end to end.
    """
    from handlers import fast_whipser_handler as handler_module

    # Force every per-turn detection to be indeterminate so _transcribe_language_runs takes
    # its detect_language(collapsed) fallback instead of the per-turn Viterbi path.
    monkeypatch.setattr(handler_module, "detect_turn_language_probs", lambda *a, **k: [None, None])
    monkeypatch.setattr(handler_module, "fill_missing_rows_from_intervals", lambda whisper, decoded, turns, rows: rows)

    response = _transcribe(
        diarizing_handler,
        response_format=ResponseFormat.VERBOSE_JSON,
        timestamp_granularities=["word"],
    )

    assert response["segments"], "fallback path must still produce segments"
    # Audio is majority German (~23s) over English (~11s): one language detected for the whole file.
    assert response["language"] == "de"
    assert {s["language"] for s in response["segments"]} == {"de"}
