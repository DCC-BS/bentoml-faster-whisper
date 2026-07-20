"""End-to-end pipeline tests against committed real-world speech audio (no network).

jfk_real_speech.wav is a public-domain JFK inaugural excerpt (~11s, known transcript);
long_example_audio.mp3 (~96s) exercises the larger-file and concurrency paths.
"""

import json
import threading
from pathlib import Path

import pytest

from api_models.enums import ResponseFormat
from api_models.TranscriptionRequest import TranscriptionRequest
from service import FasterWhisper

pytestmark = pytest.mark.integration

ASSETS = Path(__file__).resolve().parent.parent / "assets"
REAL_SPEECH = ASSETS / "jfk_real_speech.wav"
REAL_SPEECH_PHRASE = "country"  # "...ask not what your country can do for you..."
LONG_AUDIO = ASSETS / "long_example_audio.mp3"


@pytest.fixture(scope="module")
def service():
    return FasterWhisper()


def _params(file: Path, **overrides) -> dict:
    # diarization off: these tests target the Whisper path offline; pyannote needs an HF download.
    return TranscriptionRequest.model_validate(
        {"file": file, "response_format": ResponseFormat.TEXT, "diarization": False, **overrides}
    ).model_dump()


def test_real_speech_produces_non_empty_transcript(service):
    text = service.transcribe(**_params(REAL_SPEECH))

    assert isinstance(text, str)
    assert text.strip(), "transcription of real audio must not be empty"


def test_real_speech_transcript_contains_expected_phrase(service):
    text = service.transcribe(**_params(REAL_SPEECH))

    assert REAL_SPEECH_PHRASE in text.lower(), f"expected '{REAL_SPEECH_PHRASE}' in transcript, got: {text!r}"


def test_real_speech_verbose_json_has_words(service):
    params = TranscriptionRequest.model_validate(
        {
            "file": REAL_SPEECH,
            "response_format": ResponseFormat.VERBOSE_JSON,
            "timestamp_granularities": ["word"],
            "diarization": False,
        }
    ).model_dump()

    result = json.loads(service.transcribe(**params))

    assert result["words"], "verbose_json output must include word-level timestamps"
    for word in result["words"]:
        assert "start" in word and "end" in word, f"word entry missing timestamps: {word!r}"
        assert isinstance(word["start"], (int, float)) and isinstance(word["end"], (int, float))
        assert word["end"] >= word["start"]


def test_long_audio_transcribes_fully(service):
    text = service.transcribe(**_params(LONG_AUDIO))

    assert text.strip(), "long audio must produce a transcript"


def test_concurrent_transcriptions_on_larger_file(service):
    """Concurrent requests on a larger file all succeed and release the model ref."""
    errors: list[BaseException] = []
    results: list[str] = []
    lock = threading.Lock()

    def run():
        try:
            text = service.transcribe(**_params(LONG_AUDIO))
            with lock:
                results.append(text)
        except BaseException as e:  # noqa: BLE001 - surface worker failures
            with lock:
                errors.append(e)

    threads = [threading.Thread(target=run) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent transcriptions raised: {errors}"
    assert len(results) == 4
    assert all(text.strip() for text in results)
