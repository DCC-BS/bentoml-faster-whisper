"""Translations must be filtered like transcriptions.

``translate_audio`` used to bypass ``clean_transcription_segments``, so blacklisted
subtitle hallucinations and no-speech segments leaked into ``/v1/audio/translations``
output while the same audio through ``/transcriptions`` was filtered. A fake Whisper
model drives the path without loading real weights.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from bentoml_faster_whisper.models.enums import ResponseFormat
from bentoml_faster_whisper.models.translation_request import TranslationRequest
from bentoml_faster_whisper.services.diarization_service import DiarizationService
from bentoml_faster_whisper.services.faster_whisper_handler import FasterWhisperHandler
from bentoml_faster_whisper.services.model_manager import WhisperModelProvider

AUDIO_FILE = Path("./tests/assets/example_audio.mp3")


def _segment(text: str, start: float, end: float, no_speech_prob: float = 0.1, avg_logprob: float = -0.1):
    return SimpleNamespace(
        id=0,
        seek=0,
        start=start,
        end=end,
        text=text,
        tokens=[1],
        temperature=0.0,
        avg_logprob=avg_logprob,
        compression_ratio=1.0,
        no_speech_prob=no_speech_prob,
        words=None,
    )


class _FakeWhisper:
    def __init__(self, segments):
        self._segments = segments

    def transcribe(self, audio, **kwargs):
        info = SimpleNamespace(
            duration=10.0,
            language="en",
            transcription_options=SimpleNamespace(log_prob_threshold=-1.0),
        )
        return iter(self._segments), info


def _handler(segments) -> FasterWhisperHandler:
    model_manager = cast(
        WhisperModelProvider,
        SimpleNamespace(get=lambda: _FakeWhisper(segments), model_id="fake"),
    )
    return FasterWhisperHandler(model_manager=model_manager, diarization=cast(DiarizationService, None))


def _text(response) -> str:
    """JSON response_format always returns a serialized string; narrow the union for the type checker."""
    assert isinstance(response, str)
    return json.loads(response)["text"]


def test_translate_drops_blacklisted_hallucination():
    handler = _handler(
        [
            _segment(" www.mooji.org", 0.0, 2.0),  # en hallucination blacklist entry
            _segment(" hello world", 2.0, 4.0),
        ]
    )
    request = TranslationRequest.model_validate({"file": AUDIO_FILE, "response_format": ResponseFormat.JSON})

    text = _text(handler.translate_audio(request))

    assert "mooji" not in text
    assert "hello world" in text


def test_translate_drops_no_speech_segment():
    handler = _handler(
        [
            # no-speech probe fires AND decode is unconfident (avg_logprob < log_prob_threshold) -> silence
            _segment(" phantom text", 0.0, 2.0, no_speech_prob=0.99, avg_logprob=-2.0),
            _segment(" real speech", 2.0, 4.0),
        ]
    )
    request = TranslationRequest.model_validate({"file": AUDIO_FILE, "response_format": ResponseFormat.JSON})

    text = _text(handler.translate_audio(request))

    assert "phantom" not in text
    assert "real speech" in text
