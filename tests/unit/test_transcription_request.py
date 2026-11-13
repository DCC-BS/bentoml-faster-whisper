from bentoml_faster_whisper.config import faster_whisper_config
from bentoml_faster_whisper.models.enums import Language
from bentoml_faster_whisper.models.transcription_request import (
    TranscriptionRequest,
    _process_empty_language,
)


def test_invalid_language_falls_back_to_default_without_raising():
    # The shipped default_language is None (auto-detect). An invalid code must fall
    # back to it, not crash on Language(None) — that used to surface as a 422.
    assert faster_whisper_config.default_language is None
    assert _process_empty_language("not-a-language") is None


def test_empty_language_is_none():
    assert _process_empty_language("") is None
    assert _process_empty_language(None) is None


def test_valid_language_is_preserved():
    assert _process_empty_language("de") == Language.DE


def test_request_accepts_invalid_language_and_auto_detects():
    request = TranscriptionRequest.from_dict({"file": "/tmp/example.mp3", "language": "xx", "diarization": False})
    assert request.language is None


def test_voice_relabel_is_on_by_default():
    request = TranscriptionRequest.from_dict({"file": "/tmp/example.mp3"})
    assert request.diarization_voice_relabel is True


def test_voice_relabel_can_be_switched_off_per_request():
    request = TranscriptionRequest.from_dict({"file": "/tmp/example.mp3", "diarization_voice_relabel": "false"})
    assert request.diarization_voice_relabel is False


def test_voice_relabel_default_follows_env(monkeypatch):
    from bentoml_faster_whisper.config import FasterWhisperConfig

    monkeypatch.setenv("DIARIZATION_VOICE_RELABEL", "false")
    assert FasterWhisperConfig().diarization_voice_relabel is False
    monkeypatch.setenv("DIARIZATION_VOICE_RELABEL", "true")
    assert FasterWhisperConfig().diarization_voice_relabel is True
