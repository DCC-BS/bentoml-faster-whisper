"""End-to-end regression tests for the diarized transcription pipeline.

Runs the real Whisper model on a bundled audio asset but fakes the pyannote
turns, so the collapse → transcribe → restore → merge path is exercised without
a GPU, HF token, or the diarization pipeline itself. This is also the canary
for the faster-whisper internals the pipeline relies on
(restore_speech_timestamps, TranscriptionInfo dataclass): a dependency bump
that breaks them must fail here, not in production.
"""

from pathlib import Path

import pytest
from faster_whisper.audio import decode_audio

from api_models.enums import ResponseFormat
from api_models.TranscriptionRequest import TranscriptionRequest
from handlers.fast_whipser_handler import FasterWhisperHandler
from helpers.speech_regions import WHISPER_SAMPLE_RATE

LONG_AUDIO = Path("./tests/assets/long_example_audio.mp3")


class _FakeTurn:
    def __init__(self, start: float, end: float, speaker: str):
        self.start = start
        self.end = end
        self.speaker = speaker


@pytest.fixture(scope="module")
def audio_duration() -> float:
    audio = decode_audio(str(LONG_AUDIO), sampling_rate=WHISPER_SAMPLE_RATE)
    return audio.shape[0] / WHISPER_SAMPLE_RATE


@pytest.fixture(scope="module")
def fake_turns(audio_duration) -> list[_FakeTurn]:
    # Two speaker turns with a large removed-silence gap between them. The turns
    # deliberately cut mid-narration so the decoder is tempted to emit a segment
    # across the concatenation seam — the regression this suite guards against.
    return [
        _FakeTurn(0.5, audio_duration * 0.4, "SPEAKER_00"),
        _FakeTurn(audio_duration * 0.6, audio_duration - 0.5, "SPEAKER_01"),
    ]


@pytest.fixture(scope="module")
def diarized_segments(audio_duration, fake_turns):
    handler = FasterWhisperHandler()
    handler.diarization.diarize = lambda *args, **kwargs: iter(fake_turns)

    request = TranscriptionRequest.model_validate(
        {
            "file": LONG_AUDIO,
            "response_format": ResponseFormat.VERBOSE_JSON,
            "diarization": True,
        }
    )
    segments, info = handler.prepare_audio_segments(request)
    return list(segments), info


def _speech_windows(fake_turns) -> list[tuple[float, float]]:
    # SPEECH_PAD_S widens each turn before decoding; boundaries may sit in the pad.
    return [(max(t.start - 0.3, 0.0), t.end + 0.3) for t in fake_turns]


def test_info_reports_original_file_duration(diarized_segments, audio_duration):
    _, info = diarized_segments
    assert info.duration == pytest.approx(audio_duration, abs=0.5), (
        "info.duration must be the original file length, not the collapsed speech length"
    )


def test_segment_boundaries_stay_inside_speech_windows(diarized_segments, fake_turns):
    segments, info = diarized_segments
    windows = _speech_windows(fake_turns)

    assert segments
    for seg in segments:
        assert seg.end > seg.start
        assert seg.end <= info.duration + 0.05, "no timestamp may exceed the file duration"
        assert any(lo - 0.05 <= seg.start and seg.end <= hi + 0.05 for lo, hi in windows), (
            f"segment [{seg.start}, {seg.end}] spans the removed silence gap — seam split failed"
        )


def test_each_speech_window_gets_its_own_speaker(diarized_segments, fake_turns):
    segments, _ = diarized_segments
    windows = _speech_windows(fake_turns)

    speakers_per_window: dict[int, set] = {i: set() for i in range(len(windows))}
    for seg in segments:
        for i, (lo, hi) in enumerate(windows):
            if lo - 0.05 <= seg.start and seg.end <= hi + 0.05:
                speakers_per_window[i].add(seg.speaker)
                break

    assert speakers_per_window[0] == {"SPEAKER_00"}
    assert speakers_per_window[1] == {"SPEAKER_01"}


def test_words_stripped_unless_word_granularity_requested(diarized_segments):
    # Word timestamps are always decoded internally for the speaker merge, but the
    # default (segment) granularity response must not expose them.
    segments, _ = diarized_segments
    assert all(seg.words is None for seg in segments)
