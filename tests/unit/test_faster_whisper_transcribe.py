import json
from pathlib import Path

import pytest
from bentoml.exceptions import InvalidArgument

from api_models.enums import ResponseFormat, TimestampGranularity
from api_models.TranscriptionRequest import TranscriptionRequest

pytestmark = pytest.mark.model


def _extend_params(**params):
    # diarization off by default: these tests target the Whisper path, not the pyannote pipeline.
    params.setdefault("diarization", False)
    return TranscriptionRequest.model_validate(params).model_dump()


def test_transcribe_standard_case(faster_whisper_service):
    # given
    file = Path("./tests/assets/example_audio.mp3")

    # when
    transcription = faster_whisper_service.transcribe(**_extend_params(file=file, response_format=ResponseFormat.JSON))

    # then
    assert transcription is not None


@pytest.mark.parametrize("temperature", [[0.3, 0.6], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], 0.0])
def test_transcribe_temperature(faster_whisper_service, temperature):
    # given
    file = Path("./tests/assets/example_audio.mp3")

    # when
    transcription = faster_whisper_service.transcribe(
        **_extend_params(file=file, temperature=temperature, response_format=ResponseFormat.JSON)
    )

    # then
    assert transcription is not None


@pytest.mark.parametrize(
    "response_format, timestamp_granularities",
    [
        (ResponseFormat.JSON, []),
        (ResponseFormat.VERBOSE_JSON, [TimestampGranularity.WORD]),
        (ResponseFormat.SRT, []),
        (ResponseFormat.TEXT, []),
        (ResponseFormat.VTT, []),
    ],
)
def test_transcribe_response_format(faster_whisper_service, response_format, timestamp_granularities):
    # given
    file = Path("./tests/assets/example_audio.mp3")

    params = _extend_params(
        file=file,
        response_format=response_format,
        timestamp_granularities=timestamp_granularities,
    )

    # when
    transcription = faster_whisper_service.transcribe(**params)

    # then
    assert transcription is not None


def test_response_format_verbose_timestamp_granularities_segment(
    faster_whisper_service,
):
    # given
    file = Path("./tests/assets/example_audio.mp3")
    response_format = ResponseFormat.VERBOSE_JSON
    timestamp_granularities = [TimestampGranularity.SEGMENT]

    # when/then
    with pytest.raises(InvalidArgument):
        faster_whisper_service.transcribe(
            **_extend_params(
                file=file,
                response_format=response_format,
                timestamp_granularities=timestamp_granularities,
            )
        )


def test_response_format_verbose_timestamp_granularities_word(faster_whisper_service):
    # given
    file = Path("./tests/assets/example_audio.mp3")
    response_format = ResponseFormat.VERBOSE_JSON
    timestamp_granularities = [TimestampGranularity.WORD]

    # when
    transcription = faster_whisper_service.transcribe(
        **_extend_params(
            file=file,
            response_format=response_format,
            timestamp_granularities=timestamp_granularities,
        )
    )

    # then
    assert json.loads(transcription)["words"] is not None


@pytest.mark.parametrize("file_name", ["head60.mp3", "head23.mp3"])
def test_transcribe_does_not_drop_leading_speech(faster_whisper_service, file_name):
    """Regression: head60.mp3's first 30s window decoded confidently but carried
    no_speech_prob > 0.9, and the cleaner silently dropped the first 23 seconds.
    head23.mp3 (the same first 23 seconds as its own file) is the control."""
    # given
    file = Path("./tests/assets") / file_name

    # when
    transcription = faster_whisper_service.transcribe(**_extend_params(file=file, response_format=ResponseFormat.JSON))

    # then: this sentence is spoken in the first seconds of both files
    assert "mein name ist" in json.loads(transcription)["text"].lower()


class _FakeTurn:
    def __init__(self, start: float, end: float, speaker: str):
        self.start = start
        self.end = end
        self.speaker = speaker


# Real pyannote output for head60.mp3, replayed so the diarized path needs no HF token.
_HEAD60_TURNS = [
    _FakeTurn(0.03, 3.95, "SPEAKER_00"),
    _FakeTurn(4.60, 8.79, "SPEAKER_00"),
    _FakeTurn(9.68, 18.04, "SPEAKER_00"),
    _FakeTurn(18.24, 23.99, "SPEAKER_00"),
    _FakeTurn(24.80, 36.19, "SPEAKER_00"),
    _FakeTurn(36.75, 59.92, "SPEAKER_00"),
]


@pytest.mark.parametrize(
    "diarization, vad_filter, response_format",
    [
        (False, False, ResponseFormat.TEXT),
        (True, True, ResponseFormat.JSON_DIARZED),
    ],
)
def test_transcribe_keeps_leading_speech_across_pipelines(
    faster_whisper_service, monkeypatch, diarization, vad_filter, response_format
):
    """Same regression as test_transcribe_does_not_drop_leading_speech, across pipeline
    variants: the diarized decode with json_diarized output and the plain no-VAD path
    lost the first 23 seconds of head60.mp3 the same way."""
    # given
    file = Path("./tests/assets/head60.mp3")
    if diarization:
        monkeypatch.setattr(
            faster_whisper_service.handler.diarization,
            "diarize",
            lambda *args, **kwargs: iter(_HEAD60_TURNS),
        )

    # when
    transcription = faster_whisper_service.transcribe(
        **_extend_params(
            file=file,
            diarization=diarization,
            vad_filter=vad_filter,
            response_format=response_format,
        )
    )

    # then
    if response_format == ResponseFormat.JSON_DIARZED:
        text = " ".join(segment["text"] for segment in json.loads(transcription)["segments"])
    else:
        text = transcription
    assert "mein name ist" in text.lower()


def test_transcribe_after_long_leading_silence(faster_whisper_service):
    """silence_audio.m4a starts with ~71s of silence: the speech after it must be
    transcribed, and no hallucinated text may land inside the silence."""
    # given
    file = Path("./tests/assets/silence_audio.m4a")

    # when
    transcription = faster_whisper_service.transcribe(
        **_extend_params(
            file=file,
            response_format=ResponseFormat.VERBOSE_JSON,
            timestamp_granularities=[TimestampGranularity.WORD],
        )
    )

    # then
    segments = json.loads(transcription)["segments"]
    assert segments, "the speech after the leading silence must be transcribed"
    assert segments[0]["start"] >= 60, "no text may fall inside the leading ~71s of silence"


def test_transcribe_streaming(faster_whisper_service):
    # given
    file = Path("./tests/assets/example_audio.mp3")
    chunks = []

    # when
    for chunk in faster_whisper_service.streaming_transcribe(
        **_extend_params(file=file, response_format=ResponseFormat.JSON)
    ):
        chunks.append(chunk)

    # then
    assert chunks, "streaming must emit at least one chunk"
    for chunk in chunks:
        # NDJSON contract: bare, newline-delimited JSON payloads with no SSE framing.
        assert chunk.endswith("\n"), f"chunk must be newline-delimited, got {chunk!r}"
        assert not chunk.startswith("data:"), f"chunk must not carry SSE 'data:' framing, got {chunk!r}"
        payload = json.loads(chunk)  # each line must be valid, self-contained JSON
        assert "text" in payload


def test_transcribe_task(faster_whisper_service):
    # given
    file = Path("./tests/assets/example_audio.mp3")

    # when
    transcription = faster_whisper_service.task_transcribe(
        **_extend_params(file=file, response_format=ResponseFormat.JSON)
    )

    # then
    assert transcription is not None
