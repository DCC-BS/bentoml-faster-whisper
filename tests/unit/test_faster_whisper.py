import json
from pathlib import Path

from fastapi import HTTPException
import pytest
from bentoml.exceptions import InvalidArgument

from api_models.enums import TimestampGranularity, ResponseFormat
from service import FasterWhisper


class TestFasterWhisperTranscribe:

    def test_transcribe_standard_case(self):
        # given
        faster_whisper_service = FasterWhisper()
        file = Path("./tests/assets/example_audio.mp3")

        # when
        transcription = faster_whisper_service.transcribe(file, response_format=ResponseFormat.JSON)

        # then
        assert transcription is not None

    @pytest.mark.parametrize("temperature", [
        [0.3, 0.6],
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        0.0
    ])
    def test_transcribe_temperature(self, temperature):
        # given
        faster_whisper_service = FasterWhisper()
        file = Path("./tests/assets/example_audio.mp3")

        # when
        transcription = faster_whisper_service.transcribe(file,
                                                          temperature=temperature,
                                                          response_format=ResponseFormat.JSON)

        # then
        assert transcription is not None

    @pytest.mark.parametrize("response_format, timestamp_granularities", [
        (ResponseFormat.JSON, []),
        (ResponseFormat.VERBOSE_JSON, [TimestampGranularity.WORD]),
        (ResponseFormat.SRT, []),
        (ResponseFormat.TEXT, []),
        (ResponseFormat.VTT, [])
    ])
    def test_transcribe_response_format(self, response_format, timestamp_granularities):
        # given
        faster_whisper_service = FasterWhisper()
        file = Path("./tests/assets/example_audio.mp3")

        # when
        transcription = faster_whisper_service.transcribe(file,
                                                          response_format=response_format,
                                                          timestamp_granularities=timestamp_granularities)

        # then
        assert transcription is not None

    def test_response_format_verbose_timestamp_granularities_segment(self):
        # given
        faster_whisper_service = FasterWhisper()
        file = Path("./tests/assets/example_audio.mp3")
        response_format = ResponseFormat.VERBOSE_JSON
        timestamp_granularities = [TimestampGranularity.SEGMENT]

        # when/then
        with pytest.raises(InvalidArgument):
            faster_whisper_service.transcribe(file,
                                              response_format=response_format,
                                              timestamp_granularities=timestamp_granularities)

    def test_response_format_verbose_timestamp_granularities_word(self):
        # given
        faster_whisper_service = FasterWhisper()
        file = Path("./tests/assets/example_audio.mp3")
        response_format = ResponseFormat.VERBOSE_JSON
        timestamp_granularities = [TimestampGranularity.WORD]

        # when
        transcription = faster_whisper_service.transcribe(file,
                                          response_format=response_format,
                                          timestamp_granularities=timestamp_granularities)

        # then
        assert json.loads(transcription)["words"] is not None

    @pytest.mark.asyncio
    async def test_transcribe_streaming(self):
        # given
        faster_whisper_service = FasterWhisper()
        file = Path("./tests/assets/example_audio.mp3")
        chunks = []

        # when
        async for chunk in faster_whisper_service.streaming_transcribe(file, response_format=ResponseFormat.JSON):
            chunks.append(chunk)

        # then
        assert chunks is not None

    def test_transcribe_task(self):
        # given
        faster_whisper_service = FasterWhisper()
        file = Path("./tests/assets/example_audio.mp3")

        # when
        transcription = faster_whisper_service.task_transcribe(file, response_format=ResponseFormat.JSON)

        # then
        assert transcription is not None


class TestFasterWhisperTranslate:

    def test_translate_standard_case(self):
        # given
        faster_whisper_service = FasterWhisper()
        file = Path("./tests/assets/example_audio_german.mp3")

        # when
        transcription = faster_whisper_service.translate(file, response_format=ResponseFormat.JSON)

        # then
        assert transcription is not None


class TestFasterWhisperModels:

    def test_get_models_standard_case(self):
        # given
        faster_whisper_service = FasterWhisper()

        # when
        models = faster_whisper_service.get_models()

        # then
        assert models is not None

    def test_model_not_found(self):
        # given
        unknown_model_name = "unknown-model-v1"
        faster_whisper_service = FasterWhisper()

        # when / then
        with pytest.raises(HTTPException):
            faster_whisper_service.get_model(unknown_model_name)
