from pathlib import Path

import pytest
from faster_whisper import WhisperModel

from api_models.enums import ResponseFormat
from api_models.output_models import segments_to_response
from config import WhisperModelConfig
from model_manager import WhisperModelManager


class TestFasterWhisperBug:
    @pytest.mark.skip(
        reason="Only for development purposes, this test takes too long to be included in a ci/cd "
        "pipeline"
    )
    def test_transcribe_compare_with_faster_whisper(self):
        # given
        model_name = "large-v3"
        audio_file_name = "../assets/RecordedAudio.wav"

        model = WhisperModel(model_name)
        model_manager = WhisperModelManager(WhisperModelConfig())

        # when
        segments_package, transcription_info_package = model.transcribe(audio_file_name)

        with model_manager.load_model(model_name) as whisper:
            segments_service, transcription_info_service = whisper.transcribe(
                Path(audio_file_name), temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            )

        text_package = segments_to_response(
            segments_package, transcription_info_package, ResponseFormat.TEXT
        )
        text_service = segments_to_response(
            segments_service, transcription_info_service, ResponseFormat.TEXT
        )

        # then
        assert text_package == text_service
        assert transcription_info_package == transcription_info_service
