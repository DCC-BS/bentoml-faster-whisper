import pytest
from faster_whisper import WhisperModel

from bentoml_faster_whisper.config import WhisperModelConfig, faster_whisper_config
from bentoml_faster_whisper.models.enums import ResponseFormat
from bentoml_faster_whisper.models.output_models import segments_to_response
from bentoml_faster_whisper.services.model_manager import WhisperModelProvider
from bentoml_faster_whisper.utils.core import Segment


class TestFasterWhisperBug:
    @pytest.mark.skip(
        reason="Only for development purposes, this test takes too long to be included in a ci/cd pipeline"
    )
    def test_transcribe_compare_with_faster_whisper(self):
        # given
        model_name = faster_whisper_config.default_model_name
        audio_file_name = "../assets/RecordedAudio.wav"

        model = WhisperModel(model_name)
        provider = WhisperModelProvider(WhisperModelConfig())

        # when
        segments_package, transcription_info_package = model.transcribe(audio_file_name)

        whisper = provider.get()
        segments_service, transcription_info_service = whisper.transcribe(
            str(audio_file_name), temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        )

        text_package = segments_to_response(
            list(Segment.from_faster_whisper_segments(segments_package)),
            transcription_info_package,
            ResponseFormat.TEXT,
        )
        text_service = segments_to_response(
            list(Segment.from_faster_whisper_segments(segments_service)),
            transcription_info_service,
            ResponseFormat.TEXT,
        )

        # then
        assert text_package == text_service
        assert transcription_info_package == transcription_info_service
