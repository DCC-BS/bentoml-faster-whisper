from http import HTTPStatus

from bentoml.exceptions import BentoMLException

from api_models.enums import ResponseFormat, Task
from api_models.input_models import (
    validate_timestamp_granularities,
)
from api_models.output_models import (
    WhisperResponse,
    segments_to_response,
)
from api_models.TranscriptionRequest import TranscriptionRequest
from config import WhisperModelConfig
from core import Segment
from diarization_service import DiarizationService
from helpers.transcription_cleaner import clean_transcription_segments
from helpers.whiper_diarization_merger import merge_whipser_diarization
from model_manager import WhisperModelManager


class FasterWhisperHandler:
    def __init__(self):
        self.model_manager = WhisperModelManager(WhisperModelConfig())
        self.diarization = DiarizationService()
        self.diarization.load()

    def transcribe_audio(
        self,
        request: TranscriptionRequest,
    ) -> WhisperResponse:
        validate_timestamp_granularities(
            request.response_format,
            request.timestamp_granularities,
            request.diarization,
        )

        segments, transcription_info = self.prepare_audio_segments(request)
        segments = clean_transcription_segments(segments, transcription_info)
        return segments_to_response(segments, transcription_info, request.response_format)

    def translate_audio(
        self,
        file,
        model,
        prompt,
        response_format,
        temperature,
        best_of,
        vad_filter,
        vad_parameters,
        condition_on_previous_text,
        repetition_penalty,
        length_penalty,
        no_repeat_ngram_size,
        hotwords,
        beam_size,
        patience,
        compression_ratio_threshold,
        log_prob_threshold,
        prompt_reset_on_temperature,
    ):
        with self.model_manager.load_model(model) as whisper:
            segments, transcription_info = whisper.transcribe(
                file,
                task=Task.TRANSLATE,
                initial_prompt=prompt,
                temperature=temperature,
                word_timestamps=response_format == ResponseFormat.VERBOSE_JSON,
                best_of=best_of,
                hotwords=hotwords,
                vad_filter=vad_filter,
                vad_parameters=vad_parameters,
                condition_on_previous_text=condition_on_previous_text,
                beam_size=beam_size,
                patience=patience,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                compression_ratio_threshold=compression_ratio_threshold,
                log_prob_threshold=log_prob_threshold,
                prompt_reset_on_temperature=prompt_reset_on_temperature,
            )
        segments = Segment.from_faster_whisper_segments(segments)
        return segments_to_response(segments, transcription_info, response_format)

    def prepare_audio_segments(self, request: TranscriptionRequest):
        with self.model_manager.load_model(request.model) as whisper:
            segments, transcription_info = whisper.transcribe(
                request.file,
                initial_prompt=request.prompt,
                language=request.language,
                temperature=request.temperature,
                word_timestamps="word" in request.timestamp_granularities,
                best_of=request.best_of,
                hotwords=request.hotwords,
                vad_filter=request.vad_filter,
                vad_parameters=request.vad_parameters,
                condition_on_previous_text=request.condition_on_previous_text,
                beam_size=request.beam_size,
                patience=request.patience,
                repetition_penalty=request.repetition_penalty,
                length_penalty=request.length_penalty,
                no_repeat_ngram_size=request.no_repeat_ngram_size,
                compression_ratio_threshold=request.compression_ratio_threshold,
                log_prob_threshold=request.log_prob_threshold,
                prompt_reset_on_temperature=request.prompt_reset_on_temperature,
            )

        segments = Segment.from_faster_whisper_segments(segments)

        if request.diarization:
            if request.file.suffix in [".wav", ".mp3", ".flac"]:
                dia_segments = self.diarization.diarize(str(request.file), request.diarization_speaker_count)
                segments = merge_whipser_diarization(segments, dia_segments)
            else:
                raise BentoMLException(
                    error_code=HTTPStatus.BAD_REQUEST, message="Diarization is not supported for non-wav files"
                )

        return segments, transcription_info
