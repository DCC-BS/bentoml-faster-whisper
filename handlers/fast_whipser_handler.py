from pathlib import Path

from faster_whisper.vad import VadOptions

from api_models.enums import ResponseFormat, Task
from api_models.input_models import (
    validate_timestamp_granularities,
)
from api_models.output_models import (
    segments_to_response,
)
from config import WhisperModelConfig
from core import Segment
from diarization_service import DiarizationService
from model_manager import WhisperModelManager
from whiper_diarization_merger import merge_whipser_diarization


class FasterWhisperHandler:
    def __init__(self):
        self.model_manager = WhisperModelManager(WhisperModelConfig())
        self.diarization = DiarizationService()
        self.diarization.load()

    def transcribe_audio(
        self,
        file: Path,
        model: str | None,
        language: str | None,
        prompt: str | None,
        response_format: ResponseFormat,
        temperature: float | list[float] | None,
        timestamp_granularities: list[str] | None,
        best_of: int | None,
        vad_filter: bool | None,
        vad_parameters: VadOptions | None,
        condition_on_previous_text: bool | None,
        repetition_penalty: float | None,
        length_penalty: float | None,
        no_repeat_ngram_size: int | None,
        hotwords: str | None,
        beam_size: int | None,
        patience: float | None,
        compression_ratio_threshold: float | None,
        log_prob_threshold: float | None,
        prompt_reset_on_temperature: float | None,
        diarization: bool | None,
        diarization_speaker_count: int | None,
    ) -> str:
        validate_timestamp_granularities(
            response_format, timestamp_granularities, diarization
        )

        segments, transcription_info = self.prepare_audio_segments(
            file=file,
            language=language,
            model=model,
            prompt=prompt,
            temperature=temperature,
            timestamp_granularities=timestamp_granularities,
            best_of=best_of,
            vad_filter=vad_filter,
            vad_parameters=vad_parameters,
            condition_on_previous_text=condition_on_previous_text,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            hotwords=hotwords,
            beam_size=beam_size,
            patience=patience,
            compression_ratio_threshold=compression_ratio_threshold,
            log_prob_threshold=log_prob_threshold,
            prompt_reset_on_temperature=prompt_reset_on_temperature,
            diarization=diarization,
            diarization_speaker_count=diarization_speaker_count,
        )

        if diarization:
            dia_segments = self.diarization.diarize(file, diarization_speaker_count)
            segments = merge_whipser_diarization(segments, dia_segments)

        return segments_to_response(segments, transcription_info, response_format)

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

    def prepare_audio_segments(
        self,
        file,
        language,
        model,
        prompt,
        temperature,
        timestamp_granularities,
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
        diarization: bool | None,
        diarization_speaker_count: int | None,
    ):
        with self.model_manager.load_model(model) as whisper:
            segments, transcription_info = whisper.transcribe(
                file,
                initial_prompt=prompt,
                language=language,
                temperature=temperature,
                word_timestamps="word" in timestamp_granularities,
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

        if diarization:
            dia_segments = self.diarization.diarize(file, diarization_speaker_count)
            segments = merge_whipser_diarization(segments, dia_segments)

        return segments, transcription_info
