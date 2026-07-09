import dataclasses
import sys
from typing import Callable, Iterable

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
from faster_whisper.vad import VadOptions

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
from diarization_service import DiarizationSegment, DiarizationService
from helpers.language_id import (
    detect_turn_language_probs,
    fill_missing_rows_from_intervals,
    resolve_language_inventory,
    viterbi_smooth_languages,
)
from helpers.speech_regions import (
    WHISPER_SAMPLE_RATE,
    collapse_decoded_to_speech,
    diarization_to_speech_intervals,
    restore_and_split_segments,
    turns_to_language_runs,
)
from helpers.transcription_cleaner import clean_transcription_segments
from helpers.whiper_diarization_merger import merge_whipser_diarization
from model_manager import WhisperModelManager


def _strip_words(segments: Iterable[Segment]) -> Iterable[Segment]:
    """Drop per-word timestamps lazily, after the merge has consumed them."""
    for seg in segments:
        seg.words = None
        yield seg


class FasterWhisperHandler:
    def __init__(self):
        self.model_manager = WhisperModelManager(WhisperModelConfig())
        # Loaded lazily on first diarize() so workers that never diarize don't pin a pipeline to the GPU.
        self.diarization = DiarizationService()

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
        try:
            cleaned = clean_transcription_segments(segments, transcription_info)
            return segments_to_response(cleaned, transcription_info, request.response_format)
        finally:
            # Release the held model ref even if cleaning or response building raises midway.
            segments.close()

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
        # Consume the lazy generator inside the context manager so the model ref-count is held for the whole decode.
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
        request: TranscriptionRequest,
        diarization_progress_callback: Callable[[float], None] | None = None,
    ):
        # Diarize before loading Whisper: pyannote's speech turns double as the VAD for the
        # decode below, and running it first keeps its GPU peak from overlapping the decode.
        dia_segments: list[DiarizationSegment] = []
        if request.diarization:
            dia_segments = list(
                self.diarization.diarize(
                    str(request.file),
                    request.diarization_speaker_count,
                    progress_callback=diarization_progress_callback,
                )
            )

        # Cut the audio down to pyannote's speech turns before decoding — the same mechanism
        # faster-whisper applies internally for its silero VAD: silence never reaches the
        # decoder, and restore_and_split_segments() maps the results back onto the original
        # timeline afterwards, snapping boundaries to the speech regions (so the merge below
        # still lines up). Passing the regions as clip_timestamps instead is not equivalent:
        # each clip tail gets zero-padded to a 30s window and the model's timestamp tokens
        # drift there, unclamped. Word timestamps are always requested with diarization so the
        # merge assigns speakers per word (and seams can be split); silero only remains as
        # fallback when diarization is off or found no speech at all — no speech regions would
        # mean decoding the entire file.
        intervals = diarization_to_speech_intervals(dia_segments) if dia_segments else []
        word_timestamps = ("word" in request.timestamp_granularities) or bool(dia_segments)

        decoded: np.ndarray | None = None
        collapsed: tuple[np.ndarray, list[dict]] | None = None
        original_duration_s = 0.0
        if intervals:
            decoded = decode_audio(str(request.file), sampling_rate=WHISPER_SAMPLE_RATE)
            original_duration_s = decoded.shape[0] / WHISPER_SAMPLE_RATE
            collapsed = collapse_decoded_to_speech(decoded, intervals)

        # transcribe() returns a lazy generator that only decodes while iterated, so the model ref-count
        # must be held for the lifetime of the returned generator, not just this method call. We enter the
        # context manager here and exit it when the generator is exhausted, closed, or raises.
        model_ctx = self.model_manager.load_model(request.model)
        whisper = model_ctx.__enter__()
        try:
            decode_options = self._decode_options(request, word_timestamps)
            if collapsed is not None and request.language is None:
                # No language given: detect it per speaker turn so multilingual audio
                # gets each region transcribed in its own language.
                assert decoded is not None  # collapsed is derived from decoded
                turns = sorted((max(t.start, 0.0), t.end) for t in dia_segments if t.end > t.start)
                candidates = [str(c) for c in request.language_candidates] if request.language_candidates else None
                segments, transcription_info = self._transcribe_language_runs(
                    whisper,
                    decoded,
                    collapsed,
                    turns,
                    original_duration_s,
                    decode_options,
                    language_candidates=candidates,
                )
            elif collapsed is not None:
                audio, speech_chunks = collapsed
                segments, transcription_info = whisper.transcribe(
                    audio, language=request.language, vad_filter=False, **decode_options
                )
                # transcribe() saw only the concatenated speech, so its timestamps and reported
                # duration are on the collapsed timeline: map the timestamps back (clamping and
                # splitting seam-straddling segments) and report the original file's duration.
                segments = restore_and_split_segments(segments, speech_chunks, intervals, original_duration_s)
                transcription_info = dataclasses.replace(transcription_info, duration=original_duration_s)
            else:
                segments, transcription_info = whisper.transcribe(
                    str(request.file),
                    language=request.language,
                    vad_filter=request.vad_filter,
                    vad_parameters=VadOptions(**request.vad_parameters.model_dump()),
                    **decode_options,
                )
                segments = Segment.from_faster_whisper_segments(segments)

            if dia_segments:
                segments = merge_whipser_diarization(segments, dia_segments)

            # Word timestamps are requested for every diarized decode, but the response shape must
            # match the request: drop the words the merge just consumed unless they were asked for.
            if "word" not in request.timestamp_granularities:
                segments = _strip_words(segments)
        except BaseException:
            model_ctx.__exit__(*sys.exc_info())
            raise

        def _held_segments():
            try:
                yield from segments
            finally:
                model_ctx.__exit__(None, None, None)

        return _held_segments(), transcription_info

    @staticmethod
    def _decode_options(request: TranscriptionRequest, word_timestamps: bool) -> dict:
        """The transcribe() options shared by every decode of this request; language
        and VAD arguments differ per pipeline path, so they are not included."""
        return dict(
            initial_prompt=request.prompt,
            temperature=request.temperature,
            word_timestamps=word_timestamps,
            best_of=request.best_of,
            hotwords=request.hotwords,
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

    def _transcribe_language_runs(
        self,
        whisper: WhisperModel,
        decoded: np.ndarray,
        collapsed: tuple[np.ndarray, list[dict]],
        turns: list[tuple[float, float]],
        original_duration_s: float,
        decode_options: dict,
        language_candidates: list[str] | None = None,
    ):
        """Detect the language of each speaker turn (batched, smoothed over the
        whole file) and decode consecutive same-language turns as one collapsed
        run, so a speaker switching languages mid-file gets every region
        transcribed in its own language. Timestamps of every run are restored
        onto the original timeline, and each emitted segment is tagged with its
        run's language."""
        durations = [end - start for start, end in turns]
        prob_rows = detect_turn_language_probs(whisper, decoded, turns)
        prob_rows = fill_missing_rows_from_intervals(whisper, decoded, turns, prob_rows)

        if any(row is not None for row in prob_rows):
            inventory = resolve_language_inventory(prob_rows, durations, language_candidates)
            resolved = viterbi_smooth_languages(prob_rows, durations, inventory)
        else:
            # No turn long enough to detect on: detect once on all collapsed
            # speech, like the single-language path would.
            language, _, _ = whisper.detect_language(audio=collapsed[0])
            resolved = [language] * len(turns)

        runs = []
        for language, run_intervals in turns_to_language_runs(turns, resolved):
            run_collapsed = collapse_decoded_to_speech(decoded, run_intervals)
            if run_collapsed is None:
                continue
            run_audio, run_chunks = run_collapsed
            fw_segments, info = whisper.transcribe(run_audio, language=language, vad_filter=False, **decode_options)
            restored = restore_and_split_segments(fw_segments, run_chunks, run_intervals, original_duration_s)
            runs.append((language, run_intervals, info, restored))

        # The response carries a single top-level language: the one covering the most
        # speech time. Per-segment languages preserve the rest.
        majority_language, _, majority_info, _ = max(runs, key=lambda run: sum(end - start for start, end in run[1]))
        transcription_info = dataclasses.replace(
            majority_info, language=majority_language, duration=original_duration_s
        )

        def tagged_segments() -> Iterable[Segment]:
            next_id = 0
            for language, _, _, restored in runs:
                for segment in restored:
                    segment.id = next_id
                    next_id += 1
                    segment.language = language
                    yield segment

        return tagged_segments(), transcription_info
