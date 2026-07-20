import dataclasses
import logging
import time
from typing import Callable, Iterable

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
from faster_whisper.vad import VadOptions

from api_models.decode_params import DecodeParams
from api_models.enums import ResponseFormat, Task
from api_models.output_models import (
    WhisperResponse,
    segments_to_response,
)
from api_models.TranscriptionRequest import TranscriptionRequest
from api_models.TranslationRequest import TranslationRequest
from config import WhisperModelConfig
from core import Segment
from diarization_service import DiarizationSegment, DiarizationService
from helpers import metrics
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
    speech_intervals_to_chunks,
    turns_to_language_runs,
)
from helpers.transcription_cleaner import clean_transcription_segments
from helpers.whiper_diarization_merger import merge_whipser_diarization
from model_manager import WhisperModelProvider

logger = logging.getLogger(__name__)


def _strip_words(segments: Iterable[Segment]) -> Iterable[Segment]:
    """Drop per-word timestamps lazily, after the merge has consumed them."""
    for seg in segments:
        seg.words = None
        yield seg


class FasterWhisperHandler:
    def __init__(self):
        self.model_manager = WhisperModelProvider(WhisperModelConfig())
        # Loaded lazily on first diarize() so workers that never diarize don't pin a pipeline to the GPU.
        self.diarization = DiarizationService()

    def warmup(self, warm_diarization: bool = True) -> None:
        """Load models into VRAM at worker startup so the first request is fast.

        Loads the single resident Whisper model and runs a tiny synthetic decode to
        force weight upload plus CUDA/cuBLAS kernel initialisation. The pyannote
        pipeline is loaded too (it stays resident for the process lifetime).

        Resilient by design: a failure to warm either model logs and returns rather
        than crashing the worker — e.g. a token-less deployment that never diarizes.
        """
        try:
            whisper = self.model_manager.get()
            self._warm_decode(whisper)
            logger.info("Warmed Whisper model %s", self.model_manager.model_id)
        except Exception:
            logger.exception("Whisper warmup failed for model %s", self.model_manager.model_id)

        if warm_diarization:
            try:
                self.diarization.load()
                logger.info("Warmed diarization pipeline")
            except Exception:
                logger.exception("Diarization warmup failed (continuing without a pre-loaded pipeline)")

    @staticmethod
    def _warm_decode(whisper: WhisperModel) -> None:
        """Run one throwaway decode on 1s of silence to compile CUDA kernels."""
        silent = np.zeros(WHISPER_SAMPLE_RATE, dtype=np.float32)
        segments, _ = whisper.transcribe(silent, language="en")
        for _ in segments:  # drain the lazy generator so the decode actually runs
            pass

    def transcribe_audio(
        self,
        request: TranscriptionRequest,
    ) -> WhisperResponse:
        # Request validation (timestamp granularities, diarization/response-format coupling)
        # lives at the API boundary in service.py::_prepare_transcribe; internal callers are
        # trusted to pass an already-validated request.
        segments, transcription_info = self.prepare_audio_segments(request)
        try:
            cleaned = clean_transcription_segments(segments, transcription_info)
            return segments_to_response(cleaned, transcription_info, request.response_format)
        finally:
            # Release the held model ref even if cleaning or response building raises midway.
            segments.close()

    def translate_audio(self, request: TranslationRequest) -> WhisperResponse:
        t0 = time.perf_counter()
        whisper = self.model_manager.get()
        word_timestamps = request.response_format == ResponseFormat.VERBOSE_JSON
        decode_options = self._decode_options(request, word_timestamps)
        try:
            segments, transcription_info = whisper.transcribe(
                str(request.file),
                task=Task.TRANSLATE,
                vad_filter=request.vad_filter,
                vad_parameters=VadOptions(**request.vad_parameters.model_dump()),
                **decode_options,
            )
            segments = Segment.from_faster_whisper_segments(segments)
            response = segments_to_response(segments, transcription_info, request.response_format)
        except Exception as e:
            metrics.record_failure("decode", e)
            raise

        metrics.observe_decode(transcription_info.duration, transcription_info.language)
        metrics.observe_realtime_factor(t0, transcription_info.duration)
        return response

    def prepare_audio_segments(
        self,
        request: TranscriptionRequest,
        diarization_progress_callback: Callable[[float], None] | None = None,
    ):
        # Wall-clock start for the realtime-factor metric; it spans the full lazy decode and is
        # observed when _held_segments() is exhausted/closed below.
        t0 = time.perf_counter()

        # Diarize before loading Whisper: pyannote's speech turns double as the VAD for the
        # decode below, and running it first keeps its GPU peak from overlapping the decode.
        dia_segments: list[DiarizationSegment] = []
        if request.diarization:
            dia_start = time.perf_counter()
            try:
                dia_segments = list(
                    self.diarization.diarize(
                        str(request.file),
                        request.diarization_speaker_count,
                        progress_callback=diarization_progress_callback,
                    )
                )
            except Exception as e:
                metrics.record_failure("diarization", e)
                raise
            metrics.diarization_duration().observe(time.perf_counter() - dia_start)
            metrics.speaker_count().observe(len({seg.speaker for seg in dia_segments}))

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
        has_speech = False
        original_duration_s = 0.0
        if intervals:
            decoded = decode_audio(str(request.file), sampling_rate=WHISPER_SAMPLE_RATE)
            original_duration_s = decoded.shape[0] / WHISPER_SAMPLE_RATE
            # Only check that decodable speech exists — the full-file concatenate is deferred to
            # the per-run decode paths (which re-collapse per run) and the rare LID fallback, so a
            # long meeting never pays a whole-file np.concatenate that nothing downstream consumes.
            has_speech = bool(speech_intervals_to_chunks(intervals, decoded.shape[0], WHISPER_SAMPLE_RATE))

        # The single served model is resident and never unloaded, so the lazy transcribe()
        # generator can safely outlive this method with no ref guard. request.model is already
        # validated to equal the served model, so it is not used for loading.
        whisper = self.model_manager.get()
        try:
            decode_options = self._decode_options(request, word_timestamps)
            if has_speech:
                # Decode the collapsed speech as bounded runs cut at turn boundaries, never as one
                # block: a long continuous stretch handed to a single whisper.transcribe() makes its
                # long-form decode drift and drop whole windows. The single-language and the per-turn
                # language-detection paths share the same run machinery.
                assert decoded is not None  # has_speech is only set once decoded exists
                turns = sorted((max(t.start, 0.0), t.end) for t in dia_segments if t.end > t.start)
                if request.language is None:
                    # No language given: detect it per speaker turn so multilingual audio
                    # gets each region transcribed in its own language.
                    candidates = [str(c) for c in request.language_candidates] if request.language_candidates else None
                    segments, transcription_info = self._transcribe_language_runs(
                        whisper,
                        decoded,
                        intervals,
                        turns,
                        original_duration_s,
                        decode_options,
                        language_candidates=candidates,
                    )
                else:
                    # Explicit language: every run decodes in it, and segments keep language=None
                    # (per-segment language is only reported when it was auto-detected per region).
                    resolved = [str(request.language)] * len(turns)
                    segments, transcription_info = self._decode_language_runs(
                        whisper, decoded, turns, resolved, original_duration_s, decode_options, tag_language=False
                    )
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
        except Exception as e:
            metrics.record_failure("decode", e)
            raise

        metrics.observe_decode(transcription_info.duration, transcription_info.language)

        def _held_segments():
            try:
                yield from segments
            except Exception as e:
                metrics.record_failure("decode", e)
                raise
            finally:
                metrics.observe_realtime_factor(t0, transcription_info.duration)

        return _held_segments(), transcription_info

    @staticmethod
    def _decode_options(request: DecodeParams, word_timestamps: bool) -> dict:
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
        intervals: list[tuple[float, float]],
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
            # No turn long enough to detect on: collapse all speech now (deferred from the
            # caller so the common per-turn path never pays the full-file concatenate) and
            # detect once, like the single-language path would.
            collapsed = collapse_decoded_to_speech(decoded, intervals)
            assert collapsed is not None  # caller only takes this path when speech exists
            language, _, _ = whisper.detect_language(audio=collapsed[0])
            resolved = [language] * len(turns)

        return self._decode_language_runs(
            whisper, decoded, turns, resolved, original_duration_s, decode_options, tag_language=True
        )

    def _decode_language_runs(
        self,
        whisper: WhisperModel,
        decoded: np.ndarray,
        turns: list[tuple[float, float]],
        resolved: list[str],
        original_duration_s: float,
        decode_options: dict,
        tag_language: bool,
    ):
        """Decode the given turns as bounded runs and restore each onto the original
        timeline. Consecutive turns sharing a language are collapsed into one run,
        further capped at a maximum span (``turns_to_language_runs``) so a long
        continuous stretch is decoded as several clips instead of one drifting block.

        ``tag_language`` stamps each segment with its run's language; the single-
        language path leaves it off, since per-segment language is only reported when
        it was auto-detected per region."""
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
                    if tag_language:
                        segment.language = language
                    yield segment

        return tagged_segments(), transcription_info
