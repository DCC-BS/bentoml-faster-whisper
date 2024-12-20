import logging
import os
import time
from typing import TYPE_CHECKING, Annotated, Any, AsyncGenerator, List

import bentoml
import huggingface_hub
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from faster_whisper.vad import VadOptions
from pydub import AudioSegment
from api_models.enums import ResponseFormat, Task
from api_models.input_models import (
    TranscriptionRequest,
    hf_model_info_to_model_object,
    validate_timestamp_granularities,
)
from api_models.output_models import (
    ModelListResponse,
    ModelObject,
    segments_to_response,
    segments_to_streaming_response,
)
from config import WhisperModelConfig
from core import Segment
from logger import configure_logging
from model_manager import WhisperModelManager
from prometheus_client import Histogram

if TYPE_CHECKING:
    from huggingface_hub.hf_api import ModelInfo

from pathlib import Path


logger = logging.getLogger(__name__)

fastapi = FastAPI()

configure_logging()

load_dotenv()


TIMEOUT = int(os.getenv("TIMEOUT", 3000))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", 4))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", 4))
MAX_LATENCY_MS = int(os.getenv("MAX_LATENCY_MS", 60 * 1000))
DURATION_BUCKETS_S = [1., 5., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., float("inf")]
AUDIO_INPUT_LENGTH_BUCKETS_S = [
    10.,
    30.,
    60.,
    60. * 2,
    60. * 5,
    60. * 7,
    60. * 10,
    60. * 20,
    60. * 30,
    60. * 60,
    60. * 60 * 2,
    float("inf"),
]
REALTIME_FACTOR_BUCKETS = [
    0.5,
    1.0,
    3.0,
    5.0,
    10.0,
    20.0,
    30.0,
    40.0,
    50.0,
    float("inf"),
]


input_audio_length_histogram = Histogram(
    name="input_audio_length_seconds",
    documentation="Input audio length in seconds",
    buckets=AUDIO_INPUT_LENGTH_BUCKETS_S,
)

realtime_factor_histogram = Histogram(
    name="realtime_factor",
    documentation="Realtime factor, e.g. avg. audio seconds transcribed per second",
    buckets=REALTIME_FACTOR_BUCKETS,
)


def get_audio_duration(file: Path):
    """Gets the duration of an audio file in seconds."""
    duration = AudioSegment.from_file(file).duration_seconds
    return duration


class FasterWhisperHandler:
    def __init__(self):
        self.model_manager = WhisperModelManager(WhisperModelConfig())

    def transcribe_audio(
        self,
        file,
        model,
        language,
        prompt,
        response_format,
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
    ):
        validate_timestamp_granularities(response_format, timestamp_granularities)

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
        )

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
        return segments, transcription_info


@bentoml.service(traffic={"timeout": TIMEOUT})
class BatchFasterWhisper:
    def __init__(self):
        self.handler = FasterWhisperHandler()

    @bentoml.api(
        batchable=True, max_batch_size=MAX_BATCH_SIZE, max_latency_ms=MAX_LATENCY_MS
    )
    async def batch_transcribe(self, requests: List[TranscriptionRequest]) -> List[str]:
        logger.debug(f"number of requests processed: {len(requests)}")
        return [self.handler.transcribe_audio() for request in requests]


@bentoml.service(
    title="Faster Whisper API",
    description="This is a custom Faster Whisper API that is fully compatible with the OpenAI SDK and offers aditional options.",
    version="1.0.0",
    traffic={
        "timeout": TIMEOUT,
        "max_concurrency": MAX_CONCURRENCY,
    },
    metrics={
    "enabled": True,
    "namespace": "bentoml_service",
    "duration": {
        "buckets": DURATION_BUCKETS_S
    }
}
)
@bentoml.mount_asgi_app(fastapi, path="/v1")
class FasterWhisper:
    batch = bentoml.depends(BatchFasterWhisper)

    def __init__(self):
        self.handler = FasterWhisperHandler()

    @bentoml.api(route="/v1/audio/transcriptions", input_spec=TranscriptionRequest)
    def transcribe(
        self, **params: Any
    ) -> (
        Annotated[str, bentoml.validators.ContentType("text/plain")]
        | Annotated[str, bentoml.validators.ContentType("application/json")]
        | Annotated[str, bentoml.validators.ContentType("text/vtt")]
        | Annotated[str, bentoml.validators.ContentType("text/event-stream")]
    ):
        start_time = time.time()
        vad_parameters = params["vad_parameters"]
        if isinstance(vad_parameters, dict):
            vad_parameters = VadOptions(**vad_parameters)
        vad_parameters.max_speech_duration_s = (
            float("inf")
            if vad_parameters.max_speech_duration_s == 999_999
            else vad_parameters.max_speech_duration_s
        )
        params["vad_parameters"] = vad_parameters
        result = self.handler.transcribe_audio(**params)
        end_time = time.time()
        duration = end_time - start_time
        audio_file = params["file"]
        audio_duration = get_audio_duration(audio_file)
        input_audio_length_histogram.observe(audio_duration)
        realtime_factor_histogram.observe(audio_duration / duration)
        return result

    @bentoml.api(
        route="/v1/audio/transcriptions/batch", input_spec=TranscriptionRequest
    )
    async def batch_transcribe(
        self, **params: Any
    ) -> (
        Annotated[str, bentoml.validators.ContentType("text/plain")]
        | Annotated[str, bentoml.validators.ContentType("application/json")]
        | Annotated[str, bentoml.validators.ContentType("text/vtt")]
        | Annotated[str, bentoml.validators.ContentType("text/event-stream")]
    ):
        request = TranscriptionRequest(**params)
        result = await self.batch.batch_transcribe([request])
        return result[0]

    @bentoml.task(
        route="/v1/audio/transcriptions/task",
        input_spec=TranscriptionRequest,
    )
    def task_transcribe(
        self, **params: Any
    ) -> (
        Annotated[str, bentoml.validators.ContentType("text/plain")]
        | Annotated[str, bentoml.validators.ContentType("application/json")]
        | Annotated[str, bentoml.validators.ContentType("text/vtt")]
        | Annotated[str, bentoml.validators.ContentType("text/event-stream")]
    ):
        vad_parameters = params["vad_parameters"]
        if isinstance(vad_parameters, dict):
            vad_parameters = VadOptions(**vad_parameters)
        vad_parameters.max_speech_duration_s = (
            float("inf")
            if vad_parameters.max_speech_duration_s == 999_999
            else vad_parameters.max_speech_duration_s
        )
        params["vad_parameters"] = vad_parameters

        return self.handler.transcribe_audio(**params)

    @bentoml.api(
        route="/v1/audio/transcriptions/stream", input_spec=TranscriptionRequest
    )
    async def streaming_transcribe(self, **params: Any) -> AsyncGenerator[str, None]:
        vad_parameters = params["vad_parameters"]
        if isinstance(vad_parameters, dict):
            vad_parameters = VadOptions(**vad_parameters)
        vad_parameters.max_speech_duration_s = (
            float("inf")
            if vad_parameters.max_speech_duration_s == 999_999
            else vad_parameters.max_speech_duration_s
        )
        params["vad_parameters"] = vad_parameters
        response_format = params.pop("response_format")
        timestamp_granularities = params["timestamp_granularities"]
        validate_timestamp_granularities(response_format, timestamp_granularities)

        segments, transcription_info = self.handler.prepare_audio_segments(**params)
        generator = segments_to_streaming_response(
            segments, transcription_info, response_format
        )

        for chunk in generator:
            yield chunk

    @bentoml.api(route="/v1/audio/translations", input_spec=TranscriptionRequest)
    def translate(
        self, **params: Any
    ) -> (
        Annotated[str, bentoml.validators.ContentType("text/plain")]
        | Annotated[str, bentoml.validators.ContentType("application/json")]
        | Annotated[str, bentoml.validators.ContentType("text/vtt")]
        | Annotated[str, bentoml.validators.ContentType("text/event-stream")]
    ):
        start_time = time.time()
        vad_parameters = params["vad_parameters"]
        if isinstance(vad_parameters, dict):
            vad_parameters = VadOptions(**vad_parameters)
        vad_parameters.max_speech_duration_s = (
            float("inf")
            if vad_parameters.max_speech_duration_s == 999_999
            else vad_parameters.max_speech_duration_s
        )
        params["vad_parameters"] = vad_parameters
        result = self.handler.translate_audio(**params)
        end_time = time.time()
        duration = end_time - start_time
        audio_file = params["file"]
        audio_duration = get_audio_duration(audio_file)
        input_audio_length_histogram.observe(audio_duration)
        realtime_factor_histogram.observe(audio_duration / duration)
        return result

    @fastapi.get("/models")
    def get_models(self) -> ModelListResponse:
        models = huggingface_hub.list_models(
            library="ctranslate2", tags="automatic-speech-recognition", cardData=True
        )
        models = list(models)
        models.sort(key=lambda model: model.downloads, reverse=True)
        transformed_models = [hf_model_info_to_model_object(model) for model in models]
        return ModelListResponse(data=transformed_models)

    @fastapi.get("/models/{model_name:path}")
    def get_model(
        self,
        model_name=Annotated[
            str, Path(example="Systran/faster-distil-whisper-large-v3")
        ],
    ) -> ModelObject:
        models = huggingface_hub.list_models(
            model_name=model_name,
            library="ctranslate2",
            tags="automatic-speech-recognition",
            cardData=True,
        )
        models = list(models)
        models.sort(key=lambda model: model.downloads, reverse=True)  # type: ignore  # noqa: PGH003
        if len(models) == 0:
            raise HTTPException(status_code=404, detail="No models found.")
        exact_match: ModelInfo | None = None
        for model in models:
            if model.id == model_name:
                exact_match = model
                break
        if exact_match is None:
            raise HTTPException(
                status_code=404,
                detail=f"Model doesn't exists. Possible matches: {', '.join([model.id for model in models])}",
            )
        return hf_model_info_to_model_object(exact_match)
