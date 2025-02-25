import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, AsyncGenerator, List

import bentoml
import huggingface_hub
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from faster_whisper.vad import VadOptions

from api_models.input_models import (
    TranscriptionRequest,
    TranslationRequest,
    hf_model_info_to_model_object,
    validate_timestamp_granularities,
)
from api_models.output_models import (
    ModelListResponse,
    ModelObject,
    segments_to_streaming_response,
)
from handlers.fast_whipser_handler import FasterWhisperHandler
from logger import configure_logging
from utils import (
    get_audio_duration,
    input_audio_length_histogram,
    realtime_factor_histogram,
)

if TYPE_CHECKING:
    from huggingface_hub.hf_api import ModelInfo

logger = logging.getLogger(__name__)

fastapi = FastAPI()

configure_logging()

load_dotenv()

TIMEOUT = int(os.getenv("TIMEOUT", 3000))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", 4))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", 4))
MAX_LATENCY_MS = int(os.getenv("MAX_LATENCY_MS", 60 * 1000))
DURATION_BUCKETS_S = [
    1.0,
    5.0,
    10.0,
    20.0,
    30.0,
    40.0,
    50.0,
    60.0,
    70.0,
    80.0,
    90.0,
    100.0,
    float("inf"),
]


@bentoml.service(traffic={"timeout": TIMEOUT})
class BatchFasterWhisper:
    def __init__(self):
        self.handler = FasterWhisperHandler()

    @bentoml.api(
        batchable=True, max_batch_size=MAX_BATCH_SIZE, max_latency_ms=MAX_LATENCY_MS
    )
    async def batch_transcribe(self, requests: List[TranscriptionRequest]) -> List[str]:
        logger.debug(f"number of requests processed: {len(requests)}")
        return [self.handler.transcribe_audio() for _ in requests]


@bentoml.service(
    title="Faster Whisper API",
    description="This is a custom Faster Whisper API that is fully compatible with the OpenAI SDK and offers additional options.",
    version="1.0.0",
    traffic={
        "timeout": TIMEOUT,
        "max_concurrency": MAX_CONCURRENCY,
    },
    metrics={
        "enabled": True,
        "namespace": "bentoml_service",
        "duration": {"buckets": DURATION_BUCKETS_S},
    },
)
@bentoml.asgi_app(fastapi, path="/v1")
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
        validate_timestamp_granularities(
            response_format, timestamp_granularities, params.get("diarization")
        )

        segments, transcription_info = self.handler.prepare_audio_segments(**params)
        generator = segments_to_streaming_response(
            segments, transcription_info, response_format
        )

        for chunk in generator:
            yield chunk

    @bentoml.api(route="/v1/audio/translations", input_spec=TranslationRequest)
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
            str, Path(example="Systran/faster-distil-whisper-large-v2")
        ],
    ) -> ModelObject:
        models = huggingface_hub.list_models(
            model_name=model_name,
            library="ctranslate2",
            tags="automatic-speech-recognition",
            cardData=True,
        )
        models = list(models)
        models.sort(key=lambda model: model.downloads, reverse=True)  # noqa: PGH003
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
