import logging
import os
from collections.abc import Generator
from typing import TYPE_CHECKING, Annotated, Any

import bentoml
import huggingface_hub
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi import Path as FastAPIPath

from api_models.input_models import (
    hf_model_info_to_model_object,
    validate_timestamp_granularities,
)
from api_models.output_models import (
    ModelListResponse,
    ModelObject,
    WhisperResponse,
    segments_to_response,
    segments_to_streaming_response,
)
from api_models.ProgressResponse import ProgressResponse
from api_models.TranscriptionRequest import TranscriptionRequest
from api_models.TranslationRequest import TranslationRequest
from core import Segment
from handlers.fast_whipser_handler import FasterWhisperHandler
from handlers.progress_handler import ProgressHandler
from helpers.logger import configure_logging
from helpers.timing import measure_processing_time
from helpers.transcription_cleaner import clean_transcription_segments

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

    @bentoml.api(batchable=True, max_batch_size=MAX_BATCH_SIZE, max_latency_ms=MAX_LATENCY_MS)
    async def batch_transcribe(self, requests: list[TranscriptionRequest]) -> list[WhisperResponse]:
        logger.debug(f"number of requests processed: {len(requests)}")
        return [self.handler.transcribe_audio(request) for request in requests]


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
        self.progress_handler = ProgressHandler()

    @bentoml.api(route="/v1/audio/transcriptions", input_spec=TranscriptionRequest)  # type: ignore
    @measure_processing_time
    def transcribe(self, **params: Any) -> WhisperResponse:
        request = TranscriptionRequest.from_dict(params)
        self._prepare_transcribe(request)
        return self.handler.transcribe_audio(request)

    @bentoml.api(route="/v1/audio/transcriptions/batch", input_spec=TranscriptionRequest)  # type: ignore
    async def batch_transcribe(self, **params: Any) -> WhisperResponse:
        request = TranscriptionRequest.from_dict(params)
        self._prepare_transcribe(request)
        results = await self.batch.batch_transcribe([request])
        if not results:
            raise RuntimeError("Batch transcription returned no results")
        return results[0]

    @bentoml.task(
        route="/v1/audio/transcriptions/task",
        input_spec=TranscriptionRequest,  # type: ignore
    )
    def task_transcribe(self, **params: Any) -> WhisperResponse:
        request = TranscriptionRequest.from_dict(params)
        self._prepare_transcribe(request)

        result: list[Segment] = []

        # Register the entry before diarization (which runs eagerly inside prepare_audio_segments) so
        # the UI sees a tracked task and live diarization progress, not a missing/0 entry until decode.
        diarization_progress_callback = None
        if request.progress_id:
            self.progress_handler.add_progress(request.progress_id)
            progress_id = request.progress_id

            # Fold the two stages into one monotonic 0..1 value so the bar never resets:
            # diarization occupies 0..0.3, transcription 0.3..1.0.
            def diarization_progress_callback(fraction: float) -> None:
                self.progress_handler.update_progress(
                    progress_id,
                    ProgressResponse(progress=fraction * 0.3, currentTime=0, duration=0),
                )

        # prepare_audio_segments runs eager diarization/decode and can raise on bad input, so it lives
        # inside the try to guarantee the progress entry registered above is always removed.
        segments = None
        try:
            segments, transcription_info = self.handler.prepare_audio_segments(
                request, diarization_progress_callback=diarization_progress_callback
            )

            for segment in segments:
                if request.progress_id:
                    self.progress_handler.update_progress(
                        request.progress_id,
                        ProgressResponse(
                            progress=0.3 + 0.7 * (segment.end / transcription_info.duration),
                            currentTime=segment.end,
                            duration=transcription_info.duration,
                        ),
                    )

                result.append(segment)

            result = list(clean_transcription_segments(result, transcription_info))
            return segments_to_response(result, transcription_info, request.response_format)
        finally:
            # Release the held model ref and progress entry even if the decode raises midway.
            if segments is not None:
                segments.close()
            if request.progress_id is not None:
                self.progress_handler.remove_progress(request.progress_id)

    @bentoml.api(route="/v1/audio/transcriptions/stream", input_spec=TranscriptionRequest)  # type: ignore
    @measure_processing_time
    def streaming_transcribe(self, **params: Any) -> Generator[str, None, None]:
        request = TranscriptionRequest.from_dict(params)

        self._prepare_transcribe(request)

        segments, transcription_info = self.handler.prepare_audio_segments(request)
        generator = segments_to_streaming_response(segments, transcription_info, request.response_format)

        try:
            for chunk in generator:
                yield chunk
        finally:
            # Release the held model ref if the client disconnects mid-stream.
            segments.close()

    @bentoml.api(route="/v1/audio/translations", input_spec=TranslationRequest)  # type: ignore
    @measure_processing_time
    def translate(self, **params: Any) -> WhisperResponse:
        request = TranslationRequest.from_dict(params)
        self._configure_vad_options(request)
        result = self.handler.translate_audio(**params)
        return result

    @fastapi.get("/progress/{progress_id}")
    def get_progress(self, progress_id: str) -> ProgressResponse:
        return self.progress_handler.get_progress(progress_id)

    @fastapi.get("/models")
    def get_models(self) -> ModelListResponse:
        models = huggingface_hub.list_models(
            filter="ctranslate2", pipeline_tag="automatic-speech-recognition", cardData=True
        )
        models = list(models)
        models.sort(key=lambda model: model.downloads, reverse=True)
        transformed_models = [hf_model_info_to_model_object(model) for model in models]
        return ModelListResponse(data=transformed_models)

    @fastapi.get("/models/{model_name:path}")
    def get_model(
        self,
        model_name: Annotated[str, FastAPIPath(examples=["Systran/faster-distil-whisper-large-v2"])],
    ) -> ModelObject:
        models = huggingface_hub.list_models(
            search=str(model_name),
            filter="ctranslate2",
            pipeline_tag="automatic-speech-recognition",
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

    def _prepare_transcribe(self, request: TranscriptionRequest):
        validate_timestamp_granularities(
            request.response_format,
            request.timestamp_granularities,
            request.diarization,
        )
        self._configure_vad_options(request)

    def _configure_vad_options(self, request: TranscriptionRequest | TranslationRequest):
        if request.vad_parameters.max_speech_duration_s == 999_999:
            request.vad_parameters.max_speech_duration_s = float("inf")
