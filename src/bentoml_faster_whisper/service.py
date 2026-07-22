import logging
import os
from collections.abc import Generator
from typing import Annotated, Any

import bentoml
from fastapi import FastAPI, HTTPException
from fastapi import Path as FastAPIPath

from bentoml_faster_whisper.config import faster_whisper_config
from bentoml_faster_whisper.models.input_models import (
    validate_timestamp_granularities,
)
from bentoml_faster_whisper.models.output_models import (
    ModelListResponse,
    ModelObject,
    WhisperResponse,
    content_type_for_format,
    segments_to_response,
    segments_to_streaming_response,
)
from bentoml_faster_whisper.models.progress_response import ProgressResponse
from bentoml_faster_whisper.models.transcription_request import TranscriptionRequest
from bentoml_faster_whisper.models.translation_request import TranslationRequest
from bentoml_faster_whisper.container import Container
from bentoml_faster_whisper.utils.core import Segment
from bentoml_faster_whisper.utils.logger import configure_logging
from bentoml_faster_whisper.utils.transcription_cleaner import clean_transcription_segments

logger = logging.getLogger(__name__)


fastapi = FastAPI()

configure_logging()


_HIDDEN_TASK_ROUTE_SUFFIXES = ("/task/cancel", "/task/retry")


def _hide_task_routes_from_openapi() -> None:
    """Drop unused/non-functional auto-generated task routes from OpenAPI docs."""
    import importlib

    openapi = importlib.import_module("_bentoml_sdk.service.openapi")
    if getattr(openapi.generate_spec, "_hides_task_routes", False):
        return
    _generate_spec = openapi.generate_spec

    def generate_spec(svc, **kwargs):
        spec = _generate_spec(svc, **kwargs)
        for route in list(spec.paths):
            if route.endswith(_HIDDEN_TASK_ROUTE_SUFFIXES):
                del spec.paths[route]
        return spec

    generate_spec._hides_task_routes = True  # type: ignore[attr-defined]
    openapi.generate_spec = generate_spec


_hide_task_routes_from_openapi()

TIMEOUT = int(os.getenv("TIMEOUT", 3000))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", 4))
WARMUP_ON_STARTUP = os.getenv("WARMUP_ON_STARTUP", "true").lower() not in ("false", "0", "no")
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
    def __init__(self):
        self.container = Container()
        self.config = self.container.config()
        self.handler = self.container.faster_whisper_handler()
        self.progress_handler = self.container.progress_handler()

    @bentoml.on_startup
    def warmup(self):
        """Warm up worker process by pre-loading resident model into VRAM."""
        if WARMUP_ON_STARTUP:
            self.handler.warmup()

    @bentoml.api(route="/v1/audio/transcriptions", input_spec=TranscriptionRequest)  # type: ignore
    def transcribe(self, ctx: bentoml.Context = None, **params: Any) -> WhisperResponse:
        request = TranscriptionRequest.from_dict(params)
        self._prepare_transcribe(request)
        self._set_response_content_type(ctx, request.response_format)
        return self.handler.transcribe_audio(request)

    @bentoml.api(route="/v1/audio/transcriptions/batch", input_spec=TranscriptionRequest)  # type: ignore
    def batch_transcribe(self, ctx: bentoml.Context = None, **params: Any) -> WhisperResponse:
        """Transcribe audio (kept for OpenAI API backward compatibility)."""
        request = TranscriptionRequest.from_dict(params)
        self._prepare_transcribe(request)
        self._set_response_content_type(ctx, request.response_format)
        return self.handler.transcribe_audio(request)

    @bentoml.task(
        route="/v1/audio/transcriptions/task",
        input_spec=TranscriptionRequest,  # type: ignore
    )
    def task_transcribe(self, **params: Any) -> WhisperResponse:
        request = TranscriptionRequest.from_dict(params)
        self._prepare_transcribe(request)

        result: list[Segment] = []

        diarization_progress_callback = None
        if request.progress_id:
            self.progress_handler.add_progress(request.progress_id)
            progress_id = request.progress_id

            def diarization_progress_callback(fraction: float) -> None:
                self.progress_handler.update_progress(
                    progress_id,
                    ProgressResponse(progress=fraction * 0.3, currentTime=0, duration=0),
                )

        segments = None
        try:
            segments, transcription_info = self.handler.prepare_audio_segments(
                request, diarization_progress_callback=diarization_progress_callback
            )

            for segment in segments:
                if request.progress_id:
                    fraction = segment.end / transcription_info.duration if transcription_info.duration else 0.0
                    self.progress_handler.update_progress(
                        request.progress_id,
                        ProgressResponse(
                            progress=0.3 + 0.7 * fraction,
                            currentTime=segment.end,
                            duration=transcription_info.duration,
                        ),
                    )

                result.append(segment)

            result = list(clean_transcription_segments(result, transcription_info))
            return segments_to_response(result, transcription_info, request.response_format)
        finally:
            if segments is not None:
                segments.close()
            if request.progress_id is not None:
                self.progress_handler.remove_progress(request.progress_id)

    @bentoml.api(route="/v1/audio/transcriptions/stream", input_spec=TranscriptionRequest)  # type: ignore
    def streaming_transcribe(self, **params: Any) -> Generator[str, None, None]:
        request = TranscriptionRequest.from_dict(params)

        self._prepare_transcribe(request)

        segments, transcription_info = self.handler.prepare_audio_segments(request)
        cleaned = clean_transcription_segments(segments, transcription_info)
        generator = segments_to_streaming_response(cleaned, transcription_info, request.response_format)

        try:
            for chunk in generator:
                yield chunk
        finally:
            segments.close()

    @bentoml.api(route="/v1/audio/translations", input_spec=TranslationRequest)  # type: ignore
    def translate(self, ctx: bentoml.Context = None, **params: Any) -> WhisperResponse:
        request = TranslationRequest.from_dict(params)
        self._configure_vad_options(request)
        self._set_response_content_type(ctx, request.response_format)
        return self.handler.translate_audio(request)

    @fastapi.get("/progress/{progress_id}")
    async def get_progress(self, progress_id: str) -> ProgressResponse:
        """Retrieve progress for a running transcription task."""
        return self.progress_handler.get_progress(progress_id)

    def _served_model_object(self) -> ModelObject:
        """The single model this API serves, as a static OpenAI-style ModelObject."""
        return ModelObject(
            id=self.config.faster_whisper.default_model_name,
            created=1668556800,
            object_="model",
            owned_by="Systran",
            language=[],
        )

    @fastapi.get("/models")
    async def get_models(self) -> ModelListResponse:
        """List models served by this endpoint."""
        return ModelListResponse(data=[self._served_model_object()])

    @fastapi.get("/models/{model_name:path}")
    async def get_model(
        self,
        model_name: Annotated[str, FastAPIPath(examples=[faster_whisper_config.default_model_name])],
    ) -> ModelObject:
        """Retrieve details of a specific served model."""
        served = self.config.faster_whisper.default_model_name
        if model_name != served:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found. Only '{served}' is served.",
            )
        return self._served_model_object()

    def _set_response_content_type(self, ctx: "bentoml.Context | None", response_format) -> None:
        """Set HTTP Content-Type header on BentoML context based on target response format."""
        if ctx is None:
            return
        ctx.response.headers["content-type"] = content_type_for_format(response_format)

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
