import logging
import os
from collections.abc import Generator
from typing import Annotated, Any

import bentoml
from dotenv import load_dotenv
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

load_dotenv()

TIMEOUT = int(os.getenv("TIMEOUT", 3000))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", 4))
# Load models into VRAM at worker startup instead of lazily on first request. Set to
# "false" to keep the old lazy behaviour (e.g. token-less dev without cached weights).
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
        # Runs once per worker: load the resident Whisper model and the pyannote
        # pipeline into VRAM so the first request doesn't pay the load cost.
        if WARMUP_ON_STARTUP:
            self.handler.warmup()

    @bentoml.api(route="/v1/audio/transcriptions", input_spec=TranscriptionRequest)  # type: ignore
    def transcribe(self, **params: Any) -> WhisperResponse:
        request = TranscriptionRequest.from_dict(params)
        self._prepare_transcribe(request)
        return self.handler.transcribe_audio(request)

    @bentoml.api(route="/v1/audio/transcriptions/batch", input_spec=TranscriptionRequest)  # type: ignore
    def batch_transcribe(self, **params: Any) -> WhisperResponse:
        # Kept for API compatibility; the former separate batchable service is gone, so this
        # now decodes on the resident model in-process, like /v1/audio/transcriptions.
        request = TranscriptionRequest.from_dict(params)
        self._prepare_transcribe(request)
        return self.handler.transcribe_audio(request)

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
    def translate(self, **params: Any) -> WhisperResponse:
        request = TranslationRequest.from_dict(params)
        self._configure_vad_options(request)
        return self.handler.translate_audio(request)

    @fastapi.get("/progress/{progress_id}")
    def get_progress(self, progress_id: str) -> ProgressResponse:
        return self.progress_handler.get_progress(progress_id)

    def _served_model_object(self) -> ModelObject:
        """The single model this API serves, as a static OpenAI-style ModelObject."""
        return ModelObject(
            id=self.config.faster_whisper.default_model_name,
            created=1668556800,  # large-v2 release (2022-11-16); static, no HF query.
            object_="model",
            owned_by="Systran",
            language=[],
        )

    @fastapi.get("/models")
    def get_models(self) -> ModelListResponse:
        return ModelListResponse(data=[self._served_model_object()])

    @fastapi.get("/models/{model_name:path}")
    def get_model(
        self,
        # examples=[...] is evaluated at class-definition time, so it reads the import-time
        # config global rather than the injected self.config used by the runtime checks below.
        model_name: Annotated[str, FastAPIPath(examples=[faster_whisper_config.default_model_name])],
    ) -> ModelObject:
        served = self.config.faster_whisper.default_model_name
        if model_name != served:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found. Only '{served}' is served.",
            )
        return self._served_model_object()

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
