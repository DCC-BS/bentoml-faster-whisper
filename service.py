from typing import TYPE_CHECKING, Annotated, AsyncGenerator, List, Optional, Union

import bentoml
import huggingface_hub
from bentoml.validators import ContentType

from core import Segment
from model_manager import WhisperModelManager
from utils import (
    Language,
    ModelListResponse,
    ModelObject,
    ResponseFormat,
    segments_to_response,
    segments_to_streaming_response,
)
from fastapi import FastAPI

fastapi = FastAPI()

if TYPE_CHECKING:
    from huggingface_hub.hf_api import ModelInfo

from http import HTTPStatus
from pathlib import Path
from pydantic import Field

from bentoml.exceptions import NotFound

LANGUAGE_CODE = "de"

ModelName = Annotated[
    str,
    Field(
        description="The ID of the model. You can get a list of available models by calling `/v1/models`.",
        examples=[
            "Systran/faster-distil-whisper-large-v3",
            "bofenghuang/whisper-large-v2-cv11-french-ct2",
        ],
    ),
]


@bentoml.service(
    traffic={"timeout": 30},
    resources={
        "gpu": 1,
        "memory": "8Gi",
    },
)
class FasterWhisper:
    def __init__(self):
        import torch

        self.batch_size = 16
        gpu = torch.cuda.is_available()
        whisper_config = dict(
            device="cuda" if gpu else "cpu",
            device_index=0,
            compute_type="fp16" if gpu else "int8",
            cpu_threads=0,
            num_workers=1,
        )
        self.model_manager = WhisperModelManager(whisper_config)

    @bentoml.api(route="/stream/transcribe")
    async def streaming_transcribe(
        self,
        file: Annotated[Path, ContentType("audio/mp3")],
        model: Optional[ModelName] = Field(
            default=None, description="Whisper model to load"
        ),
        language: Optional[Language] = Field(
            default=None,
            description='The language spoken in the audio. It should be a language code such as "en" or "fr". If not set, the language will be detected in the first 30 seconds of audio.',
        ),
        prompt: Optional[float] = Field(
            default=None,
            description="Optional text string or iterable of token ids to provide as a prompt for the first window.",
        ),
        response_format: Optional[ResponseFormat] = Field(
            default=None,
            description=f"One of: {[format.name for format in ResponseFormat]}",
        ),
        temperature: Union[List[float], float] = Field(
            default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            description="Temperature values as a list or single float",
        ),
    ) -> AsyncGenerator[str, None]:
        segments, info = self.model.transcribe(file, batch_size=self.batch_size)

        async for segment in segments:
            yield segment.text

    @bentoml.api(route="/v1/audio/transcriptions")
    def transcribe(
        self,
        file: Annotated[Path, ContentType("audio/mp3")],
        model: Optional[ModelName] = Field(
            default="large-v3", description="Whisper model to load"
        ),
        language: Optional[Language] = Field(
            default=None,
            description='The language spoken in the audio. It should be a language code such as "en" or "fr". If not set, the language will be detected in the first 30 seconds of audio.',
        ),
        prompt: Optional[float] = Field(
            default=None,
            description="Optional text string or iterable of token ids to provide as a prompt for the first window.",
        ),
        response_format: Optional[ResponseFormat] = Field(
            default=None,
            description=f"One of: {[format.name for format in ResponseFormat]}",
        ),
        temperature: Union[List[float], float] = Field(
            default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            description="Temperature values as a list or single float",
        ),
        stream: bool = Field(default=False, description="Use streaming mode"),
    ) -> (
        Annotated[str, bentoml.validators.ContentType("text/plain")]
        | Annotated[str, bentoml.validators.ContentType("application/json")]
        | Annotated[str, bentoml.validators.ContentType("text/vtt")]
        | Annotated[str, bentoml.validators.ContentType("text/event-stream")]
    ):
        with self.model_manager.load_model(model) as whisper:
            segments, transcription_info = whisper.transcribe(
                file,
                batch_size=self.batch_size,
                initial_prompt=prompt,
                language=language,
                temperature=temperature,
            )
        segments = Segment.from_faster_whisper_segments(segments)
        if stream:
            return segments_to_streaming_response(
                segments, transcription_info, response_format
            )
        else:
            return segments_to_response(segments, transcription_info, response_format)

    @fastapi.get("/v1/models")
    def get_models() -> ModelListResponse:
        models = huggingface_hub.list_models(
            library="ctranslate2", tags="automatic-speech-recognition", cardData=True
        )
        models = list(models)
        models.sort(key=lambda model: model.downloads, reverse=True)  # type: ignore  # noqa: PGH003
        transformed_models: list[ModelObject] = []
        for model in models:
            assert model.created_at is not None
            assert model.card_data is not None
            assert model.card_data.language is None or isinstance(
                model.card_data.language, str | list
            )
            if model.card_data.language is None:
                language = []
            elif isinstance(model.card_data.language, str):
                language = [model.card_data.language]
            else:
                language = model.card_data.language
            transformed_model = ModelObject(
                id=model.id,
                created=int(model.created_at.timestamp()),
                object_="model",
                owned_by=model.id.split("/")[0],
                language=language,
            )
            transformed_models.append(transformed_model)
        return ModelListResponse(data=transformed_models)

    @fastapi.get("/v1/models/{model_name:path}")
    # NOTE: `examples` doesn't work https://github.com/tiangolo/fastapi/discussions/10537
    def get_model(
        model_name: Annotated[
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
            raise NotFound(
                error_code=HTTPStatus.NOT_FOUND, detail="Model doesn't exists"
            )
        exact_match: ModelInfo | None = None
        for model in models:
            if model.id == model_name:
                exact_match = model
                break
        if exact_match is None:
            raise NotFound(
                error_code=HTTPStatus.NOT_FOUND,
                detail=f"Model doesn't exists. Possible matches: {', '.join([model.id for model in models])}",
            )
        assert exact_match.created_at is not None
        assert exact_match.card_data is not None
        assert exact_match.card_data.language is None or isinstance(
            exact_match.card_data.language, str | list
        )
        if exact_match.card_data.language is None:
            language = []
        elif isinstance(exact_match.card_data.language, str):
            language = [exact_match.card_data.language]
        else:
            language = exact_match.card_data.language
        return ModelObject(
            id=exact_match.id,
            created=int(exact_match.created_at.timestamp()),
            object_="model",
            owned_by=exact_match.id.split("/")[0],
            language=language,
        )
