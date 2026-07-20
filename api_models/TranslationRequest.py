from pathlib import Path
from typing import Annotated

from bentoml.validators import ContentType

from api_models.decode_params import DecodeParams


class TranslationRequest(DecodeParams):
    @classmethod
    def from_dict(cls, d: dict) -> "TranslationRequest":
        return cls(**d)

    file: Annotated[Path, ContentType("audio/*")]
