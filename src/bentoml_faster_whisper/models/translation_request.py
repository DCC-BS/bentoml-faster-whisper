from pathlib import Path
from typing import Annotated, Any

from bentoml.exceptions import InvalidArgument
from bentoml.validators import ContentType

from bentoml_faster_whisper.models.decode_params import DecodeParams


class TranslationRequest(DecodeParams):
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TranslationRequest":
        # Mirror TranscriptionRequest.from_dict so both entrypoints normalize bad input
        # to InvalidArgument (a 4xx) instead of leaking a raw TypeError.
        try:
            return cls.model_validate(d)
        except TypeError as e:
            raise InvalidArgument(str(e))

    file: Annotated[Path, ContentType("audio/*")]
