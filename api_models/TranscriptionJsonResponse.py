from pydantic import BaseModel

from core import Segment, segments_to_text


# https://platform.openai.com/docs/api-reference/audio/json-object
class TranscriptionJsonResponse(BaseModel):
    text: str

    @classmethod
    def from_segments(cls, segments: list[Segment]) -> "TranscriptionJsonResponse":
        return cls(text=segments_to_text(segments))
