from pydantic import BaseModel

from api_models.SmallSegment import SmallSegment
from core import Segment


class TranscriptionJsonDiariexedResponse(BaseModel):
    segments: list[SmallSegment]

    @classmethod
    def from_segments(cls, segments: list[Segment]) -> "TranscriptionJsonDiariexedResponse":
        return cls(
            segments=[
                SmallSegment(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                    speaker=segment.speaker,
                )
                for segment in segments
            ]
        )
