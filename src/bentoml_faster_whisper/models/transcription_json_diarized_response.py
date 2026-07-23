from pydantic import BaseModel

from bentoml_faster_whisper.models.small_segment import SmallSegment
from bentoml_faster_whisper.utils.core import Segment


class TranscriptionJsonDiarizedResponse(BaseModel):
    segments: list[SmallSegment]

    @classmethod
    def from_segments(cls, segments: list[Segment]) -> "TranscriptionJsonDiarizedResponse":
        return cls(
            segments=[
                SmallSegment(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                    speaker=segment.speaker,
                    language=segment.language,
                )
                for segment in segments
            ]
        )
