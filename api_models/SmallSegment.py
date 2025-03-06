from pydantic import BaseModel


class SmallSegment(BaseModel):
    start: float
    end: float
    text: str
    speaker: str | None = None
