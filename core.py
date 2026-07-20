from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel


if TYPE_CHECKING:
    from collections.abc import Iterable

    import faster_whisper.transcribe


class Word(BaseModel):
    start: float
    end: float
    word: str
    probability: float
    speaker: str | None = None

    @classmethod
    def from_segments(cls, segments: Iterable[Segment]) -> list[Word]:
        words: list[Word] = []
        for segment in segments:
            # NOTE: a temporary "fix" for https://github.com/fedirz/faster-whisper-server/issues/58.
            # TODO: properly address the issue
            assert segment.words is not None, (
                "Segment must have words. If you are using an API ensure `timestamp_granularities[]=word` is set"
            )
            words.extend(segment.words)
        return words


class Segment(BaseModel):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: list[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: list[Word] | None
    speaker: str | None = None
    language: str | None = None
    """Per-segment decode language; only set when the language was auto-detected per
    speech region (diarized request without an explicit language)."""

    @classmethod
    def from_faster_whisper_segments(cls, segments: Iterable[faster_whisper.transcribe.Segment]) -> Iterable[Segment]:
        for segment in segments:
            yield cls(
                id=segment.id,
                seek=segment.seek,
                start=segment.start,
                end=segment.end,
                text=segment.text,
                tokens=segment.tokens,
                temperature=segment.temperature if segment.temperature is not None else 0.0,
                avg_logprob=segment.avg_logprob,
                compression_ratio=segment.compression_ratio,
                no_speech_prob=segment.no_speech_prob,
                words=[
                    Word(
                        start=word.start,
                        end=word.end,
                        word=word.word,
                        probability=word.probability,
                    )
                    for word in segment.words
                ]
                if segment.words is not None
                else None,
            )


def segments_to_text(segments: Iterable[Segment]) -> str:
    return "".join(segment.text for segment in segments).strip()


def _format_timestamp(ts: float, millis_sep: str) -> str:
    hours = ts // 3600
    minutes = (ts % 3600) // 60
    seconds = ts % 60
    milliseconds = (ts * 1000) % 1000
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}{millis_sep}{int(milliseconds):03d}"


def srt_format_timestamp(ts: float) -> str:
    return _format_timestamp(ts, ",")


def vtt_format_timestamp(ts: float) -> str:
    return _format_timestamp(ts, ".")


def segments_to_vtt(segment: Segment, i: int) -> str:
    start = segment.start if i > 0 else 0.0
    result = f"{vtt_format_timestamp(start)} --> {vtt_format_timestamp(segment.end)}\n{segment.text}\n\n"

    if i == 0:
        return f"WEBVTT\n\n{result}"
    else:
        return result


def segments_to_srt(segment: Segment, i: int) -> str:
    return f"{i + 1}\n{srt_format_timestamp(segment.start)} --> {srt_format_timestamp(segment.end)}\n{segment.text}\n\n"
