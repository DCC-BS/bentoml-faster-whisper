from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Callable, TypeVar

from pydantic import BaseModel

from bentoml_faster_whisper.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable

    import faster_whisper.transcribe

logger = get_logger(__name__)

_NumberT = TypeVar("_NumberT", int, float)


def clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def positive_env(name: str, default: _NumberT, cast: Callable[[str], _NumberT]) -> _NumberT:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = cast(raw)
    except ValueError:
        logger.warning("Invalid env var value; using default", name=name, raw=raw, default=default)
        return default
    if value <= 0:
        logger.warning("Env var must be > 0; using default", name=name, value=value, default=default)
        return default
    return value


def get_audio_duration(file: Path) -> float:
    """Gets the duration of an audio file in seconds.

    Uses ffprobe to read container metadata only, avoiding decoding the whole file into RAM just for a metric.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(file),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        return float(json.loads(result.stdout)["format"]["duration"])
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError, KeyError, ValueError) as e:
        logger.warning("ffprobe duration probe failed", file=str(file), error=str(e))
        return 0.0


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
            # A word-less segment (faster-whisper can emit one even with word_timestamps=True,
            # and restore_and_split yields the word-less branch) simply contributes no words —
            # it must not crash the whole verbose_json response.
            if segment.words is None:
                continue
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
    total_ms = round(ts * 1000)
    hours = total_ms // 3_600_000
    minutes = (total_ms // 60_000) % 60
    seconds = (total_ms // 1000) % 60
    milliseconds = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{millis_sep}{milliseconds:03d}"


def srt_format_timestamp(ts: float) -> str:
    return _format_timestamp(ts, ",")


def vtt_format_timestamp(ts: float) -> str:
    return _format_timestamp(ts, ".")


def segments_to_vtt(segment: Segment, i: int) -> str:
    result = f"{vtt_format_timestamp(segment.start)} --> {vtt_format_timestamp(segment.end)}\n{segment.text}\n\n"

    if i == 0:
        return f"WEBVTT\n\n{result}"
    else:
        return result


def segments_to_srt(segment: Segment, i: int) -> str:
    return f"{i + 1}\n{srt_format_timestamp(segment.start)} --> {srt_format_timestamp(segment.end)}\n{segment.text}\n\n"
