import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Callable, TypeVar

from helpers.logger import get_logger

logger = logging.getLogger(__name__)

_NumberT = TypeVar("_NumberT", int, float)


def clamp(value: float, low: float, high: float) -> float:
    """Constrain ``value`` to the closed interval ``[low, high]``."""
    return min(max(value, low), high)


def positive_env(name: str, default: _NumberT, cast: Callable[[str], _NumberT]) -> _NumberT:
    """Parse a positive-number env var, falling back to default on missing/invalid/non-positive values."""
    raw = os.getenv(name)
    if raw is None:
        return default
    slog = get_logger(__name__)
    try:
        value = cast(raw)
    except ValueError:
        slog.warning("Invalid env var value; using default", name=name, raw=raw, default=default)
        return default
    if value <= 0:
        slog.warning("Env var must be > 0; using default", name=name, value=value, default=default)
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
        logger.warning("ffprobe duration probe failed for %s: %s", file, e)
        return 0.0
