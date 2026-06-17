import json
import logging
import subprocess
from pathlib import Path

from prometheus_client import Histogram

logger = logging.getLogger(__name__)

AUDIO_INPUT_LENGTH_BUCKETS_S = [
    10.0,
    30.0,
    60.0,
    60.0 * 2,
    60.0 * 5,
    60.0 * 7,
    60.0 * 10,
    60.0 * 20,
    60.0 * 30,
    60.0 * 60,
    60.0 * 60 * 2,
    float("inf"),
]
REALTIME_FACTOR_BUCKETS = [
    0.5,
    1.0,
    3.0,
    5.0,
    10.0,
    20.0,
    30.0,
    40.0,
    50.0,
    float("inf"),
]

input_audio_length_histogram = Histogram(
    name="input_audio_length_seconds",
    documentation="Input audio length in seconds",
    buckets=AUDIO_INPUT_LENGTH_BUCKETS_S,
)

realtime_factor_histogram = Histogram(
    name="realtime_factor",
    documentation="Realtime factor, e.g. avg. audio seconds transcribed per second",
    buckets=REALTIME_FACTOR_BUCKETS,
)


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
        )
        return float(json.loads(result.stdout)["format"]["duration"])
    except (subprocess.CalledProcessError, OSError, KeyError, ValueError) as e:
        logger.warning("ffprobe duration probe failed for %s: %s", file, e)
        return 0.0
