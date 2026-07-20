import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


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
