from pathlib import Path

from prometheus_client import Histogram
from pydub import AudioSegment

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


def get_audio_duration(file: Path):
    """Gets the duration of an audio file in seconds."""
    duration = AudioSegment.from_file(file).duration_seconds
    return duration
