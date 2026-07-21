"""Lazy, multiprocess-safe Prometheus metric registry.

Every metric is created on **first access**, never at import time. This is
load-bearing: ``prometheus_client`` freezes its storage backend the first time
it is imported (``values.py`` runs ``ValueClass = get_value_class()``), picking
the mmap-backed multiprocess value class only if ``PROMETHEUS_MULTIPROC_DIR`` is
already in ``os.environ``. BentoML workers set that env var *after* importing
the service module, so importing ``prometheus_client`` (or building a metric) at
module-import time locks it into single-process mode and the metric never
reaches the mmap ``.db`` files that BentoML's ``/metrics`` endpoint scrapes via
``MultiProcessCollector`` — the metric silently vanishes from Grafana.

Deferring both the import and the construction to first use (which always
happens inside the worker, after the env var exists) keeps every custom metric
mmap-backed and visible. Each accessor is memoised with ``lru_cache`` so it
returns a singleton and never triggers prometheus_client's "Duplicated
timeseries in CollectorRegistry" error.
"""

import functools
import time

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
SPEAKER_COUNT_BUCKETS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, float("inf")]
DIARIZATION_DURATION_BUCKETS_S = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, float("inf")]
MODEL_LOAD_DURATION_BUCKETS_S = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, float("inf")]


@functools.lru_cache(maxsize=1)
def audio_length():
    from prometheus_client import Histogram

    return Histogram(
        name="input_audio_length_seconds",
        documentation="Input audio length in seconds",
        buckets=AUDIO_INPUT_LENGTH_BUCKETS_S,
    )


@functools.lru_cache(maxsize=1)
def realtime_factor():
    from prometheus_client import Histogram

    return Histogram(
        name="realtime_factor",
        documentation="Realtime factor, e.g. avg. audio seconds transcribed per wall-clock second",
        buckets=REALTIME_FACTOR_BUCKETS,
    )


@functools.lru_cache(maxsize=1)
def diarization_duration():
    from prometheus_client import Histogram

    return Histogram(
        name="diarization_duration_seconds",
        documentation="Wall-clock time spent in pyannote speaker diarization",
        buckets=DIARIZATION_DURATION_BUCKETS_S,
    )


@functools.lru_cache(maxsize=1)
def speaker_count():
    from prometheus_client import Histogram

    return Histogram(
        name="diarization_speaker_count",
        documentation="Number of distinct speakers detected per diarized request",
        buckets=SPEAKER_COUNT_BUCKETS,
    )


@functools.lru_cache(maxsize=1)
def detected_language():
    from prometheus_client import Counter

    return Counter(
        name="detected_language",
        documentation="Count of requests per detected/reported transcription language",
        labelnames=["language"],
    )


@functools.lru_cache(maxsize=1)
def transcription_failures():
    from prometheus_client import Counter

    return Counter(
        name="transcription_failures",
        documentation="Count of transcription failures by pipeline stage and error type",
        labelnames=["stage", "error_type"],
    )


@functools.lru_cache(maxsize=1)
def model_load_duration():
    from prometheus_client import Histogram

    return Histogram(
        name="model_load_duration_seconds",
        documentation="Wall-clock time to load a Whisper model into memory",
        buckets=MODEL_LOAD_DURATION_BUCKETS_S,
    )


@functools.lru_cache(maxsize=1)
def models_loaded():
    from prometheus_client import Gauge

    return Gauge(
        name="models_loaded",
        documentation="Number of Whisper models currently resident in memory",
    )


@functools.lru_cache(maxsize=1)
def model_loads_total():
    from prometheus_client import Counter

    return Counter(
        name="model_loads",
        documentation="Total number of Whisper model loads",
    )


# --- Recording helpers -------------------------------------------------------
# Every decode path shares the same instrumentation contract, so the label
# conventions and the divide-by-zero guard live here rather than being
# reconstructed at each handler call site.


def record_failure(stage: str, exc: BaseException) -> None:
    """Count one pipeline failure, labelled by stage and exception type."""
    transcription_failures().labels(stage, type(exc).__name__).inc()


def observe_decode(duration_s: float, language: str | None) -> None:
    """Record the decoded audio length and detected/reported language for one request."""
    audio_length().observe(duration_s)
    detected_language().labels(language or "unknown").inc()


def observe_realtime_factor(t0: float, duration_s: float) -> None:
    """Record audio-seconds decoded per wall-clock second, timed from ``t0``."""
    elapsed = time.perf_counter() - t0
    if elapsed > 0 and duration_s > 0:
        realtime_factor().observe(duration_s / elapsed)
