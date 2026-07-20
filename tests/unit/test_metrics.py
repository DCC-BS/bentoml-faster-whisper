"""Guards for the lazy, multiprocess-safe metric registry (helpers/metrics.py).

The regression these protect against: creating Prometheus metrics at module-import
time freezes prometheus_client into single-process mode before BentoML sets
PROMETHEUS_MULTIPROC_DIR, so the metrics never reach the mmap files the /metrics
endpoint scrapes and silently disappear from Grafana. See helpers/metrics.py.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from api_models.enums import ResponseFormat
from api_models.TranscriptionRequest import TranscriptionRequest
from helpers import metrics

ASSETS = Path(__file__).resolve().parent.parent / "assets"
AUDIO = ASSETS / "example_audio.mp3"


def test_import_does_not_eagerly_import_prometheus_client():
    """Importing helpers.metrics must not pull in prometheus_client — that eager
    import is exactly what breaks multiprocess metric collection. Checked in a fresh
    subprocess because prometheus_client is already imported in this test session."""
    code = "import helpers.metrics, sys; assert 'prometheus_client' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "accessor",
    [
        metrics.audio_length,
        metrics.realtime_factor,
        metrics.diarization_duration,
        metrics.speaker_count,
        metrics.detected_language,
        metrics.transcription_failures,
        metrics.model_load_duration,
        metrics.models_loaded,
        metrics.model_loads_total,
        metrics.model_unloads_total,
    ],
)
def test_accessor_is_idempotent_singleton(accessor):
    """Each accessor returns the same registered collector every call — a second
    construction would raise 'Duplicated timeseries in CollectorRegistry'."""
    assert accessor() is accessor()


@pytest.mark.model
def test_transcription_records_metrics(handler):
    from prometheus_client import REGISTRY

    request = TranscriptionRequest.model_validate(
        {"file": AUDIO, "language": "en", "response_format": ResponseFormat.JSON}
    )

    before = REGISTRY.get_sample_value("input_audio_length_seconds_count") or 0.0

    handler.transcribe_audio(request)

    after = REGISTRY.get_sample_value("input_audio_length_seconds_count") or 0.0
    assert after == before + 1, "transcription should record one input_audio_length observation"

    # Language counter and realtime-factor observation must both fire for the same request.
    assert (REGISTRY.get_sample_value("detected_language_total", {"language": "en"}) or 0.0) >= 1
    assert (REGISTRY.get_sample_value("realtime_factor_count") or 0.0) >= 1
