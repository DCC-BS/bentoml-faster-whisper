"""Performance Load Test Suite.

Executes end-to-end performance measurements on long audio assets from tests/assets/
for single requests and multi-concurrent requests (with and without diarization).
Writes structured tracking results into load_test_results.json.
Only executed when explicitly running 'make performance' or 'pytest -m performance'.
"""

import json
from pathlib import Path
import pytest

from tools.load_test import (
    DEFAULT_LONG_AUDIO,
    DEFAULT_RADIO_AUDIO,
    get_audio_duration_seconds,
    run_concurrency_batch,
    run_single_request_in_process,
)

# Only marked as 'performance' - strictly isolated from 'integration' and 'unit' tests!
pytestmark = pytest.mark.performance


@pytest.fixture(scope="module")
def long_audio_path() -> Path:
    assert DEFAULT_LONG_AUDIO.exists(), f"Long audio asset missing: {DEFAULT_LONG_AUDIO}"
    return DEFAULT_LONG_AUDIO


@pytest.fixture(scope="module")
def radio_audio_path() -> Path:
    assert DEFAULT_RADIO_AUDIO.exists(), f"Radio audio asset missing: {DEFAULT_RADIO_AUDIO}"
    return DEFAULT_RADIO_AUDIO


def test_single_request_performance_measurement(faster_whisper_service, long_audio_path):
    """Measure single request latency, RTF, and Speedup on long audio."""
    duration_s = get_audio_duration_seconds(long_audio_path)
    result = run_single_request_in_process(faster_whisper_service, long_audio_path, diarization=False)

    assert result["success"] is True, f"Request failed: {result.get('error')}"
    assert result["latency_s"] > 0
    assert result["transcript_len"] > 0

    rtf = result["latency_s"] / duration_s if duration_s > 0 else 0.0
    speedup = duration_s / result["latency_s"] if result["latency_s"] > 0 else 0.0
    print(
        f"\n[Single Request Performance] Audio: {long_audio_path.name} | Duration: {duration_s:.1f}s | "
        f"Latency: {result['latency_s']:.2f}s | RTF: {rtf:.3f}x | Speedup: {speedup:.2f}x"
    )


def test_multi_concurrent_requests_load_test(faster_whisper_service, long_audio_path):
    """Measure multi-concurrent request throughput and percentile latencies."""
    concurrency = 2
    total_requests = 4
    duration_s = get_audio_duration_seconds(long_audio_path)

    batch_metrics = run_concurrency_batch(
        service_or_url=faster_whisper_service,
        is_http=False,
        audio_path=long_audio_path,
        concurrency=concurrency,
        total_requests=total_requests,
        diarization=False,
    )

    assert batch_metrics["successful_requests"] == total_requests
    assert batch_metrics["failed_requests"] == 0
    assert batch_metrics["throughput_req_per_sec"] > 0

    stats = batch_metrics["latency_stats"]
    print(
        f"\n[Concurrent Load Test] Concurrency: {concurrency} | Total Reqs: {total_requests} | "
        f"Throughput: {batch_metrics['throughput_req_per_sec']:.2f} req/s | "
        f"Mean Latency: {stats['mean_s']:.2f}s | P90 Latency: {stats['p90_s']:.2f}s"
    )

    # Save results to load_test_results.json for history tracking
    output_file = Path("load_test_results.json").resolve()
    record = {
        "audio": long_audio_path.name,
        "concurrency": concurrency,
        "total_requests": total_requests,
        "throughput_req_per_sec": batch_metrics["throughput_req_per_sec"],
        "mean_latency_s": stats["mean_s"],
        "p90_latency_s": stats["p90_s"],
    }

    history = []
    if output_file.exists():
        try:
            with open(output_file, "r") as f:
                history = json.load(f)
        except Exception:
            history = []
    history.append(record)
    with open(output_file, "w") as f:
        json.dump(history, f, indent=2)
