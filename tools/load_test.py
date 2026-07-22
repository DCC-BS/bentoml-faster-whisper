#!/usr/bin/env python3
"""Load and E2E Performance Test Suite for bentoml-faster-whisper.

Measures single-request latency, Real-Time Factor (RTF), and multi-concurrent
request throughput/percentile latencies using long test audio assets.
Saves structured JSON & CSV benchmarks to track performance across settings.
"""

import argparse
import datetime
import json
import logging
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import av
import numpy as np

# Add src/ to python path if running standalone
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bentoml_faster_whisper.models.enums import ResponseFormat
from bentoml_faster_whisper.models.transcription_request import TranscriptionRequest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("load_test")

DEFAULT_ASSETS_DIR = PROJECT_ROOT / "tests" / "assets"
DEFAULT_LONG_AUDIO = (
    DEFAULT_ASSETS_DIR
    / "Regionaljournal_Basel_Baselland_radio_AUDI20260710_NR_0011_1004cf68be404710b30a996ac4f1ff93.mp3"
)
DEFAULT_RADIO_AUDIO = DEFAULT_LONG_AUDIO


def get_audio_duration_seconds(audio_path: Path) -> float:
    """Get exact duration in seconds of an audio file using PyAV or soundfile fallback."""
    try:
        with av.open(str(audio_path)) as container:
            stream = container.streams.audio[0]
            if container.duration is not None:
                tb = getattr(av, "TIME_BASE", 1_000_000)
                return float(container.duration / tb)
            if stream.duration is not None and stream.time_base is not None:
                return float(stream.duration * stream.time_base)
    except Exception as e:
        logger.warning("Could not determine duration via PyAV for %s: %s", audio_path, e)
    return 0.0


def get_git_commit_hash() -> str:
    """Return current git commit hash if available."""
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def run_single_request_in_process(
    service: Any,
    audio_path: Path,
    diarization: bool = False,
    response_format: ResponseFormat = ResponseFormat.JSON,
) -> dict[str, Any]:
    """Execute a single transcription request against in-process FasterWhisper service."""
    req_dict = {
        "file": audio_path,
        "response_format": response_format,
        "diarization": diarization,
    }
    request = TranscriptionRequest.model_validate(req_dict)
    params = request.model_dump()

    start_time = time.perf_counter()
    success = True
    error_msg = None
    transcript_len = 0

    try:
        response = service.transcribe(**params)
        if isinstance(response, str):
            transcript_len = len(response)
        elif hasattr(response, "text"):
            transcript_len = len(response.text)
        elif isinstance(response, dict) and "text" in response:
            transcript_len = len(response["text"])
    except Exception as e:
        success = False
        error_msg = str(e)
        logger.error("Request failed: %s", e)

    latency_s = time.perf_counter() - start_time
    return {
        "latency_s": latency_s,
        "success": success,
        "error": error_msg,
        "transcript_len": transcript_len,
    }


def run_single_request_http(
    base_url: str,
    audio_path: Path,
    diarization: bool = False,
    response_format: str = "json",
) -> dict[str, Any]:
    """Execute a single transcription HTTP POST request against live BentoML server."""
    import requests

    endpoint = f"{base_url.rstrip('/')}/v1/audio/transcriptions"
    data = {
        "diarization": "true" if diarization else "false",
        "response_format": response_format,
    }

    start_time = time.perf_counter()
    success = True
    error_msg = None
    transcript_len = 0

    try:
        with open(audio_path, "rb") as f:
            resp = requests.post(endpoint, files={"file": f}, data=data, timeout=3000)
        if resp.status_code == 200:
            transcript_len = len(resp.text)
        else:
            success = False
            error_msg = f"HTTP {resp.status_code}: {resp.text}"
    except Exception as e:
        success = False
        error_msg = str(e)

    latency_s = time.perf_counter() - start_time
    return {
        "latency_s": latency_s,
        "success": success,
        "error": error_msg,
        "transcript_len": transcript_len,
    }


def run_concurrency_batch(
    service_or_url: Any,
    is_http: bool,
    audio_path: Path,
    concurrency: int,
    total_requests: int,
    diarization: bool,
) -> dict[str, Any]:
    """Run a batch of concurrent requests and compute aggregated latency & throughput metrics."""
    logger.info(
        "Starting concurrency batch: concurrency=%d, total_requests=%d, diarization=%s",
        concurrency,
        total_requests,
        diarization,
    )

    batch_start = time.perf_counter()
    results = []

    def task_worker(_: int) -> dict[str, Any]:
        if is_http:
            return run_single_request_http(service_or_url, audio_path, diarization=diarization)
        return run_single_request_in_process(service_or_url, audio_path, diarization=diarization)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(task_worker, i) for i in range(total_requests)]
        for future in as_completed(futures):
            results.append(future.result())

    batch_duration_s = time.perf_counter() - batch_start
    latencies = [r["latency_s"] for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]

    throughput_req_per_sec = len(results) / batch_duration_s if batch_duration_s > 0 else 0.0

    if latencies:
        mean_latency = float(np.mean(latencies))
        p50_latency = float(np.percentile(latencies, 50))
        p90_latency = float(np.percentile(latencies, 90))
        p99_latency = float(np.percentile(latencies, 99))
        min_latency = float(np.min(latencies))
        max_latency = float(np.max(latencies))
    else:
        mean_latency = p50_latency = p90_latency = p99_latency = min_latency = max_latency = 0.0

    return {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "successful_requests": len(latencies),
        "failed_requests": len(failures),
        "total_batch_duration_s": batch_duration_s,
        "throughput_req_per_sec": throughput_req_per_sec,
        "latency_stats": {
            "mean_s": mean_latency,
            "p50_s": p50_latency,
            "p90_s": p90_latency,
            "p99_s": p99_latency,
            "min_s": min_latency,
            "max_s": max_latency,
        },
        "errors": [f["error"] for f in failures],
    }


def main():
    parser = argparse.ArgumentParser(description="Load and Performance Benchmark Test Suite")
    parser.add_argument(
        "--audio",
        type=str,
        default=str(DEFAULT_LONG_AUDIO),
        help="Path to audio file (default: long_example_audio.mp3)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="",
        help="BentoML HTTP endpoint URL (e.g. http://localhost:50001). If empty, runs in-process.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Concurrency levels to test (default: 1 2 4)",
    )
    parser.add_argument(
        "--requests-per-level",
        type=int,
        default=4,
        help="Number of requests per concurrency test level (default: 4)",
    )
    parser.add_argument(
        "--diarization",
        choices=["true", "false", "both"],
        default="both",
        help="Test diarization setting (default: both)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="load_test_results.json",
        help="Output JSON results file path (default: load_test_results.json)",
    )

    args = parser.parse_args()

    audio_path = Path(args.audio).resolve()
    if not audio_path.exists():
        logger.error("Audio file not found: %s", audio_path)
        sys.exit(1)

    audio_duration_s = get_audio_duration_seconds(audio_path)
    logger.info("Loaded Audio File: %s (Duration: %.2fs)", audio_path.name, audio_duration_s)

    service_or_url = None
    is_http = bool(args.url)

    if is_http:
        service_or_url = args.url
        logger.info("Running load test against HTTP endpoint: %s", service_or_url)
    else:
        logger.info("Loading in-process FasterWhisper service...")
        from bentoml_faster_whisper.service import FasterWhisper

        service_or_url = FasterWhisper()

    diarization_modes = []
    if args.diarization in ("false", "both"):
        diarization_modes.append(False)
    if args.diarization in ("true", "both"):
        diarization_modes.append(True)

    benchmark_runs = []

    for diarization in diarization_modes:
        mode_str = "with_diarization" if diarization else "without_diarization"
        logger.info("=== Running Benchmark Mode: %s ===", mode_str)

        # 1. Warmup / Single Request Baseline Test
        logger.info("Executing Single Request Baseline Test (%s)...", mode_str)
        if is_http:
            baseline_result = run_single_request_http(service_or_url, audio_path, diarization=diarization)
        else:
            baseline_result = run_single_request_in_process(service_or_url, audio_path, diarization=diarization)

        rtf = (baseline_result["latency_s"] / audio_duration_s) if audio_duration_s > 0 else 0.0
        logger.info(
            "Baseline Single Request Latency: %.3fs | Real-Time Factor (RTF): %.3fx (Lower is faster)",
            baseline_result["latency_s"],
            rtf,
        )

        single_run_summary = {
            "mode": mode_str,
            "diarization": diarization,
            "test_type": "single_request_baseline",
            "audio_file": audio_path.name,
            "audio_duration_s": audio_duration_s,
            "latency_s": baseline_result["latency_s"],
            "real_time_factor_rtf": rtf,
            "success": baseline_result["success"],
        }
        benchmark_runs.append(single_run_summary)

        # 2. Multi Concurrent Request Load Tests
        for c in args.concurrency:
            total_reqs = max(c, args.requests_per_level)
            batch_metrics = run_concurrency_batch(
                service_or_url=service_or_url,
                is_http=is_http,
                audio_path=audio_path,
                concurrency=c,
                total_requests=total_reqs,
                diarization=diarization,
            )

            # Compute effective concurrent RTF
            batch_rtf = (batch_metrics["latency_stats"]["mean_s"] / audio_duration_s) if audio_duration_s > 0 else 0.0

            batch_summary = {
                "mode": mode_str,
                "diarization": diarization,
                "test_type": "concurrent_load_test",
                "audio_file": audio_path.name,
                "audio_duration_s": audio_duration_s,
                "concurrency": c,
                "total_requests": total_reqs,
                "successful_requests": batch_metrics["successful_requests"],
                "failed_requests": batch_metrics["failed_requests"],
                "throughput_req_per_sec": batch_metrics["throughput_req_per_sec"],
                "mean_latency_s": batch_metrics["latency_stats"]["mean_s"],
                "p50_latency_s": batch_metrics["latency_stats"]["p50_s"],
                "p90_latency_s": batch_metrics["latency_stats"]["p90_s"],
                "p99_latency_s": batch_metrics["latency_stats"]["p99_s"],
                "real_time_factor_rtf": batch_rtf,
            }
            benchmark_runs.append(batch_summary)

    # Compile Master Benchmark Output Document
    output_data = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": get_git_commit_hash(),
        "audio_file": audio_path.name,
        "audio_duration_s": audio_duration_s,
        "test_environment": {
            "mode": "http" if is_http else "in_process",
            "url": args.url,
            "python_version": sys.version,
        },
        "results": benchmark_runs,
    }

    output_file = Path(args.output).resolve()
    # Read existing results if tracking file exists to append history
    history = []
    if output_file.exists():
        try:
            with open(output_file, "r") as f:
                existing = json.load(f)
                if isinstance(existing, list):
                    history = existing
                elif isinstance(existing, dict):
                    history = [existing]
        except Exception:
            history = []

    history.append(output_data)

    with open(output_file, "w") as f:
        json.dump(history, f, indent=2)

    logger.info("Successfully wrote load test results to %s", output_file)

    # Print clean summary table to stdout
    print("\n" + "=" * 85)
    print(f"LOAD TEST SUMMARY RESULT ({audio_path.name}, duration: {audio_duration_s:.1f}s)")
    print("=" * 85)
    print(
        f"{'Mode':<20} | {'Concurrency':<12} | {'Mean Latency':<12} | {'P90 Latency':<12} | {'Throughput':<12} | {'RTF':<8}"
    )
    print("-" * 85)
    for run in benchmark_runs:
        c_str = str(run.get("concurrency", 1))
        lat_str = f"{run.get('mean_latency_s', run.get('latency_s', 0)):.2f}s"
        p90_str = f"{run.get('p90_latency_s', 0):.2f}s" if "p90_latency_s" in run else "N/A"
        tp_str = f"{run.get('throughput_req_per_sec', 0):.2f} req/s" if "throughput_req_per_sec" in run else "N/A"
        rtf_str = f"{run.get('real_time_factor_rtf', 0):.3f}x"
        print(f"{run['mode']:<20} | {c_str:<12} | {lat_str:<12} | {p90_str:<12} | {tp_str:<12} | {rtf_str:<8}")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
