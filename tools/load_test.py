#!/usr/bin/env python3
"""Load and E2E Performance Test Suite for bentoml-faster-whisper.

Measures single-request latency, Real-Time Factor (RTF), Speedup Factor, CPU %,
GPU Compute Utilization %, Peak VRAM (MB), and multi-concurrent request
throughput/percentile latencies using long test audio assets.

Saves structured JSON benchmarks to track performance gains across implementation iterations.
"""

import argparse
import datetime
import json
import logging
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

import av
import numpy as np
import psutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

from bentoml_faster_whisper.models.enums import ResponseFormat
from bentoml_faster_whisper.models.transcription_request import TranscriptionRequest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("load_test")

DEFAULT_ASSETS_DIR = PROJECT_ROOT / "tests" / "assets"
RESULTS_DIR = PROJECT_ROOT / "eval_results"
DEFAULT_RESULTS_FILE = RESULTS_DIR / "load_test_results.json"
DEFAULT_LONG_AUDIO = (
    DEFAULT_ASSETS_DIR
    / "Regionaljournal_Basel_Baselland_radio_AUDI20260710_NR_0011_1004cf68be404710b30a996ac4f1ff93.mp3"
)
DEFAULT_RADIO_AUDIO = DEFAULT_LONG_AUDIO


class SystemMonitor:
    """Background sampling thread for CPU %, GPU compute %, and GPU VRAM usage."""

    def __init__(self, sample_interval_s: float = 0.5):
        self.interval = sample_interval_s
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.cpu_samples: list[float] = []
        self.gpu_util_samples: list[float] = []
        self.vram_used_samples: list[float] = []
        self.vram_total_mb: float = 0.0

    def _query_gpu(self) -> tuple[float, float]:
        """Query GPU compute utilization (%) and VRAM used (MB) via nvidia-smi."""
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ]
            out = subprocess.check_output(cmd, text=True, timeout=1.0).strip()
            lines = out.splitlines()
            if lines:
                parts = [p.strip() for p in lines[0].split(",")]
                gpu_util = float(parts[0])
                vram_used = float(parts[1])
                self.vram_total_mb = float(parts[2])
                return gpu_util, vram_used
        except Exception:
            pass
        return 0.0, 0.0

    def _monitor_loop(self):
        psutil.cpu_percent(interval=None)
        while not self._stop_event.is_set():
            cpu = psutil.cpu_percent(interval=None)
            gpu_util, vram_used = self._query_gpu()
            self.cpu_samples.append(cpu)
            if gpu_util > 0 or vram_used > 0:
                self.gpu_util_samples.append(gpu_util)
                self.vram_used_samples.append(vram_used)
            time.sleep(self.interval)

    def start(self):
        self.cpu_samples.clear()
        self.gpu_util_samples.clear()
        self.vram_used_samples.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float]:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        torch_vram_mb = 0.0
        if torch.cuda.is_available():
            try:
                torch_vram_mb = float(torch.cuda.max_memory_allocated() / (1024**2))
            except Exception:
                pass

        avg_cpu = float(np.mean(self.cpu_samples)) if self.cpu_samples else 0.0
        avg_gpu = float(np.mean(self.gpu_util_samples)) if self.gpu_util_samples else 0.0
        peak_vram = max(
            max(self.vram_used_samples) if self.vram_used_samples else 0.0,
            torch_vram_mb,
        )

        return {
            "avg_cpu_pct": round(avg_cpu, 1),
            "avg_gpu_util_pct": round(avg_gpu, 1),
            "peak_vram_mb": round(peak_vram, 1),
            "vram_total_mb": round(self.vram_total_mb, 1),
        }


def get_audio_duration_seconds(audio_path: Path) -> float:
    """Get exact duration in seconds of an audio file using PyAV or fallback."""
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


def reset_peak_vram_stats():
    """Reset CUDA peak memory stats if CUDA is available."""
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


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

    reset_peak_vram_stats()
    monitor = SystemMonitor()
    monitor.start()
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
    sys_stats = monitor.stop()

    return {
        "latency_s": latency_s,
        "avg_cpu_pct": sys_stats["avg_cpu_pct"],
        "avg_gpu_util_pct": sys_stats["avg_gpu_util_pct"],
        "peak_vram_mb": sys_stats["peak_vram_mb"],
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

    monitor = SystemMonitor()
    monitor.start()
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
    sys_stats = monitor.stop()

    return {
        "latency_s": latency_s,
        "avg_cpu_pct": sys_stats["avg_cpu_pct"],
        "avg_gpu_util_pct": sys_stats["avg_gpu_util_pct"],
        "peak_vram_mb": sys_stats["peak_vram_mb"],
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
    """Run a batch of concurrent requests and compute aggregated latency, throughput & system metrics."""
    logger.info(
        "Starting concurrency batch: concurrency=%d, total_requests=%d, diarization=%s",
        concurrency,
        total_requests,
        diarization,
    )

    reset_peak_vram_stats()
    monitor = SystemMonitor()
    monitor.start()
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
    sys_stats = monitor.stop()

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
        "avg_cpu_pct": sys_stats["avg_cpu_pct"],
        "avg_gpu_util_pct": sys_stats["avg_gpu_util_pct"],
        "peak_vram_mb": sys_stats["peak_vram_mb"],
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


def print_performance_gains_comparison(history: list[dict[str, Any]]):
    """Print comparative table highlighting performance gains between earliest baseline and latest run."""
    if len(history) < 2:
        return

    baseline_session = history[0]
    current_session = history[-1]

    baseline_runs = {
        (r.get("mode"), r.get("test_type"), r.get("concurrency", 1)): r for r in baseline_session.get("results", [])
    }
    current_runs = {
        (r.get("mode"), r.get("test_type"), r.get("concurrency", 1)): r for r in current_session.get("results", [])
    }

    print("\n" + "=" * 115)
    print(
        f"PERFORMANCE GAINS COMPARISON (Baseline [{baseline_session.get('git_commit')}] vs Latest [{current_session.get('git_commit')}])"
    )
    print("=" * 115)
    print(
        f"{'Mode':<18} | {'Test Type':<15} | {'Concurrency':<11} | {'Baseline Lat.':<13} | {'Latest Lat.':<13} | {'Latency Δ':<12} | {'Speedup Gain':<13} | {'CPU % Δ':<8}"
    )
    print("-" * 115)

    for key, curr in current_runs.items():
        base = baseline_runs.get(key)
        if not base:
            continue
        mode, test_type, c = key
        label_type = "Single (C=1)" if test_type == "single_request_baseline" else f"Batch (C={c})"
        base_lat = base.get("mean_latency_s", base.get("latency_s", 0.0))
        curr_lat = curr.get("mean_latency_s", curr.get("latency_s", 0.0))

        if base_lat > 0 and curr_lat > 0:
            lat_delta_pct = ((curr_lat - base_lat) / base_lat) * 100.0
            speedup_mult = base_lat / curr_lat
            lat_delta_str = f"{lat_delta_pct:+.1f}%"
            speedup_str = f"{speedup_mult:.2f}x faster"
        else:
            lat_delta_str = "N/A"
            speedup_str = "N/A"

        base_cpu = base.get("avg_cpu_pct", 0.0)
        curr_cpu = curr.get("avg_cpu_pct", 0.0)
        cpu_str = f"{curr_cpu - base_cpu:+.1f}%" if base_cpu > 0 else "N/A"

        print(
            f"{mode:<18} | {label_type:<15} | {c:<11} | {base_lat:<13.2f}s | {curr_lat:<13.2f}s | {lat_delta_str:<12} | {speedup_str:<13} | {cpu_str:<8}"
        )

    print("=" * 115 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Load and Performance Benchmark Test Suite")
    parser.add_argument(
        "--audio",
        type=str,
        default=str(DEFAULT_LONG_AUDIO),
        help="Path to audio file (default: long radio audio)",
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
        default=str(DEFAULT_RESULTS_FILE),
        help=f"Output JSON results file path (default: {DEFAULT_RESULTS_FILE})",
    )

    args = parser.parse_args()

    audio_path = Path(args.audio).resolve()
    if not audio_path.exists():
        logger.error("Audio file not found: %s", audio_path)
        sys.exit(1)

    audio_duration_s = get_audio_duration_seconds(audio_path)
    logger.info("Loaded Audio File: %s (Duration: %.2fs)", audio_path.name, audio_duration_s)

    service_or_url: Any = None
    is_http = bool(args.url)

    if is_http:
        service_or_url = str(args.url)
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

        # 1. Single Request Baseline Test (1 isolated request)
        logger.info("Executing Single Request Baseline Test (%s)...", mode_str)
        if is_http:
            baseline_result = run_single_request_http(cast(str, service_or_url), audio_path, diarization=diarization)
        else:
            baseline_result = run_single_request_in_process(service_or_url, audio_path, diarization=diarization)

        rtf = (baseline_result["latency_s"] / audio_duration_s) if audio_duration_s > 0 else 0.0
        speedup = (audio_duration_s / baseline_result["latency_s"]) if baseline_result["latency_s"] > 0 else 0.0
        logger.info(
            "Baseline Single Request Latency: %.3fs | RTF: %.3fx (lower=faster) | Speedup: %.2fx (higher=faster) | CPU: %.1f%% | GPU: %.1f%% | VRAM: %.1fMB",
            baseline_result["latency_s"],
            rtf,
            speedup,
            baseline_result.get("avg_cpu_pct", 0.0),
            baseline_result.get("avg_gpu_util_pct", 0.0),
            baseline_result.get("peak_vram_mb", 0.0),
        )

        single_run_summary = {
            "mode": mode_str,
            "diarization": diarization,
            "test_type": "single_request_baseline",
            "audio_file": audio_path.name,
            "audio_duration_s": audio_duration_s,
            "concurrency": 1,
            "latency_s": baseline_result["latency_s"],
            "real_time_factor_rtf": rtf,
            "speedup_factor": speedup,
            "avg_cpu_pct": baseline_result.get("avg_cpu_pct", 0.0),
            "avg_gpu_util_pct": baseline_result.get("avg_gpu_util_pct", 0.0),
            "peak_vram_mb": baseline_result.get("peak_vram_mb", 0.0),
            "success": baseline_result["success"],
        }
        benchmark_runs.append(single_run_summary)

        # 2. Multi Concurrent Request Load Tests (Batch of N requests)
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

            mean_lat = batch_metrics["latency_stats"]["mean_s"]
            batch_rtf = (mean_lat / audio_duration_s) if audio_duration_s > 0 else 0.0
            batch_speedup = (audio_duration_s / mean_lat) if mean_lat > 0 else 0.0

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
                "mean_latency_s": mean_lat,
                "p50_latency_s": batch_metrics["latency_stats"]["p50_s"],
                "p90_latency_s": batch_metrics["latency_stats"]["p90_s"],
                "p99_latency_s": batch_metrics["latency_stats"]["p99_s"],
                "real_time_factor_rtf": batch_rtf,
                "speedup_factor": batch_speedup,
                "avg_cpu_pct": batch_metrics.get("avg_cpu_pct", 0.0),
                "avg_gpu_util_pct": batch_metrics.get("avg_gpu_util_pct", 0.0),
                "peak_vram_mb": batch_metrics.get("peak_vram_mb", 0.0),
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
    output_file.parent.mkdir(parents=True, exist_ok=True)
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
    print("\n" + "=" * 125)
    print(f"LOAD TEST SUMMARY RESULT ({audio_path.name}, duration: {audio_duration_s:.1f}s)")
    print("=" * 125)
    print(
        f"{'Mode':<18} | {'Test Type':<15} | {'Concurrency':<11} | {'Mean Latency':<12} | {'P90 Latency':<12} | {'Throughput':<12} | {'RTF (↓)':<8} | {'Speedup (↑)':<10} | {'CPU %':<7} | {'GPU %':<7} | {'VRAM (MB)':<9}"
    )
    print("-" * 125)
    for run in benchmark_runs:
        t_type = "Single Baseline" if run.get("test_type") == "single_request_baseline" else "Batch Load Test"
        c_str = str(run.get("concurrency", 1))
        lat_str = f"{run.get('mean_latency_s', run.get('latency_s', 0)):.2f}s"
        p90_str = f"{run.get('p90_latency_s', 0):.2f}s" if "p90_latency_s" in run else "N/A"
        tp_str = f"{run.get('throughput_req_per_sec', 0):.2f} req/s" if "throughput_req_per_sec" in run else "N/A"
        rtf_str = f"{run.get('real_time_factor_rtf', 0):.3f}x"
        sp_str = f"{run.get('speedup_factor', 0):.2f}x"
        cpu_str = f"{run.get('avg_cpu_pct', 0.0):.1f}%"
        gpu_str = f"{run.get('avg_gpu_util_pct', 0.0):.1f}%"
        vram_str = f"{run.get('peak_vram_mb', 0.0):.0f}"
        print(
            f"{run['mode']:<18} | {t_type:<15} | {c_str:<11} | {lat_str:<12} | {p90_str:<12} | {tp_str:<12} | {rtf_str:<8} | {sp_str:<10} | {cpu_str:<7} | {gpu_str:<7} | {vram_str:<9}"
        )
    print("=" * 125 + "\n")

    # Print comparative performance gains table against baseline run
    print_performance_gains_comparison(history)


if __name__ == "__main__":
    main()
