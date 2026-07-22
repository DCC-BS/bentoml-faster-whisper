#!/usr/bin/env python3
"""Quality Evaluation Harness for bentoml-faster-whisper.

Reuses the ASR evaluation suite from /home/yanick/code/research/whisper-evaluation
to calculate WER (Word Error Rate), CER (Character Error Rate), and BLEU scores
against the curated test suite dataset.

All results and predictions are written to 'eval_results/' (ignored by git).
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval_quality")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_REPO_DIR = Path("/home/yanick/code/research/whisper-evaluation")
DEFAULT_DATA_DIR = EVAL_REPO_DIR / "data"
OUTPUT_DIR = PROJECT_ROOT / "eval_results"


def main():
    parser = argparse.ArgumentParser(description="ASR Quality Evaluation Suite Harness")
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:50001/v1",
        help="Base API URL of the running ASR service (default: http://localhost:50001/v1)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help=f"Path to curated test data directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="large-v2",
        help="Model ID/name passed in transcription API requests (default: large-v2)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="de",
        help="Language code for transcription evaluation (default: de)",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        logger.error("Test data directory not found: %s", data_dir)
        sys.exit(1)

    if not EVAL_REPO_DIR.exists():
        logger.error("Research evaluation suite repository not found at %s", EVAL_REPO_DIR)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("==========================================================")
    logger.info("Starting ASR Quality Evaluation Suite")
    logger.info("API URL: %s", args.api_url)
    logger.info("Data Dir: %s", data_dir)
    logger.info("Model: %s | Language: %s", args.model, args.language)
    logger.info("Output Directory: %s (Git-ignored)", OUTPUT_DIR)
    logger.info("==========================================================")

    # Build command to invoke asr-eval via uv in the evaluation suite project
    cmd = [
        "uv",
        "run",
        "--project",
        str(EVAL_REPO_DIR),
        "asr-eval",
        "openai_api",
        "--model",
        args.model,
        "--api-url",
        args.api_url,
        "--data-dir",
        str(data_dir),
        "--arg",
        f"language={args.language}",
    ]

    logger.info("Executing command: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, cwd=str(EVAL_REPO_DIR), check=True, text=True)
        logger.info("Quality evaluation completed successfully with exit code %d", proc.returncode)
    except subprocess.CalledProcessError as e:
        logger.error("Quality evaluation failed with exit code %d", e.returncode)
        sys.exit(e.returncode)
    except Exception as e:
        logger.error("Failed to execute quality evaluation: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
