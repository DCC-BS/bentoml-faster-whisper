"""Multi-language regression on a real Teams meeting recording (Swiss de/fr).

teams_konferenz.mp3 is a hard case for the per-region language detection path:
long leading silence, sub-second noise turns, speakers switching language
within seconds, and long single-speaker runs in one language. The expected
regions below were annotated by ear.

The asset lives in tests/assets/internal/ (gitignored, internal recording),
so this module skips entirely when it is absent. Regenerate it with:

    ffmpeg -i tests/assets/internal/teams_konferenz.mp4 -t 900 -vn \
        -ar 16000 -ac 1 tests/assets/internal/teams_konferenz.mp3

The transcription fixture also records wall-clock and language-detection
metrics and prints them (run with -s), so the same run doubles as the
performance reference for changes to the multi-language path.

Baseline on the pre-turn-level implementation (2026-07-09, RTX 4090,
per-merged-interval sequential detect_language): total=35.9s,
detect_language calls=17, detect_language time=1.2s. Failing then:
727-763 (de instead of fr), 862-900 (fr instead of de), exotic
languages {ru, en} detected on noise turns.

After the turn-level batched LID + Viterbi rework (same day, same GPU):
all regions correct, total=37.4-42.3s across runs, detect_language
calls=0, batched LID encodes=82 windows in 11 batches, ~5.6s.
"""

import dataclasses
import json
import time
from pathlib import Path

import pytest
from faster_whisper import WhisperModel

from api_models.enums import ResponseFormat
from api_models.TranscriptionRequest import TranscriptionRequest

AUDIO = Path("./tests/assets/internal/teams_konferenz.mp3")

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(not AUDIO.exists(), reason=f"internal asset {AUDIO} not present"),
]

# Annotated (start_s, end_s, language) regions. Edges are trimmed by
# EDGE_TOLERANCE_S before matching so segments straddling a boundary
# don't flip the verdict.
EXPECTED_REGIONS = [
    (95.0, 105.0, "fr"),  # two speakers, fast alternation, French throughout
    (727.0, 763.0, "fr"),  # single speaker, long French run (was misdetected as de)
    (765.0, 820.0, "de"),  # next speaker, German
    (825.0, 861.0, "fr"),  # first speaker back to French
    (862.0, 900.0, "de"),  # fast two-speaker exchange, German (was decoded as fr)
]
EDGE_TOLERANCE_S = 2.0

# The recording contains only German and French speech; anything else is a
# misdetection on noise/mumbling (e.g. the 1s background-noise turn around
# 90s used to come back as Russian).
EXPECTED_LANGUAGES = {"de", "fr"}


@dataclasses.dataclass
class _LidMetrics:
    detect_calls: int = 0
    detect_seconds: float = 0.0
    lid_encode_calls: int = 0
    lid_encode_windows: int = 0
    lid_encode_seconds: float = 0.0
    total_seconds: float = 0.0


@pytest.fixture(scope="module")
def transcription(handler) -> tuple[dict, _LidMetrics]:
    metrics = _LidMetrics()
    original_detect = WhisperModel.detect_language
    original_encode = WhisperModel.encode

    def timed_detect(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return original_detect(self, *args, **kwargs)
        finally:
            metrics.detect_calls += 1
            metrics.detect_seconds += time.perf_counter() - started

    def timed_encode(self, features):
        # Batched (ndim == 3) encodes only happen in the LID path; the decode
        # loop encodes one 2-D mel window at a time.
        if getattr(features, "ndim", None) != 3:
            return original_encode(self, features)
        started = time.perf_counter()
        try:
            return original_encode(self, features)
        finally:
            metrics.lid_encode_calls += 1
            metrics.lid_encode_windows += features.shape[0]
            metrics.lid_encode_seconds += time.perf_counter() - started

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(WhisperModel, "detect_language", timed_detect)
    monkeypatch.setattr(WhisperModel, "encode", timed_encode)
    try:
        request = TranscriptionRequest.model_validate(
            {
                "file": AUDIO,
                "diarization": True,
                "response_format": ResponseFormat.VERBOSE_JSON,
                "timestamp_granularities": ["word"],
            }
        )
        started = time.perf_counter()
        response = json.loads(handler.transcribe_audio(request))
        metrics.total_seconds = time.perf_counter() - started
    finally:
        monkeypatch.undo()

    print(
        f"\n[teams_konferenz metrics] total={metrics.total_seconds:.1f}s "
        f"detect_language calls={metrics.detect_calls} "
        f"detect_language time={metrics.detect_seconds:.1f}s "
        f"lid_encode batches={metrics.lid_encode_calls} "
        f"windows={metrics.lid_encode_windows} "
        f"time={metrics.lid_encode_seconds:.1f}s"
    )
    return response, metrics


def _majority_language(segments: list[dict], start_s: float, end_s: float) -> str | None:
    """Language covering the most segment time inside the (trimmed) region."""
    start_s += EDGE_TOLERANCE_S
    end_s -= EDGE_TOLERANCE_S
    weights: dict[str | None, float] = {}
    for segment in segments:
        overlap = min(segment["end"], end_s) - max(segment["start"], start_s)
        if overlap > 0:
            weights[segment["language"]] = weights.get(segment["language"], 0.0) + overlap
    return max(weights, key=lambda lang: weights[lang]) if weights else None


@pytest.mark.parametrize(("start_s", "end_s", "language"), EXPECTED_REGIONS)
def test_region_majority_language(transcription, start_s, end_s, language):
    response, _ = transcription
    majority = _majority_language(response["segments"], start_s, end_s)
    assert majority == language, f"region {start_s}-{end_s}s: expected {language}, got {majority}; segments: " + str(
        [
            (round(s["start"], 1), round(s["end"], 1), s["language"], s["text"])
            for s in response["segments"]
            if s["end"] > start_s and s["start"] < end_s
        ]
    )


def test_no_exotic_language_detected(transcription):
    response, _ = transcription
    detected = {segment["language"] for segment in response["segments"]}
    assert detected <= EXPECTED_LANGUAGES, f"unexpected languages: {detected - EXPECTED_LANGUAGES}"


def test_top_level_language_detected(transcription):
    response, _ = transcription
    assert response["language"] in EXPECTED_LANGUAGES


def test_fast_exchange_keeps_speaker_alternation(transcription):
    """95-105s is a rapid two-speaker French exchange of sub-2s turns. Decoding
    them as one collapsed run must not flatten them into a single segment with
    one speaker — segments are split at word-level speaker changes."""
    response, _ = transcription
    segments = [s for s in response["segments"] if s["end"] > 95.0 and s["start"] < 105.0]

    assert len(segments) >= 3, f"expected the fast exchange to stay split into turns: {segments}"
    assert len({s["speaker"] for s in segments}) >= 2, "expected both speakers to survive the fast exchange: " + str(
        [(round(s["start"], 1), round(s["end"], 1), s["speaker"], s["text"]) for s in segments]
    )
