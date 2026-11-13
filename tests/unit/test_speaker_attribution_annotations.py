"""Speaker attribution scored against human listening annotations.

Each ``*_speaker_annotations*.json`` in tests/assets/internal lists time spans with the
speaker a listener heard there (exported by tools/speaker_review, or written by hand
with an ``anchor`` span). Spans labelled OTHER/OVERLAP/UNSURE are not scored.
Annotation labels are mapped onto output labels by maximum overlap, so any number of
speakers works. The recording is cropped to the last annotated span and transcribed
with the cached full-file pyannote turns (``<audio>_turns.json``) replayed, so only
decode → merge → split is under test. Failure messages also report what pyannote's
turns alone say for each wrong span, which separates diarization errors from
attribution errors. Internal recordings, skipped when absent.
"""

import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from pyannote.core import Segment as PyannoteSegment
from scipy.optimize import linear_sum_assignment

from bentoml_faster_whisper.models.enums import ResponseFormat
from bentoml_faster_whisper.models.transcription_request import TranscriptionRequest
from bentoml_faster_whisper.services.diarization_service import DiarizationSegment

pytestmark = pytest.mark.model

INTERNAL = Path(__file__).resolve().parent.parent / "assets" / "internal"
ANNOTATION_FILES = sorted(INTERNAL.glob("*_speaker_annotations*.json"))
UNSCORED = {"OTHER", "OVERLAP", "UNSURE"}
CROP_MARGIN_S = 15.0
MIN_AGREEMENT = 0.9
WORST_SPANS_SHOWN = 15


def _load(path: Path) -> dict:
    annotation = json.loads(path.read_text(encoding="utf-8"))
    spans = [*([annotation["anchor"]] if "anchor" in annotation else []), *annotation["spans"]]
    annotation["scored"] = [span for span in spans if span["speaker"] not in UNSCORED and span["end"] > span["start"]]
    return annotation


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _speaker_durations(items, start: float, end: float) -> dict[str, float]:
    durations: dict[str, float] = defaultdict(float)
    for item_start, item_end, speaker in items:
        if speaker is not None:
            durations[speaker] += _overlap(item_start, item_end, start, end)
    return {speaker: d for speaker, d in durations.items() if d > 0}


def _words_in(words: list, start: float, end: float) -> list:
    return [w for w in words if start <= (w[0] + w[1]) / 2 < end]


def _label_mapping(spans: list[dict], words: list) -> dict[str, str]:
    """Output label -> annotation label, one-to-one by maximum agreeing duration."""
    annotated = sorted({span["speaker"] for span in spans})
    predicted = sorted({w[2] for w in words if w[2] is not None})
    matrix = np.zeros((len(predicted), len(annotated)))
    for span in spans:
        for w_start, w_end, speaker, *_ in _words_in(words, span["start"], span["end"]):
            if speaker is None:
                continue
            matrix[predicted.index(speaker), annotated.index(span["speaker"])] += w_end - w_start
    rows, cols = linear_sum_assignment(-matrix)
    return {predicted[row]: annotated[col] for row, col in zip(rows, cols)}


@pytest.fixture(scope="module")
def transcribed(handler, tmp_path_factory):
    cache: dict[str, dict] = {}

    def run(audio_name: str) -> dict:
        if audio_name in cache:
            return cache[audio_name]
        audio = INTERNAL / audio_name
        turns_path = INTERNAL / f"{audio_name.replace('.', '_')}_turns.json"
        if not audio.exists() or not turns_path.exists():
            pytest.skip(f"internal asset {audio.name} or its turns fixture not present")

        crop_s = CROP_MARGIN_S + max(
            span["end"]
            for path in ANNOTATION_FILES
            for span in _load(path)["scored"]
            if _load(path)["audio"] == audio_name
        )
        cropped = tmp_path_factory.mktemp("crop") / "cropped.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(audio), "-t", str(crop_s), "-ar", "16000", "-ac", "1", str(cropped)],
            check=True,
            capture_output=True,
        )
        turns = [
            (start, min(end, crop_s), speaker)
            for start, end, speaker in json.loads(turns_path.read_text())
            if start < crop_s
        ]
        replay = [DiarizationSegment(PyannoteSegment(start, end), speaker) for start, end, speaker in turns]
        request = TranscriptionRequest.model_validate(
            {
                "file": cropped,
                "diarization": True,
                "response_format": ResponseFormat.VERBOSE_JSON,
                "timestamp_granularities": ["segment", "word"],
            }
        )
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(handler.diarization, "diarize", lambda *args, **kwargs: iter(replay))
            segments = json.loads(handler.transcribe_audio(request))["segments"]

        words = [
            (word["start"], word["end"], segment["speaker"], word.get("speaker"), word["word"].strip())
            for segment in segments
            for word in segment["words"] or []
        ]
        cache[audio_name] = {"words": words, "turns": turns}
        return cache[audio_name]

    return run


@pytest.mark.parametrize("annotation_path", ANNOTATION_FILES, ids=[path.stem for path in ANNOTATION_FILES])
def test_spans_attributed_to_annotated_speaker(transcribed, annotation_path):
    annotation = _load(annotation_path)
    result = transcribed(annotation["audio"])
    words, turns = result["words"], result["turns"]
    mapping = _label_mapping(annotation["scored"], words)

    total, correct = 0.0, 0.0
    by_origin: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    wrong = []
    for span in annotation["scored"]:
        in_span = _words_in(words, span["start"], span["end"])
        span_total = sum(w[1] - w[0] for w in in_span)
        span_correct = sum(w[1] - w[0] for w in in_span if mapping.get(w[2]) == span["speaker"])
        total += span_total
        correct += span_correct
        origin = by_origin[span.get("origin", "manual")]
        origin[0] += span_correct
        origin[1] += span_total
        if span_total and span_correct / span_total < 0.5:
            pyannote = {
                mapping.get(s, s): round(d, 2) for s, d in _speaker_durations(turns, span["start"], span["end"]).items()
            }
            wrong.append(
                (
                    span_total - span_correct,
                    (
                        f"{span['start']:.1f}-{span['end']:.1f}s expected {span['speaker']}: "
                        f"{[(w[4], mapping.get(w[2], w[2])) for w in in_span]}; pyannote {pyannote}"
                    ),
                )
            )

    assert total > 0, f"{annotation_path.name}: no transcribed word inside any annotated span"
    agreement = correct / total
    breakdown = {origin: f"{c / t:.1%} of {t:.0f}s" for origin, (c, t) in by_origin.items() if t}
    worst = "\n".join(line for _, line in sorted(wrong, reverse=True)[:WORST_SPANS_SHOWN])
    assert agreement >= MIN_AGREEMENT, (
        f"{annotation_path.name}: {agreement:.1%} of annotated speech attributed correctly "
        f"({len(wrong)} wrong spans; by origin {breakdown}); worst:\n{worst}"
    )
