"""Regression: nested pyannote turns must not delete speech before the decode.

brain_dump.mp3 is a ~12 min German brain dump recorded by a single fast speaker
with a French accent. pyannote reports it as near-continuous speech, but it emits
several *nested* turns: a sub-second turn whose span lies entirely inside a much
longer turn (e.g. (679.35, 679.52) inside (673.83, 712.61)). Turns are sorted by
start, so the nested turn becomes the "last" turn of its decode run, and
``turns_to_language_runs`` clamps the run's upper bound to the midpoint between
*that* end and the next run's start — cutting the long turn's remaining speech out
of the collapsed audio entirely. It never reaches Whisper, so its transcript is
missing with no gap in the decode to hint at it.

The windows below were annotated by ear against the response and each one is
covered by pyannote speech turns, so every one of them must show up in the
transcript.

The asset lives in tests/assets/internal/ (gitignored, internal recording), so
this module skips when it is absent. Real Whisper decode and real pyannote, hence
the ``model`` marker; needs HF_TOKEN.

Perf reference (2026-07-17, RTX 4090, after the fix): ~31 s wall clock to
transcribe the 12 min recording, ~45 s for the module including diarization.
Unioning the overlapping turns does not measurably change it — the same speech is
decoded either way, just no longer clamped away.
"""

import json
from pathlib import Path

import pytest

from api_models.enums import ResponseFormat
from api_models.TranscriptionRequest import TranscriptionRequest
from diarization_service import DiarizationService
from helpers.speech_regions import diarization_to_speech_intervals
from tests.fuzzy_match import text_similarity

AUDIO = Path("./tests/assets/internal/brain_dump.mp3")

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(not AUDIO.exists(), reason=f"internal asset {AUDIO} not present"),
]

# (start_s, end_s, reference_text) passages missing from the transcript, transcribed
# by ear. All of them sit inside pyannote speech turns (asserted separately below, so
# a pyannote change that silences a window fails loudly instead of making the coverage
# test vacuously pass).
MISSING_PASSAGES = [
    (285.0, 289.0, "und für die Sache. Das würde sie interessieren."),
    (
        391.0,
        413.0,
        "wie sie selber sagte und von dem wieder eigentlich möchte sie Managerin sein. Sie möchte "
        "mitgestalten, sie möchte mitentscheiden, sie möchte Prozesse optimieren und sie möchte am "
        "liebsten für eine Firma bauen die schon einen Namen hat, eine Firmenkultur hat, die bekannt "
        "ist, die für etwas steht.",
    ),
    (473.0, 477.0, "Eine Strategie ist"),
    (558.0, 561.0, "da es nur 5 Monate dauert"),
    (
        613.0,
        632.0,
        "die ist offen diese Stelle und es ist eigentlich nur intern ausgeschrieben aber durch einen "
        "Kollegen hat sie Wind davon gekriegt und steht im Kontakt mit der Person die die Stelle "
        "letztendlich vergeben wird und da soll sie unbedingt nachhacken und sich melden.",
    ),
    (
        658.0,
        677.0,
        "habe ich etwas vergessen? Ja, eine grosse Erkenntniss ist auch dass sie sich nicht mehr "
        "beschränken sollte auf Menschen, die sie kennt, sondern wirklich den Mut haben sollte,",
    ),
    (
        696.0,
        712.0,
        "Sie ist keine Bittstellerin, sie ist eine Anbieterin. Und natürlich in dieser "
        "Kriesensituation, in der sie sich befindet, ist das nicht immer ersichtlich. Aber die ganze "
        "Beratung hat das Ziel auch mit den verschiedenen Arbeitsmitteln",
    ),
]

# Share of a window's pyannote speech that must be covered by emitted segments. Not
# 1.0: segment edges are word-aligned, so a little slack at the borders is normal.
MIN_SPEECH_COVERAGE = 0.6

# Word-level similarity between the reference and the words Whisper timestamped into
# the window. Well below 1.0: the reference is an ear transcript, so spelling variants
# ("nachhacken"/"nachhaken"), filler words and edge words drifting a few tenths across
# the window border must not fail the test — this asserts the passage is *there*, not
# that it is transcribed perfectly.
MIN_TEXT_SIMILARITY = 0.6

# Words that must be timestamped near where they are actually spoken. This is the 658 s
# passage, whose failure mode differs from the others: "eine grosse Erkenntnis ist" *is*
# emitted, but ~5 s early, because the truncated run drops "hab ich etwas vergessen"
# ahead of it and the decode closes the hole by pulling the rest forward. Windows are
# the spoken times read off the fixed decode; on the truncated run "vergessen" appeared
# only at 694.5 s (a different sentence) and "Erkenntnis" at 661.2 s.
TIMESTAMP_CANARIES = [
    ("vergessen", 659.0, 664.0),
    ("Erkenntnis", 664.0, 670.0),
]


@pytest.fixture(scope="module")
def transcription(handler) -> dict:
    request = TranscriptionRequest.model_validate(
        {
            "file": AUDIO,
            # Language is pinned to keep the test on the collapsed-decode path without
            # the LID/Viterbi machinery on top. The speaker count is deliberately *not*
            # pinned: letting pyannote estimate it is what produces the nested turns
            # that trigger the bug, and it is how the service is called in practice.
            "language": "de",
            "diarization": True,
            "response_format": ResponseFormat.VERBOSE_JSON,
            "timestamp_granularities": ["word"],
        }
    )
    return json.loads(handler.transcribe_audio(request))


@pytest.fixture(scope="module")
def speech_intervals() -> list[tuple[float, float]]:
    turns = DiarizationService().diarize(str(AUDIO))
    return diarization_to_speech_intervals([turn.segment for turn in turns])


def _covered_seconds(spans: list[tuple[float, float]], start_s: float, end_s: float) -> float:
    return sum(max(0.0, min(end, end_s) - max(start, start_s)) for start, end in spans)


def _words_in(transcription: dict, start_s: float, end_s: float) -> list[dict]:
    """Words whose midpoint falls inside the window, so a word straddling the border
    counts for exactly one side."""
    return [w for w in transcription["words"] if start_s <= (w["start"] + w["end"]) / 2 <= end_s]


@pytest.mark.parametrize(("start_s", "end_s", "reference"), MISSING_PASSAGES)
def test_missing_passage_is_transcribed(transcription, start_s, end_s, reference):
    """The passage's words must be transcribed *and* land in the window they are
    spoken in — the strongest statement of what the bug destroys."""
    predicted = "".join(w["word"] for w in _words_in(transcription, start_s, end_s))
    similarity = text_similarity(reference, predicted)

    assert similarity >= MIN_TEXT_SIMILARITY, (
        f"window {start_s}-{end_s}s: transcript similarity {similarity:.2f} < {MIN_TEXT_SIMILARITY}\n"
        f"  expected: {reference}\n"
        f"  got     : {predicted.strip() or '<nothing>'}"
    )


@pytest.mark.parametrize(("word", "start_s", "end_s"), TIMESTAMP_CANARIES)
def test_canary_word_is_timestamped_where_it_is_spoken(transcription, word, start_s, end_s):
    spoken_at = [round(w["start"], 1) for w in transcription["words"] if word.lower() in w["word"].strip().lower()]
    assert any(start_s <= at <= end_s for at in spoken_at), (
        f"{word!r} is spoken at {start_s}-{end_s}s but timestamped at {spoken_at or '<never emitted>'}"
    )


@pytest.mark.parametrize(("start_s", "end_s", "reference"), MISSING_PASSAGES)
def test_window_is_speech_according_to_pyannote(speech_intervals, start_s, end_s, reference):
    speech_s = _covered_seconds(speech_intervals, start_s, end_s)
    assert speech_s >= 0.5 * (end_s - start_s), (
        f"window {start_s}-{end_s}s is no longer mostly speech per pyannote ({speech_s:.1f}s) — "
        "the annotation or the diarization changed; re-check before trusting the coverage test"
    )


@pytest.mark.parametrize(("start_s", "end_s", "reference"), MISSING_PASSAGES)
def test_speech_window_is_transcribed(transcription, speech_intervals, start_s, end_s, reference):
    segments = [(s["start"], s["end"]) for s in transcription["segments"]]
    speech_s = _covered_seconds(speech_intervals, start_s, end_s)
    transcribed_s = _covered_seconds(segments, start_s, end_s)

    assert transcribed_s >= MIN_SPEECH_COVERAGE * speech_s, (
        f"window {start_s}-{end_s}s: pyannote reports {speech_s:.1f}s of speech but only "
        f"{transcribed_s:.1f}s is covered by segments — speech was dropped before or during the decode. "
        "Nearby segments: "
        + str(
            [
                (round(s["start"], 1), round(s["end"], 1), s["text"])
                for s in transcription["segments"]
                if s["end"] > start_s - 5 and s["start"] < end_s + 5
            ]
        )
    )


def test_no_speech_interval_is_missing_from_the_decode_runs(speech_intervals, transcription):
    """Global version of the above: no pyannote speech interval longer than 3 s may
    come back with almost nothing transcribed in it."""
    segments = [(s["start"], s["end"]) for s in transcription["segments"]]
    starved = [
        (round(start, 1), round(end, 1))
        for start, end in speech_intervals
        if end - start > 3.0 and _covered_seconds(segments, start, end) < 0.5 * (end - start)
    ]
    assert not starved, f"speech intervals with under half their speech transcribed: {starved}"
