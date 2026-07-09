"""Fuzzy-match helpers to compare a transcription+diarization result against a
reference transcript (ElevenLabs scribe export: full text plus word entries with
start/end/speaker_id). Shared by the unit and integration quality tests."""

import itertools
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, NamedTuple


class ReferenceWord(NamedTuple):
    start: float
    end: float
    text: str
    speaker: str

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2


class ReferenceTurn(NamedTuple):
    start: float
    end: float
    speaker: str


def load_reference(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def reference_words(reference: dict) -> list[ReferenceWord]:
    return [
        ReferenceWord(word["start"], word["end"], word["text"], word["speaker_id"])
        for word in reference["words"]
        if word["type"] == "word"
    ]


def reference_turns(reference: dict) -> list[ReferenceTurn]:
    """Collapse consecutive same-speaker reference words into speaker turns —
    a stand-in for pyannote output when the real pipeline is not under test."""
    turns: list[ReferenceTurn] = []
    for word in reference_words(reference):
        if turns and turns[-1].speaker == word.speaker:
            turns[-1] = turns[-1]._replace(end=word.end)
        else:
            turns.append(ReferenceTurn(word.start, word.end, word.speaker))
    return turns


def normalize_words(text: str) -> list[str]:
    """Tokenize for fuzzy comparison: case, punctuation, 'ß'/'ss' spelling and
    bracketed annotations like '[lacht]' must not count as differences."""
    text = re.sub(r"\[[^\]]*\]", " ", text.lower()).replace("ß", "ss")
    return re.findall(r"[\wäöüàâéèêëîïôû]+", text)


def text_similarity(a: str, b: str) -> float:
    """Word-level similarity in [0, 1] (1 = identical after normalization)."""
    return SequenceMatcher(None, normalize_words(a), normalize_words(b), autojunk=False).ratio()


class SpeakerMatch(NamedTuple):
    agreement: float
    """Fraction of covered reference words whose predicted speaker matches, under
    the best possible mapping between the two label sets."""
    coverage: float
    """Fraction of reference words whose midpoint falls inside a predicted segment."""


def match_speakers(
    ref_words: Iterable[ReferenceWord],
    predicted_segments: Iterable[tuple[float, float, str]],
) -> SpeakerMatch:
    """Score predicted (start, end, speaker) segments against reference words.
    Speaker labels are arbitrary on both sides, so the score maximizes over all
    injective mappings between the label sets."""
    segments = list(predicted_segments)
    ref_words = list(ref_words)

    pairs: list[tuple[str, str]] = []
    for word in ref_words:
        for start, end, speaker in segments:
            if start <= word.midpoint <= end:
                pairs.append((word.speaker, speaker))
                break

    if not pairs:
        return SpeakerMatch(agreement=0.0, coverage=0.0)

    ref_labels = sorted({ref for ref, _ in pairs})
    pred_labels = sorted({pred for _, pred in pairs})
    if len(pred_labels) >= len(ref_labels):
        mappings = (dict(zip(ref_labels, perm)) for perm in itertools.permutations(pred_labels, len(ref_labels)))
    else:
        mappings = (dict(zip(perm, pred_labels)) for perm in itertools.permutations(ref_labels, len(pred_labels)))

    best = max(sum(mapping.get(ref) == pred for ref, pred in pairs) for mapping in mappings)
    return SpeakerMatch(agreement=best / len(pairs), coverage=len(pairs) / len(ref_words))
