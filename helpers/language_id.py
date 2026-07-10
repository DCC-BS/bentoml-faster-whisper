"""Batched Whisper language identification over diarization turns.

Replaces the one-encoder-pass-per-interval detection loop: the mel windows of
all turns are encoded in batches, long turns average the distributions of all
their 30s windows instead of trusting only the first one, and the per-turn
decisions are resolved jointly over the whole file (language inventory +
Viterbi smoothing) instead of taking each top-1 in isolation.

All tunables live in ``config.LanguageIdConfig`` and can be overridden per
deployment through ``LID_*`` environment variables; the function parameters
below exist for tests and default to the config values.
"""

import math
from typing import Sequence

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.audio import pad_or_trim

from config import language_id_config
from helpers.speech_regions import WHISPER_SAMPLE_RATE, pad_and_merge_intervals

_MIN_PROB = 1e-6


def detect_turn_language_probs(
    whisper: WhisperModel,
    decoded: np.ndarray,
    turns: Sequence[tuple[float, float]],
    batch_size: int | None = None,
    min_turn_s: float | None = None,
) -> list[dict[str, float] | None]:
    """Detect one full language probability distribution per turn, batched.

    Turns shorter than ``min_turn_s`` return ``None`` without costing an encoder
    pass — Whisper's language ID is unreliable on very short clips (a 1s noise
    turn can come back as any language with high confidence); the Viterbi
    smoothing resolves them from context instead. Turns longer than one 30s
    window get the duration-weighted average over all their windows, so one
    ambiguous stretch can't pin the whole turn alone.
    """
    if batch_size is None:
        batch_size = language_id_config.batch_size
    if min_turn_s is None:
        min_turn_s = language_id_config.min_turn_s
    extractor = whisper.feature_extractor
    frames_per_second = extractor.nb_max_frames / 30.0
    min_frames = int(min_turn_s * frames_per_second)

    # (turn index, evidence weight in seconds, one padded 30s mel window)
    windows: list[tuple[int, float, np.ndarray]] = []
    for idx, (start_s, end_s) in enumerate(turns):
        chunk = decoded[int(start_s * WHISPER_SAMPLE_RATE) : int(end_s * WHISPER_SAMPLE_RATE)]
        if chunk.shape[0] < int(min_turn_s * WHISPER_SAMPLE_RATE):
            continue
        features = extractor(chunk)
        for offset in range(0, features.shape[-1], extractor.nb_max_frames):
            window = features[..., offset : offset + extractor.nb_max_frames]
            if offset > 0 and window.shape[-1] < min_frames:
                break  # tail too short to detect on; the earlier windows carry the turn
            windows.append((idx, window.shape[-1] / frames_per_second, pad_or_trim(window)))

    mass: dict[int, dict[str, float]] = {}
    total_weight: dict[int, float] = {}
    for batch_start in range(0, len(windows), batch_size):
        batch = windows[batch_start : batch_start + batch_size]
        encoder_output = whisper.encode(np.stack([window for _, _, window in batch]))
        for (idx, weight, _), results in zip(batch, whisper.model.detect_language(encoder_output)):
            turn_mass = mass.setdefault(idx, {})
            for token, prob in results:
                language = token[2:-2]  # "<|de|>" -> "de"
                turn_mass[language] = turn_mass.get(language, 0.0) + weight * prob
            total_weight[idx] = total_weight.get(idx, 0.0) + weight

    rows: list[dict[str, float] | None] = [None] * len(turns)
    for idx, turn_mass in mass.items():
        rows[idx] = {language: value / total_weight[idx] for language, value in turn_mass.items()}
    return rows


def fill_missing_rows_from_intervals(
    whisper: WhisperModel,
    decoded: np.ndarray,
    turns: Sequence[tuple[float, float]],
    rows: Sequence[dict[str, float] | None],
    batch_size: int | None = None,
    min_turn_s: float | None = None,
) -> list[dict[str, float] | None]:
    """Second detection pass for turns too short to detect on alone: pad/merge
    all turns into speech intervals — several short turns in quick succession
    (fast two-speaker exchanges) form one detectable chunk — and give every
    undetected turn its covering interval's distribution. Turns whose interval
    is still too short stay ``None`` and are resolved by the Viterbi context.
    """
    filled = list(rows)
    missing = [idx for idx, row in enumerate(filled) if row is None]
    if not missing:
        return filled

    intervals = pad_and_merge_intervals(turns)

    def covering_interval(turn: tuple[float, float]) -> int | None:
        midpoint = (turn[0] + turn[1]) / 2
        for i, (start_s, end_s) in enumerate(intervals):
            if start_s <= midpoint <= end_s:
                return i
        return None

    turn_to_interval = {idx: covering_interval(turns[idx]) for idx in missing}
    needed = sorted({i for i in turn_to_interval.values() if i is not None})
    interval_rows = detect_turn_language_probs(whisper, decoded, [intervals[i] for i in needed], batch_size, min_turn_s)
    row_by_interval = dict(zip(needed, interval_rows))

    for idx in missing:
        interval_idx = turn_to_interval[idx]
        if interval_idx is not None:
            filled[idx] = row_by_interval[interval_idx]
    return filled


def resolve_language_inventory(
    prob_rows: Sequence[dict[str, float] | None],
    durations: Sequence[float],
    candidates: Sequence[str] | None = None,
    mass_share_threshold: float | None = None,
    min_mass_s: float | None = None,
) -> list[str]:
    """The set of languages the file plausibly contains, most speech first.

    Explicit ``candidates`` win outright. Otherwise the duration-weighted
    probability mass over all detected turns is aggregated and a language
    survives with either ``mass_share_threshold`` of the total or ``min_mass_s``
    seconds outright — the share catches every real language in short files, the
    absolute floor keeps a minority language alive in long ones (a minute of
    French in a 45-minute German meeting is far below any workable share), and a
    one-off "russian" on a noise turn passes neither.
    """
    if candidates:
        return list(dict.fromkeys(candidates))
    if mass_share_threshold is None:
        mass_share_threshold = language_id_config.inventory_mass_share
    if min_mass_s is None:
        min_mass_s = language_id_config.min_language_mass_s

    mass: dict[str, float] = {}
    total = 0.0
    for row, duration in zip(prob_rows, durations, strict=True):
        if row is None:
            continue
        for language, prob in row.items():
            mass[language] = mass.get(language, 0.0) + duration * prob
        total += duration
    if not mass:
        return []

    threshold = min(mass_share_threshold * total, min_mass_s)
    inventory = [language for language, value in mass.items() if value >= threshold]
    if not inventory:
        inventory = [max(mass, key=lambda language: mass[language])]
    return sorted(inventory, key=lambda language: -mass[language])


def viterbi_smooth_languages(
    prob_rows: Sequence[dict[str, float] | None],
    durations: Sequence[float],
    inventory: Sequence[str],
    switch_penalty: float | None = None,
    evidence_cap_s: float | None = None,
) -> list[str]:
    """Assign every turn a language from ``inventory`` by Viterbi decoding.

    Emissions are the turn's detected probabilities renormalized over the
    inventory, weighted by the turn's duration (capped at ``evidence_cap_s`` so
    a single confidently-misdetected long turn can still be outvoted by its
    context); undetected turns (``None`` rows) emit uniformly and follow their
    context. Each language switch between adjacent turns costs
    ``switch_penalty``, so an isolated misdetection inside a long same-language
    stretch gets flipped while a genuine sustained switch survives.
    """
    if not inventory:
        raise ValueError("inventory must contain at least one language")
    if not prob_rows:
        return []
    if len(inventory) == 1:
        return [inventory[0]] * len(prob_rows)
    if switch_penalty is None:
        switch_penalty = language_id_config.switch_penalty
    if evidence_cap_s is None:
        evidence_cap_s = language_id_config.evidence_cap_s

    emissions: list[list[float]] = []
    for row, duration in zip(prob_rows, durations, strict=True):
        if row is None:
            emissions.append([0.0] * len(inventory))
            continue
        probs = [max(row.get(language, 0.0), _MIN_PROB) for language in inventory]
        norm = sum(probs)
        weight = min(duration, evidence_cap_s)
        emissions.append([weight * math.log(prob / norm) for prob in probs])

    scores = emissions[0]
    backpointers: list[list[int]] = []
    for emission in emissions[1:]:
        best_idx = max(range(len(inventory)), key=lambda i: scores[i])
        step_pointers = []
        step_scores = []
        for j, e in enumerate(emission):
            stay = scores[j]
            switch = scores[best_idx] - switch_penalty
            if best_idx != j and switch > stay:
                step_pointers.append(best_idx)
                step_scores.append(switch + e)
            else:
                step_pointers.append(j)
                step_scores.append(stay + e)
        backpointers.append(step_pointers)
        scores = step_scores

    idx = max(range(len(inventory)), key=lambda i: scores[i])
    path = [idx]
    for step_pointers in reversed(backpointers):
        idx = step_pointers[path[-1]]
        path.append(idx)
    return [inventory[i] for i in reversed(path)]
