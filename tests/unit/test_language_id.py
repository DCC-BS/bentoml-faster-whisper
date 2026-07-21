"""Unit tests for the batched turn-level language identification helpers.

Everything runs against fakes/synthetic probability rows — no model weights.
"""

import math
from typing import cast

import numpy as np
import pytest
from faster_whisper import WhisperModel
from pydantic import ValidationError

from bentoml_faster_whisper.config import LanguageIdConfig
from bentoml_faster_whisper.models.enums import Language
from bentoml_faster_whisper.models.transcription_request import TranscriptionRequest
from bentoml_faster_whisper.utils.language_id import (
    detect_turn_language_probs,
    fill_missing_rows_from_intervals,
    resolve_language_inventory,
    viterbi_smooth_languages,
)
from bentoml_faster_whisper.utils.speech_regions import turns_to_language_runs

# ---------------------------------------------------------------------------
# viterbi_smooth_languages


def test_viterbi_flips_isolated_misdetection_island():
    rows = [
        {"de": 0.9, "fr": 0.1},
        {"fr": 0.6, "de": 0.4},  # short, mildly confident misdetection
        {"de": 0.9, "fr": 0.1},
    ]
    durations = [8.0, 2.5, 8.0]
    assert viterbi_smooth_languages(rows, durations, ["de", "fr"]) == ["de", "de", "de"]


def test_viterbi_keeps_sustained_language_switch():
    rows = [
        {"de": 0.9, "fr": 0.1},
        {"de": 0.9, "fr": 0.1},
        {"fr": 0.9, "de": 0.1},
        {"fr": 0.9, "de": 0.1},
        {"fr": 0.9, "de": 0.1},
    ]
    durations = [10.0] * 5
    assert viterbi_smooth_languages(rows, durations, ["de", "fr"]) == ["de", "de", "fr", "fr", "fr"]


def test_viterbi_none_rows_follow_context():
    rows = [{"de": 0.9, "fr": 0.1}, None, {"de": 0.9, "fr": 0.1}]
    assert viterbi_smooth_languages(rows, [5.0, 1.0, 5.0], ["de", "fr"]) == ["de", "de", "de"]


def test_viterbi_leading_none_rows_inherit_first_detection():
    rows = [None, None, {"fr": 0.8, "de": 0.2}]
    assert viterbi_smooth_languages(rows, [1.0, 0.5, 6.0], ["de", "fr"]) == ["fr", "fr", "fr"]


def test_viterbi_single_language_inventory_short_circuits():
    rows = [{"de": 0.2, "fr": 0.8}, None]
    assert viterbi_smooth_languages(rows, [3.0, 1.0], ["de"]) == ["de", "de"]


def test_viterbi_empty_inventory_raises():
    with pytest.raises(ValueError):
        viterbi_smooth_languages([{"de": 1.0}], [1.0], [])


def test_viterbi_empty_rows():
    assert viterbi_smooth_languages([], [], ["de", "fr"]) == []


# ---------------------------------------------------------------------------
# resolve_language_inventory


def test_inventory_drops_noise_language_below_mass_share():
    rows = [
        {"de": 0.80, "fr": 0.15, "ru": 0.05},
        {"ru": 0.90, "de": 0.05, "fr": 0.05},  # 1s noise turn, confidently "russian"
        {"fr": 0.70, "de": 0.25, "ru": 0.05},
    ]
    durations = [10.0, 1.0, 8.0]
    assert resolve_language_inventory(rows, durations) == ["de", "fr"]


def test_inventory_candidates_override_detection():
    rows = [{"ru": 1.0}]
    assert resolve_language_inventory(rows, [10.0], candidates=["de", "fr", "de"]) == ["de", "fr"]


def test_inventory_all_none_rows_is_empty():
    assert resolve_language_inventory([None, None], [1.0, 1.0]) == []


def test_inventory_absolute_floor_keeps_minority_language_in_long_file():
    # ~68s of confident French in a mostly-German meeting: far below a 15% share,
    # but far above the absolute floor — it must stay in the inventory.
    rows = [{"de": 0.99, "fr": 0.01}, {"fr": 0.97, "de": 0.03}, {"ru": 0.9, "de": 0.1}]
    durations = [700.0, 68.0, 1.5]
    assert resolve_language_inventory(rows, durations) == ["de", "fr"]


def test_inventory_keeps_top_language_when_nothing_passes_threshold():
    row = {lang: 0.125 for lang in ("de", "fr", "it", "en", "es", "pt", "nl", "pl")}
    row["de"] = 0.126
    row["fr"] = 0.124
    assert resolve_language_inventory([row], [10.0], mass_share_threshold=0.5) == ["de"]


def test_inventory_noise_language_passes_neither_share_nor_floor():
    rows = [{"de": 1.0}, {"ru": 0.9, "de": 0.1}]
    assert resolve_language_inventory(rows, [100.0, 1.0]) == ["de"]


# ---------------------------------------------------------------------------
# turns_to_language_runs


def test_runs_merge_same_language_turns():
    runs = turns_to_language_runs([(1.0, 5.0), (5.2, 10.0)], ["de", "de"])
    assert runs == [("de", [(0.7, 10.3)])]


def test_runs_clamp_padding_at_language_boundary():
    runs = turns_to_language_runs([(0.0, 5.0), (5.4, 10.0)], ["de", "fr"])
    # midpoint of the 5.0-5.4 gap is 5.2: neither run's padding may cross it
    assert runs == [("de", [(0.0, 5.2)]), ("fr", [(5.2, 10.3)])]


def test_runs_split_crosstalk_overlap_at_midpoint():
    runs = turns_to_language_runs([(0.0, 6.0), (5.0, 10.0)], ["de", "fr"])
    assert runs == [("de", [(0.0, 5.5)]), ("fr", [(5.5, 10.3)])]


def test_runs_interior_gaps_larger_than_merge_gap_stay_split():
    runs = turns_to_language_runs([(0.0, 2.0), (5.0, 7.0)], ["de", "de"])
    assert runs == [("de", [(0.0, 2.3), (4.7, 7.3)])]


# ---------------------------------------------------------------------------
# detect_turn_language_probs (against a fake model: no weights, real windowing)

_FRAMES_PER_SECOND = 100  # hop 160 at 16 kHz


class _FakeFeatureExtractor:
    nb_max_frames = 30 * _FRAMES_PER_SECOND

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        return np.zeros((80, math.ceil(chunk.shape[0] / 160)), dtype=np.float32)


class _FakeCt2Model:
    """Returns one queued language distribution per window, across batches."""

    def __init__(self, results: list[list[tuple[str, float]]]):
        self._queue = list(results)

    def detect_language(self, encoder_output) -> list[list[tuple[str, float]]]:
        batch_size = encoder_output.shape[0]
        out, self._queue = self._queue[:batch_size], self._queue[batch_size:]
        return out


class _FakeWhisper:
    def __init__(self, results: list[list[tuple[str, float]]]):
        self.feature_extractor = _FakeFeatureExtractor()
        self.model = _FakeCt2Model(results)
        self.encode_batch_sizes: list[int] = []

    def encode(self, features: np.ndarray):
        assert features.ndim == 3 and features.shape[2] == self.feature_extractor.nb_max_frames
        self.encode_batch_sizes.append(features.shape[0])
        return features


def _samples(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * 16000), dtype=np.float32)


def test_detect_skips_short_turns_entirely():
    fake = _FakeWhisper(results=[[("<|de|>", 1.0)]])
    rows = detect_turn_language_probs(
        cast(WhisperModel, fake), _samples(20.0), [(0.0, 1.5), (2.0, 8.0)], min_turn_s=2.0
    )
    assert rows[0] is None
    assert rows[1] is not None and rows[1]["de"] == pytest.approx(1.0)
    assert fake.encode_batch_sizes == [1]  # the short turn cost no encoder pass


def test_detect_averages_windows_of_long_turn_weighted_by_duration():
    # 40s turn: 30s window says de, 10s tail says fr -> de wins 3:1
    fake = _FakeWhisper(
        results=[
            [("<|de|>", 1.0), ("<|fr|>", 0.0)],
            [("<|fr|>", 1.0), ("<|de|>", 0.0)],
        ]
    )
    rows = detect_turn_language_probs(cast(WhisperModel, fake), _samples(40.0), [(0.0, 40.0)])
    assert rows[0] is not None
    assert rows[0]["de"] == pytest.approx(0.75, abs=0.01)
    assert rows[0]["fr"] == pytest.approx(0.25, abs=0.01)


def test_detect_batches_windows_across_turns():
    fake = _FakeWhisper(results=[[("<|de|>", 1.0)]] * 5)
    turns = [(float(i * 10), float(i * 10 + 5)) for i in range(5)]
    rows = detect_turn_language_probs(cast(WhisperModel, fake), _samples(50.0), turns, batch_size=2)
    assert all(row is not None for row in rows)
    assert fake.encode_batch_sizes == [2, 2, 1]


def test_detect_drops_sub_minimum_tail_window():
    # 31s turn: 30s window + 1s tail; the tail is below the minimum and skipped
    fake = _FakeWhisper(results=[[("<|de|>", 1.0)]])
    rows = detect_turn_language_probs(cast(WhisperModel, fake), _samples(31.0), [(0.0, 31.0)], min_turn_s=2.0)
    assert rows[0] is not None
    assert rows[0]["de"] == pytest.approx(1.0)
    assert fake.encode_batch_sizes == [1]


def test_fill_missing_rows_detects_merged_short_turn_cluster():
    # Three sub-2s turns in quick succession: individually undetectable, but their
    # merged interval is ~4s and detects fine -> all three inherit its row.
    fake = _FakeWhisper(results=[[("<|fr|>", 0.9), ("<|de|>", 0.1)]])
    turns = [(1.0, 2.0), (2.5, 3.6), (4.0, 5.1)]
    rows = fill_missing_rows_from_intervals(cast(WhisperModel, fake), _samples(10.0), turns, [None, None, None])
    assert all(row is not None and row["fr"] == pytest.approx(0.9) for row in rows)
    assert fake.encode_batch_sizes == [1]  # one merged interval, one encoder pass


def test_fill_missing_rows_leaves_isolated_short_turn_none():
    fake = _FakeWhisper(results=[])
    rows = fill_missing_rows_from_intervals(
        cast(WhisperModel, fake), _samples(10.0), [(5.0, 5.8)], [None], min_turn_s=2.0
    )
    assert rows == [None]
    assert fake.encode_batch_sizes == []


def test_fill_missing_rows_keeps_detected_rows_untouched():
    fake = _FakeWhisper(results=[[("<|fr|>", 1.0)]])
    turns = [(0.0, 6.0), (6.2, 7.0)]
    detected = {"de": 0.8, "fr": 0.2}
    rows = fill_missing_rows_from_intervals(cast(WhisperModel, fake), _samples(10.0), turns, [detected, None])
    assert rows[0] is detected
    assert rows[1] is not None and rows[1]["fr"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TranscriptionRequest.language_candidates


def _request(**overrides) -> TranscriptionRequest:
    return TranscriptionRequest.model_validate({"file": "audio.mp3", **overrides})


def test_language_candidates_default_is_none():
    assert _request().language_candidates is None


@pytest.mark.parametrize("value", [["de", "fr"], "de,fr", b"de, fr"])
def test_language_candidates_accepts_list_and_comma_string(value):
    assert _request(language_candidates=value).language_candidates == [Language.DE, Language.FR]


@pytest.mark.parametrize("value", ["", [], None])
def test_language_candidates_empty_means_none(value):
    assert _request(language_candidates=value).language_candidates is None


def test_language_candidates_bracket_alias():
    request = _request(**{"language_candidates[]": ["de"]})
    assert request.language_candidates == [Language.DE]


def test_language_candidates_invalid_code_raises():
    with pytest.raises(ValidationError):
        _request(language_candidates="de,xx")


# ---------------------------------------------------------------------------
# LanguageIdConfig


def test_language_id_config_env_overrides(monkeypatch):
    monkeypatch.setenv("LID_SWITCH_PENALTY", "3.5")
    monkeypatch.setenv("LID_BATCH_SIZE", "16")
    config = LanguageIdConfig.from_env()
    assert config.switch_penalty == 3.5
    assert config.batch_size == 16
    assert config.min_turn_s == 1.0  # untouched fields keep their defaults


def test_language_id_config_invalid_env_value_raises(monkeypatch):
    monkeypatch.setenv("LID_BATCH_SIZE", "zero")
    with pytest.raises(ValidationError):
        LanguageIdConfig.from_env()


def test_language_id_config_out_of_range_env_value_raises(monkeypatch):
    monkeypatch.setenv("LID_INVENTORY_MASS_SHARE", "1.5")
    with pytest.raises(ValidationError):
        LanguageIdConfig.from_env()
