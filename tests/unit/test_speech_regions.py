from dataclasses import dataclass

from helpers.speech_regions import diarization_to_clip_timestamps


@dataclass
class _Turn:
    start: float
    end: float


def test_empty_input_returns_empty_list():
    assert diarization_to_clip_timestamps([]) == []


def test_single_turn_is_padded_and_clamped_at_zero():
    clips = diarization_to_clip_timestamps([_Turn(0.1, 5.0)], pad_s=0.3, merge_gap_s=1.0)

    assert clips == [0.0, 5.3]


def test_overlapping_turns_are_merged():
    clips = diarization_to_clip_timestamps(
        [_Turn(1.0, 4.0), _Turn(3.0, 6.0)],
        pad_s=0.0,
        merge_gap_s=0.0,
    )

    assert clips == [1.0, 6.0]


def test_turns_within_merge_gap_are_merged():
    clips = diarization_to_clip_timestamps(
        [_Turn(1.0, 2.0), _Turn(2.8, 4.0)],
        pad_s=0.0,
        merge_gap_s=1.0,
    )

    assert clips == [1.0, 4.0]


def test_turns_beyond_merge_gap_stay_separate():
    clips = diarization_to_clip_timestamps(
        [_Turn(1.0, 2.0), _Turn(10.0, 12.0)],
        pad_s=0.0,
        merge_gap_s=1.0,
    )

    assert clips == [1.0, 2.0, 10.0, 12.0]


def test_unsorted_input_is_sorted():
    clips = diarization_to_clip_timestamps(
        [_Turn(10.0, 12.0), _Turn(1.0, 2.0)],
        pad_s=0.0,
        merge_gap_s=1.0,
    )

    assert clips == [1.0, 2.0, 10.0, 12.0]


def test_zero_or_negative_length_turns_are_dropped():
    clips = diarization_to_clip_timestamps(
        [_Turn(1.0, 1.0), _Turn(5.0, 4.0)],
        pad_s=0.3,
        merge_gap_s=1.0,
    )

    assert clips == []
