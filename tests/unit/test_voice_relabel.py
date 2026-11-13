from types import SimpleNamespace

import numpy as np

from bentoml_faster_whisper.utils.core import Segment, Word
from bentoml_faster_whisper.utils.voice_relabel import relabel_segments, speaker_profiles

VOICE_A = np.array([1.0, 0.0])
VOICE_B = np.array([0.0, 1.0])


def _turn(start: float, end: float, speaker: str):
    return SimpleNamespace(start=start, end=end, speaker=speaker)


def _voice_track(*spans: tuple[float, float, np.ndarray]):
    """Fake embedder: the voice actually speaking at the midpoint of the requested span."""

    def embed(start: float, end: float) -> np.ndarray:
        mid = (start + end) / 2
        return next(voice for a, b, voice in spans if a <= mid < b)

    return embed


def _segment(start: float, end: float, speaker: str | None) -> Segment:
    words = [
        Word(start=start, end=(start + end) / 2, word=" eins", probability=1.0, speaker=speaker),
        Word(start=(start + end) / 2, end=end, word=" zwei", probability=1.0, speaker=speaker),
    ]
    return Segment(
        id=0,
        seek=0,
        start=start,
        end=end,
        text=" eins zwei",
        tokens=[],
        temperature=0.0,
        avg_logprob=0.0,
        compression_ratio=1.0,
        no_speech_prob=0.0,
        words=words,
        speaker=speaker,
    )


TURNS = [_turn(0.0, 10.0, "SPEAKER_00"), _turn(10.0, 20.0, "SPEAKER_01")]


def test_segment_in_the_other_speakers_voice_is_relabelled_with_its_words():
    embed = _voice_track((0.0, 10.0, VOICE_A), (10.0, 20.0, VOICE_B), (20.0, 30.0, VOICE_B))
    profiles = speaker_profiles(TURNS, embed)

    [segment] = relabel_segments([_segment(20.0, 22.0, "SPEAKER_00")], profiles, embed)

    assert segment.speaker == "SPEAKER_01"
    assert [word.speaker for word in segment.words or []] == ["SPEAKER_01", "SPEAKER_01"]


def test_segment_matching_its_own_speaker_keeps_label():
    embed = _voice_track((0.0, 10.0, VOICE_A), (10.0, 20.0, VOICE_B), (20.0, 30.0, VOICE_A))
    profiles = speaker_profiles(TURNS, embed)

    [segment] = relabel_segments([_segment(20.0, 22.0, "SPEAKER_00")], profiles, embed)

    assert segment.speaker == "SPEAKER_00"


def test_close_call_between_voices_keeps_label():
    ambiguous = np.array([1.0, 1.02])
    embed = _voice_track((0.0, 10.0, VOICE_A), (10.0, 20.0, VOICE_B), (20.0, 30.0, ambiguous))
    profiles = speaker_profiles(TURNS, embed)

    [segment] = relabel_segments([_segment(20.0, 22.0, "SPEAKER_00")], profiles, embed)

    assert segment.speaker == "SPEAKER_00"


def test_segment_too_short_for_a_reliable_voiceprint_keeps_label():
    embed = _voice_track((0.0, 10.0, VOICE_A), (10.0, 20.0, VOICE_B), (20.0, 30.0, VOICE_B))
    profiles = speaker_profiles(TURNS, embed)

    [segment] = relabel_segments([_segment(20.0, 20.4, "SPEAKER_00")], profiles, embed)

    assert segment.speaker == "SPEAKER_00"


def test_unattributed_segment_stays_unattributed():
    embed = _voice_track((0.0, 10.0, VOICE_A), (10.0, 20.0, VOICE_B), (20.0, 30.0, VOICE_B))
    profiles = speaker_profiles(TURNS, embed)

    [segment] = relabel_segments([_segment(20.0, 22.0, None)], profiles, embed)

    assert segment.speaker is None


def test_profiles_ignore_turns_with_crosstalk_and_short_turns():
    turns = [
        _turn(0.0, 10.0, "SPEAKER_00"),
        _turn(4.0, 5.0, "SPEAKER_01"),
        _turn(12.0, 14.0, "SPEAKER_01"),
        _turn(20.0, 30.0, "SPEAKER_02"),
    ]
    embed = _voice_track((0.0, 10.0, VOICE_A), (10.0, 20.0, VOICE_B), (20.0, 30.0, VOICE_B))

    profiles = speaker_profiles(turns, embed)

    assert sorted(profiles) == ["SPEAKER_02"]


def test_with_fewer_than_two_profiles_nothing_is_relabelled():
    embed = _voice_track((0.0, 10.0, VOICE_A), (10.0, 30.0, VOICE_B))
    profiles = speaker_profiles([_turn(0.0, 10.0, "SPEAKER_00")], embed)

    [segment] = relabel_segments([_segment(20.0, 22.0, "SPEAKER_00")], profiles, embed)

    assert segment.speaker == "SPEAKER_00"


def test_non_finite_embedding_keeps_label():
    embed_profiles = _voice_track((0.0, 10.0, VOICE_A), (10.0, 20.0, VOICE_B))
    profiles = speaker_profiles(TURNS, embed_profiles)

    [segment] = relabel_segments(
        [_segment(20.0, 22.0, "SPEAKER_00")], profiles, lambda start, end: np.array([np.nan, np.nan])
    )

    assert segment.speaker == "SPEAKER_00"
