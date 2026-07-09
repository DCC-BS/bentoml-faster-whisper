from types import SimpleNamespace

from core import Segment
from helpers.transcription_cleaner import clean_transcription_segments


def _segment(**overrides):
    defaults = dict(
        id=0,
        seek=0,
        start=0.0,
        end=2.0,
        text=" Mein Name ist Janik.",
        tokens=[],
        temperature=0.0,
        avg_logprob=-0.3,
        compression_ratio=1.1,
        no_speech_prob=0.05,
        words=None,
    )
    return Segment(**{**defaults, **overrides})


def _info(language="de", log_prob_threshold=-1.0):
    return SimpleNamespace(
        language=language,
        transcription_options=SimpleNamespace(log_prob_threshold=log_prob_threshold),
    )


def test_keeps_confident_speech_despite_high_no_speech_prob():
    # Regression for head60.mp3: the whole first 30s window carried
    # no_speech_prob ~0.94 while the decode itself was confident
    # (avg_logprob ~ -0.3). Dropping on no_speech_prob alone deleted
    # the first 23 seconds of real speech.
    segments = [_segment(no_speech_prob=0.94, avg_logprob=-0.30)]

    cleaned = list(clean_transcription_segments(segments, _info()))

    assert [s.text for s in cleaned] == [" Mein Name ist Janik."]


def test_drops_segment_when_no_speech_and_low_confidence():
    segments = [_segment(no_speech_prob=0.95, avg_logprob=-2.0)]

    assert list(clean_transcription_segments(segments, _info())) == []


def test_no_speech_alone_decides_when_log_prob_threshold_disabled():
    segments = [_segment(no_speech_prob=0.95, avg_logprob=-0.3)]

    assert list(clean_transcription_segments(segments, _info(log_prob_threshold=None))) == []


def test_drops_known_hallucination():
    segments = [_segment(text=" Untertitel der Amara.org-Community")]

    assert list(clean_transcription_segments(segments, _info())) == []


def test_drops_hallucination_in_segment_language_not_majority_language():
    # Multilingual file: the top-level language is the majority one, but a segment
    # decoded in another language must be matched against that language's blacklist.
    segments = [_segment(text=" www.mooji.org", language="en")]

    assert list(clean_transcription_segments(segments, _info(language="de"))) == []


def test_keeps_text_that_is_only_a_hallucination_in_another_language():
    segments = [_segment(text=" Untertitel der Amara.org-Community", language="en")]

    (segment,) = clean_transcription_segments(segments, _info(language="de"))
    assert segment.text == " Untertitel der Amara.org-Community"


def test_replaces_eszett():
    (segment,) = clean_transcription_segments([_segment(text=" Strauße")], _info())

    assert segment.text == " Strausse"
