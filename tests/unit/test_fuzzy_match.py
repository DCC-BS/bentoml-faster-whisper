from tests.fuzzy_match import (
    ReferenceWord,
    match_speakers,
    normalize_words,
    reference_turns,
    reference_words,
    text_similarity,
)


def test_normalize_words_ignores_case_punctuation_and_annotations():
    assert normalize_words("Das ist der Grund. [lacht] Okay!") == ["das", "ist", "der", "grund", "okay"]


def test_normalize_words_folds_eszett():
    assert normalize_words("heißen") == normalize_words("heissen")


def test_text_similarity_identical_after_normalization():
    assert text_similarity("Hat es bei dir auch geheissen?", "hat es bei dir auch geheißen") == 1.0


def test_text_similarity_detects_divergence():
    assert text_similarity("Hat es bei dir auch geheissen?", "Etwas komplett anderes hier") < 0.3


def test_reference_words_and_turns_group_speakers():
    reference = {
        "words": [
            {"text": "Hallo", "start": 0.0, "end": 0.5, "type": "word", "speaker_id": "speaker_0"},
            {"text": " ", "start": 0.5, "end": 0.6, "type": "spacing", "speaker_id": "speaker_0"},
            {"text": "du", "start": 0.6, "end": 0.8, "type": "word", "speaker_id": "speaker_0"},
            {"text": "Ja", "start": 1.0, "end": 1.2, "type": "word", "speaker_id": "speaker_1"},
        ]
    }

    words = reference_words(reference)
    turns = reference_turns(reference)

    assert [w.text for w in words] == ["Hallo", "du", "Ja"]
    assert [(t.start, t.end, t.speaker) for t in turns] == [(0.0, 0.8, "speaker_0"), (1.0, 1.2, "speaker_1")]


def test_match_speakers_is_label_invariant():
    ref = [
        ReferenceWord(0.0, 1.0, "a", "speaker_0"),
        ReferenceWord(1.0, 2.0, "b", "speaker_0"),
        ReferenceWord(2.0, 3.0, "c", "speaker_1"),
    ]
    # Same split, opposite label names: must count as full agreement.
    predicted = [(0.0, 2.0, "SPEAKER_01"), (2.0, 3.0, "SPEAKER_00")]

    match = match_speakers(ref, predicted)

    assert match.agreement == 1.0
    assert match.coverage == 1.0


def test_match_speakers_scores_wrong_assignment_and_gaps():
    ref = [
        ReferenceWord(0.0, 1.0, "a", "speaker_0"),
        ReferenceWord(1.0, 2.0, "b", "speaker_1"),
        ReferenceWord(5.0, 6.0, "c", "speaker_1"),  # not covered by any segment
    ]
    predicted = [(0.0, 2.0, "SPEAKER_00")]  # one speaker predicted for two ref speakers

    match = match_speakers(ref, predicted)

    assert match.coverage == 2 / 3
    assert match.agreement == 1 / 2
