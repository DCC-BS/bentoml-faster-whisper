"""Top-level language selection over multilingual decode runs.

``turns_to_language_runs`` caps each run at ``MAX_RUN_S``, so a language spanning a
long stretch is split into several runs. The reported top-level language must be the
one with the most *total* speech, not the one owning the single longest run.
"""

from bentoml_faster_whisper.services.faster_whisper_handler import _majority_run_language


def test_majority_language_sums_runs_not_single_longest():
    # "de" is fragmented into two 60s runs (120s total) by the length cap; "en" sits in one
    # 70s run. Summed per language "de" wins; picking the longest single run would return "en".
    runs = [
        ("de", [(0.0, 60.0)]),
        ("en", [(60.0, 130.0)]),
        ("de", [(130.0, 190.0)]),
    ]

    assert _majority_run_language(runs) == "de"


def test_majority_language_single_language():
    runs = [("fr", [(0.0, 30.0)]), ("fr", [(30.0, 45.0)])]

    assert _majority_run_language(runs) == "fr"


def test_majority_language_counts_every_interval_in_a_run():
    # A run can carry several intervals (padded/merged turns); all count toward its language.
    runs = [
        ("en", [(0.0, 5.0), (6.0, 40.0)]),  # 39s
        ("de", [(40.0, 70.0)]),  # 30s
    ]

    assert _majority_run_language(runs) == "en"
