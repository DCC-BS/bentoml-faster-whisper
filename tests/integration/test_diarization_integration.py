from diarization_service import DiarizationService


def test_diarize():
    sut = DiarizationService()
    sut.load()

    segments = list(sut.diarize("./tests/assets/example_audio_german.mp3"))
    assert len(segments) == 1
