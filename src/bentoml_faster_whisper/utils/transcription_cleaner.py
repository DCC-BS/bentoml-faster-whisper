from typing import Iterable

from faster_whisper.transcribe import TranscriptionInfo

from bentoml_faster_whisper.utils.hallucinations import detect_hallucinations
from bentoml_faster_whisper.utils.whisper_diarization_merger import WhisperSegment

NO_SPEECH_PROB_THRESHOLD = 0.9


def clean_transcription_segments(
    whisper_segments: Iterable[WhisperSegment],
    transcription_info: TranscriptionInfo,
    text_language: str | None = None,
) -> Iterable[WhisperSegment]:
    """Filter out silence and hallucinations from segments, and normalize text."""
    log_prob_threshold = transcription_info.transcription_options.log_prob_threshold
    for segment in whisper_segments:
        if segment.no_speech_prob > NO_SPEECH_PROB_THRESHOLD and (
            log_prob_threshold is None or segment.avg_logprob < log_prob_threshold
        ):
            continue

        hallucination_language = text_language or segment.language or transcription_info.language
        if detect_hallucinations(segment.text.strip(), hallucination_language):
            continue
        segment.text = segment.text.replace("ß", "ss")
        for word in segment.words or []:
            word.word = word.word.replace("ß", "ss")
        yield segment
