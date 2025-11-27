from typing import Iterable

from faster_whisper.transcribe import TranscriptionInfo

from helpers.hallucinations import detect_hallucinations
from helpers.whiper_diarization_merger import WhisperSegment


def clean_transcription_segments(
    whisper_segments: Iterable[WhisperSegment], transcription_info: TranscriptionInfo
) -> Iterable[WhisperSegment]:
    language = transcription_info.language
    for segment in whisper_segments:
        if segment.no_speech_prob > 0.9:
            continue
        if detect_hallucinations(segment.text.strip(), language):
            continue
        yield segment
