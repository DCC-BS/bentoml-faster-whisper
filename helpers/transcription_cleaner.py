from typing import Iterable

from faster_whisper.transcribe import TranscriptionInfo

from helpers.hallucinations import detect_hallucinations
from helpers.whiper_diarization_merger import WhisperSegment

# no_speech_prob alone is unreliable: whole windows of confidently decoded speech can
# score > 0.9 (head60.mp3 lost its first 23 seconds this way). Mirror the Whisper
# reference rule instead: treat a segment as silence only when the no-speech probe
# fires AND the decode itself was unconfident.
NO_SPEECH_PROB_THRESHOLD = 0.9


def clean_transcription_segments(
    whisper_segments: Iterable[WhisperSegment], transcription_info: TranscriptionInfo
) -> Iterable[WhisperSegment]:
    log_prob_threshold = transcription_info.transcription_options.log_prob_threshold
    for segment in whisper_segments:
        if segment.no_speech_prob > NO_SPEECH_PROB_THRESHOLD and (
            log_prob_threshold is None or segment.avg_logprob < log_prob_threshold
        ):
            continue
        # transcription_info.language is the majority language of a multilingual file; a segment
        # decoded in another language must be matched against that language's blacklist.
        if detect_hallucinations(segment.text.strip(), segment.language or transcription_info.language):
            continue
        segment.text = segment.text.replace("ß", "ss")
        yield segment
