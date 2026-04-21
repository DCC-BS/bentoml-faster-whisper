import contextlib
import os
import subprocess
import tempfile
from typing import Iterable

import torch
from loguru import logger
from pyannote.audio import Pipeline
from pyannote.core import Segment


class DiarizationSegment:
    def __init__(self, segment: Segment, label: str, speaker: str):
        self.segment = segment
        self.label = label
        self.speaker = speaker
        self.start = segment.start
        self.end = segment.end

    def __str__(self):
        return (
            f"Segment: {self.segment}, Label: {self.label}, Speaker: {self.speaker}, Time: [{self.start} - {self.end}]"
        )

    def __repr__(self):
        return self.__str__()


@contextlib.contextmanager
def _as_wav(audio_path: str):
    # MP3/FLAC headers report duration imprecisely; convert to WAV so pyannote
    # gets an exact sample count without loading the whole file into RAM.
    if audio_path.endswith(".wav"):
        yield audio_path
        return

    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", tmp_path],
            check=True,
            capture_output=True,
        )
        yield tmp_path
    finally:
        os.unlink(tmp_path)


class DiarizationService:
    """
    A service for speaker diarization using the pyannote.audio library.
    """

    pipeline: Pipeline

    @logger.catch(reraise=True)
    def load(self):
        """
        Load the speaker diarization pipeline from the Hugging Face model hub.
        The pipeline is sent to the GPU if available.
        """
        logger.info("Loading speaker diarization pipeline")
        hf_token = os.getenv("HF_AUTH_TOKEN")
        pipeline = Pipeline.from_pretrained(  # type: ignore[call-arg]
            "pyannote/speaker-diarization-community-1",
            token=hf_token or None,
        )
        assert pipeline is not None, "Failed to load diarization pipeline"
        self.pipeline = pipeline

        try:
            self.pipeline._models.segmentation_batch_size = 4  # type: ignore
            self.pipeline._models.embedding_batch_size = 4  # type: ignore
        except AttributeError:
            logger.warning("Could not set batch sizes on pipeline._models — skipping")

        self.pipeline.to(torch.device("cuda"))

    @logger.catch(reraise=True)
    def diarize(self, audio_path: str, num_speaker: int | None = None) -> Iterable[DiarizationSegment]:
        """
        Perform speaker diarization on the given audio file.

        Args:
            audio_path (str): Path to the audio file to be diarized.
            num_speaker (int, optional): The number of speakers to be identified. Defaults to None.

        Returns:
            Pipeline: The diarization pipeline with the results.
        """

        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"File not found: {audio_path}")

        if num_speaker is not None and num_speaker <= 0:
            raise ValueError("num_speaker must be a positive integer or None.")

        with _as_wav(audio_path) as wav_path:
            output = self.pipeline(wav_path, num_speakers=num_speaker)

        logger.info("Diarization completed")

        for turn, speaker in output.speaker_diarization:
            yield DiarizationSegment(turn, speaker, speaker)
