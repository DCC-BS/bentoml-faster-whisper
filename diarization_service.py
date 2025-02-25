import os
from typing import List

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
        return f"Segment: {self.segment}, Label: {self.label}, Speaker: {self.speaker}"

    def __repr__(self):
        return self.__str__()


class DiarizationService:
    """
    A service for speaker diarization using the pyannote.audio library.
    """

    @logger.catch
    def load(self):
        """
        Load the speaker diarization pipeline from the Hugging Face model hub.
        The pipeline is sent to the GPU if available.
        """
        logger.info("Loading speaker diarization pipeline")
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=os.getenv("HF_AUTH_TOKEN"),
        )

        # send pipeline to GPU (when available)
        self.pipeline.to(torch.device("cuda"))

    @logger.catch
    def diarize(
        self, audio_path: str, num_speaker: int | None = None
    ) -> List[DiarizationSegment]:
        """
        Perform speaker diarization on the given audio file.

        Args:
            audio_path (str): Path to the audio file to be diarized.
            num_speaker (int, optional): The number of speakers to be identified. Defaults to None.

        Returns:
            Pipeline: The diarization pipeline with the results.
        """

        # check if file exists
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"File not found: {audio_path}")

        if num_speaker is not None and num_speaker <= 0:
            raise ValueError("num_speaker must be a positive integer or None.")

        segments = self.pipeline(audio_path, num_speakers=num_speaker)
        logger.info("Diarization completed")

        diarization_segments = list(
            map(
                lambda x: DiarizationSegment(x[0], x[1], x[2]),
                segments.itertracks(yield_label=True),
            )
        )

        return diarization_segments
