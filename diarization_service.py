import contextlib
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Iterable, Iterator

import pyannote.audio as _pyannote_audio
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
def _as_wav(audio_path: str) -> Iterator[str]:
    # MP3/FLAC headers report duration imprecisely; convert to WAV so pyannote
    # gets an exact sample count without loading the whole file into RAM.
    if Path(audio_path).suffix.lower() == ".wav":
        yield audio_path
        return

    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", tmp_path],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error("ffmpeg conversion failed: {}", e.stderr.decode(errors="replace"))
            raise
        yield tmp_path
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _positive_int_env(name: str, default: int) -> int:
    """Parse a positive-int env var, falling back to default on missing/invalid/non-positive values."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid value for %s: %r; using default %d", name, raw, default)
        return default
    if value < 1:
        logger.warning("%s must be >= 1, got %d; using default %d", name, value, default)
        return default
    return value


class DiarizationService:
    """
    A service for speaker diarization using the pyannote.audio library.
    """

    def __init__(self) -> None:
        self.pipeline: Pipeline | None = None
        # Guards both lazy loading and inference: a pyannote pipeline is not thread-safe.
        self._lock = threading.Lock()
        # Lower batch sizes to reduce peak GPU activation memory on large files / tight GPUs.
        self._segmentation_batch_size = _positive_int_env("DIARIZATION_SEGMENTATION_BATCH_SIZE", 4)
        self._embedding_batch_size = _positive_int_env("DIARIZATION_EMBEDDING_BATCH_SIZE", 4)

    @logger.catch(reraise=True)
    def load(self):
        """
        Load the speaker diarization pipeline from the Hugging Face model hub.
        The pipeline is sent to the GPU if available. Idempotent and lazy: loading
        only happens on first use so workers that never diarize don't pin a
        pipeline to the GPU.
        """
        with self._lock:
            if self.pipeline is not None:
                return
            logger.info("Loading speaker diarization pipeline")
            hf_token = os.getenv("HF_AUTH_TOKEN")
            pipeline = Pipeline.from_pretrained(  # type: ignore[call-arg]
                "pyannote/speaker-diarization-community-1",
                token=hf_token or None,
            )
            if pipeline is None:
                raise RuntimeError("Failed to load diarization pipeline")

            _version = getattr(_pyannote_audio, "__version__", "unknown")
            logger.info("pyannote.audio version: {}", _version)
            try:
                pipeline._models.segmentation_batch_size = self._segmentation_batch_size  # type: ignore
                pipeline._models.embedding_batch_size = self._embedding_batch_size  # type: ignore
            except AttributeError:
                logger.warning(
                    "pipeline._models batch-size attributes not found (pyannote.audio {}); "
                    "private API may have changed — batch sizes not configured",
                    _version,
                )

            if torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))

            self.pipeline = pipeline

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

        self.load()
        assert self.pipeline is not None  # guaranteed by load()

        with _as_wav(audio_path) as wav_path:
            with self._lock:
                try:
                    output = self.pipeline(wav_path, num_speakers=num_speaker)
                finally:
                    # Return this run's activation blocks to the allocator for the Whisper model / next request.
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        logger.info("Diarization completed")

        for turn, speaker in output.speaker_diarization:
            yield DiarizationSegment(turn, speaker, speaker)
