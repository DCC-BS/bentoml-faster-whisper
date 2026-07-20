import contextlib
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, cast

import pyannote.audio as _pyannote_audio
import torch
from pyannote.audio import Pipeline
from pyannote.core import Segment

from helpers.logger import get_logger, log_exceptions
from helpers.utils import clamp, positive_env

logger = get_logger(__name__)

# Ordered weights for the pyannote speaker-diarization steps (they fire in this order). "embeddings"
# is the heavy GPU step that reports granular completed/total, so it gets the bulk of the band.
_DIARIZATION_STEP_WEIGHTS: dict[str, float] = {
    "segmentation": 0.15,
    "speaker_counting": 0.05,
    "embeddings": 0.70,
    "discrete_diarization": 0.10,
}


class _DiarizationProgressHook:
    """pyannote-compatible hook that maps internal diarization steps to a monotonic 0..1 fraction
    and forwards it to ``on_progress``. Unknown steps are ignored and the value never decreases, so
    reordered/extra steps from a future pyannote version degrade gracefully instead of breaking."""

    def __init__(self, on_progress: Callable[[float], None]) -> None:
        self._on_progress = on_progress
        self._last = 0.0
        self._base: dict[str, float] = {}
        offset = 0.0
        for name, weight in _DIARIZATION_STEP_WEIGHTS.items():
            self._base[name] = offset
            offset += weight

    def __enter__(self) -> "_DiarizationProgressHook":
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def __call__(
        self,
        step_name: str,
        step_artifact: Any,
        file: Mapping | None = None,
        total: int | None = None,
        completed: int | None = None,
    ) -> None:
        weight = _DIARIZATION_STEP_WEIGHTS.get(step_name)
        if weight is None:
            return  # unknown step: leave the bar where it is
        within = (completed / total) if (total and completed is not None) else 1.0
        within = clamp(within, 0.0, 1.0)
        fraction = min(self._base[step_name] + weight * within, 1.0)
        if fraction <= self._last:
            return  # pyannote emits completed=0 first; never move backwards
        self._last = fraction
        self._on_progress(fraction)


class DiarizationSegment:
    def __init__(self, segment: Segment, speaker: str):
        self.segment = segment
        self.speaker = speaker
        self.start = segment.start
        self.end = segment.end

    def __str__(self):
        return f"Segment: {self.segment}, Speaker: {self.speaker}, Time: [{self.start} - {self.end}]"

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
            logger.error("ffmpeg conversion failed", stderr=e.stderr.decode(errors="replace"))
            raise
        yield tmp_path
    finally:
        Path(tmp_path).unlink(missing_ok=True)


class DiarizationService:
    """
    A service for speaker diarization using the pyannote.audio library.
    """

    def __init__(self) -> None:
        self.pipeline: Pipeline | None = None
        # Guards both lazy loading and inference: a pyannote pipeline is not thread-safe.
        self._lock = threading.Lock()
        # Lower batch sizes to reduce peak GPU activation memory on large files / tight GPUs.
        self._segmentation_batch_size = positive_env("DIARIZATION_SEGMENTATION_BATCH_SIZE", 4, int)
        self._embedding_batch_size = positive_env("DIARIZATION_EMBEDDING_BATCH_SIZE", 4, int)

    @log_exceptions
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
            hf_token = os.getenv("HF_TOKEN")
            pipeline = Pipeline.from_pretrained(  # type: ignore[call-arg]
                "pyannote/speaker-diarization-community-1",
                token=hf_token or None,
            )
            if pipeline is None:
                raise RuntimeError("Failed to load diarization pipeline")

            _version = getattr(_pyannote_audio, "__version__", "unknown")
            logger.info("pyannote.audio loaded", version=_version)
            try:
                pipeline._models.segmentation_batch_size = self._segmentation_batch_size  # type: ignore
                pipeline._models.embedding_batch_size = self._embedding_batch_size  # type: ignore
            except AttributeError:
                logger.warning(
                    "pipeline._models batch-size attributes not found; private API may have "
                    "changed — batch sizes not configured",
                    version=_version,
                )

            if torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))

            self.pipeline = pipeline

    @log_exceptions
    def diarize(
        self,
        audio_path: str,
        num_speaker: int | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> Iterable[DiarizationSegment]:
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
                    if progress_callback is not None:
                        with _DiarizationProgressHook(progress_callback) as hook:
                            output = self.pipeline(wav_path, num_speakers=num_speaker, hook=hook)
                    else:
                        output = self.pipeline(wav_path, num_speakers=num_speaker)
                finally:
                    # Return this run's activation blocks to the allocator for the Whisper model / next request.
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        logger.info("Diarization completed")

        # pyannote types __call__ loosely; the speaker-diarization pipeline returns an
        # object carrying `speaker_diarization`, which the static union can't express.
        for turn, speaker in cast(Any, output).speaker_diarization:
            yield DiarizationSegment(turn, speaker)
