import contextlib
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, cast

import av
import pyannote.audio as _pyannote_audio
import torch
from bentoml.exceptions import InvalidArgument
from pyannote.audio import Pipeline
from pyannote.core import Segment

from bentoml_faster_whisper.utils.core import clamp, positive_env
from bentoml_faster_whisper.utils.logger import get_logger, log_exceptions

logger = get_logger(__name__)

_DIARIZATION_STEP_WEIGHTS: dict[str, float] = {
    "segmentation": 0.15,
    "speaker_counting": 0.05,
    "embeddings": 0.70,
    "discrete_diarization": 0.10,
}


class _DiarizationProgressHook:
    """pyannote-compatible hook mapping internal diarization steps to a monotonic 0..1 fraction."""

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
            return
        within = (completed / total) if (total and completed is not None) else 1.0
        within = clamp(within, 0.0, 1.0)
        fraction = min(self._base[step_name] + weight * within, 1.0)
        if fraction <= self._last:
            return
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


def _is_16k_mono_wav(audio_path: str) -> bool:
    """Check if the audio file is already 16 kHz mono WAV."""
    try:
        with av.open(audio_path) as container:
            stream = container.streams.audio[0]
            cc = stream.codec_context
            return cc.sample_rate == 16000 and cc.layout is not None and cc.layout.nb_channels == 1
    except Exception:
        return False


@contextlib.contextmanager
def _as_wav(audio_path: str) -> Iterator[str]:
    """Context manager converting audio to 16 kHz mono WAV if necessary."""
    if Path(audio_path).suffix.lower() == ".wav" and _is_16k_mono_wav(audio_path):
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
            stderr = e.stderr.decode(errors="replace")
            logger.warning("ffmpeg audio conversion failed", stderr=stderr)
            raise InvalidArgument("Failed to decode audio file") from e
        yield tmp_path
    finally:
        Path(tmp_path).unlink(missing_ok=True)


class DiarizationService:
    def __init__(self) -> None:
        self.pipeline: Pipeline | None = None
        self._lock = threading.Lock()
        self._segmentation_batch_size = positive_env("DIARIZATION_SEGMENTATION_BATCH_SIZE", 32, int)
        self._embedding_batch_size = positive_env("DIARIZATION_EMBEDDING_BATCH_SIZE", 32, int)

    @log_exceptions
    def load(self):
        """Load the speaker diarization pipeline from Hugging Face model hub."""
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
                pipeline.segmentation_batch_size = self._segmentation_batch_size  # type: ignore[attr-defined]
                pipeline.embedding_batch_size = self._embedding_batch_size  # type: ignore[attr-defined]
            except AttributeError:
                logger.warning(
                    "pipeline batch-size attributes not found; pyannote API may have changed — "
                    "batch sizes not configured",
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
        """Perform speaker diarization on the given audio file."""
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"File not found: {audio_path}")

        if num_speaker is not None and num_speaker <= 0:
            raise ValueError("num_speaker must be a positive integer or None.")

        self.load()
        assert self.pipeline is not None

        with _as_wav(audio_path) as wav_path:
            with self._lock:
                try:
                    if progress_callback is not None:
                        with _DiarizationProgressHook(progress_callback) as hook:
                            output = self.pipeline(wav_path, num_speakers=num_speaker, hook=hook)
                    else:
                        output = self.pipeline(wav_path, num_speakers=num_speaker)
                finally:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        logger.debug("Diarization completed")

        for turn, speaker in cast(Any, output).speaker_diarization:
            yield DiarizationSegment(turn, speaker)
