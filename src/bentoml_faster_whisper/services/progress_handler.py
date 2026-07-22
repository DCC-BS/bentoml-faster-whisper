import threading

from bentoml_faster_whisper.models.progress_response import ProgressResponse

_MAX_TRACKED_PROGRESS = 1000


class ProgressHandler:
    def __init__(self) -> None:
        self.progress_dict: dict[str, ProgressResponse] = {}
        self._lock = threading.Lock()

    def _set_locked(self, id: str, progress: ProgressResponse) -> None:
        """Stores progress and evicts the least-recently-updated entry past the cap.
        Caller must hold the lock."""
        self.progress_dict.pop(id, None)
        self.progress_dict[id] = progress

        if len(self.progress_dict) > _MAX_TRACKED_PROGRESS:
            self.progress_dict.pop(next(iter(self.progress_dict)))

    def add_progress(self, id: str) -> None:
        with self._lock:
            self._set_locked(id, ProgressResponse(progress=0, currentTime=0, duration=0))

    def update_progress(self, id: str, progress: ProgressResponse) -> None:
        with self._lock:
            self._set_locked(id, progress)

    def get_progress(self, id: str) -> ProgressResponse:
        with self._lock:
            return self.progress_dict.get(id, ProgressResponse(progress=0, currentTime=0, duration=0))

    def remove_progress(self, id: str) -> None:
        with self._lock:
            self.progress_dict.pop(id, None)
