import threading

from api_models.ProgressResponse import ProgressResponse

_MAX_TRACKED_PROGRESS = 1000


class ProgressHandler:
    def __init__(self) -> None:
        # Per-instance and lock-guarded: a class-level dict would be shared and mutated across concurrent requests.
        self.progress_dict: dict[str, ProgressResponse] = {}
        self._lock = threading.Lock()

    def _set_locked(self, id: str, progress: ProgressResponse) -> None:
        """Stores progress and evicts the oldest entry past the cap. Caller must hold the lock."""
        self.progress_dict[id] = progress

        if len(self.progress_dict) > _MAX_TRACKED_PROGRESS:
            self.progress_dict.pop(next(iter(self.progress_dict)))

    def add_progress(self, id: str) -> None:
        """Adds a new task to the progress tracker."""
        with self._lock:
            self._set_locked(id, ProgressResponse(progress=0, currentTime=0, duration=0))

    def update_progress(self, id: str, progress: ProgressResponse) -> None:
        """Updates the progress of a given task."""
        with self._lock:
            self._set_locked(id, progress)

    def get_progress(self, id: str) -> ProgressResponse:
        """Retrieves the progress of a given task. Returns 0 if the task is not found."""
        with self._lock:
            return self.progress_dict.get(id, ProgressResponse(progress=0, currentTime=0, duration=0))

    def remove_progress(self, id: str) -> None:
        """Removes a task from the progress tracker."""
        with self._lock:
            self.progress_dict.pop(id, None)
