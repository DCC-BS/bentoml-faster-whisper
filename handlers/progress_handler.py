from api_models.ProgressResponse import ProgressResponse


class ProgressHandler:
    progress_dict: dict[str, ProgressResponse] = {}

    def add_progress(self, id: str) -> None:
        """Adds a new task to the progress tracker."""
        self.progress_dict[id] = ProgressResponse(progress=0, currentTime=0, duration=0)

    def update_progress(self, id: str, progress: ProgressResponse) -> None:
        """Updates the progress of a given task."""
        self.progress_dict[id] = progress

    def get_progress(self, id: str) -> ProgressResponse:
        """Retrieves the progress of a given task. Returns 0 if the task is not found."""
        return self.progress_dict.get(id, ProgressResponse(progress=0, currentTime=0, duration=0))

    def remove_progress(self, id: str) -> None:
        """Removes a task from the progress tracker."""
        self.progress_dict.pop(id, None)
