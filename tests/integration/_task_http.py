"""Minimal ``requests`` driver for the BentoML task HTTP protocol.

BentoML's ``SyncHTTPClient.<task>.submit()`` crashes in its own cleanup path — the
``finally`` in ``_submit`` references ``self._opened_files``, but that list now lives on
``self._file_manager`` (``_bentoml_impl/client/http.py``). Until that upstream bug is fixed,
these integration tests talk to the task endpoints directly.

Task routes (relative to the task endpoint route, e.g. ``/v1/audio/transcriptions/task``):
    POST  {route}/submit   multipart file + form fields  -> {"task_id": ...}
    GET   {route}/status   ?task_id=...                  -> {"status": <ResultStatus value>}
    GET   {route}/get      ?task_id=...                  -> result body (HTTP error if not completed)

ResultStatus values: pending, in_progress, completed, failed, canceled.
"""

import time

import requests

TERMINAL_STATES = {"completed", "failed", "canceled"}


class TaskHandle:
    def __init__(self, base_url: str, route: str, task_id: str) -> None:
        self._base = base_url.rstrip("/")
        self._route = route
        self.task_id = task_id

    def _url(self, action: str) -> str:
        return f"{self._base}{self._route}/{action}"

    def get_status(self, timeout: float = 30) -> str:
        resp = requests.get(self._url("status"), params={"task_id": self.task_id}, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["status"]

    def get_result(self, timeout: float = 120) -> requests.Response:
        """Raw ``/get`` response. A non-completed (failed/canceled) task returns an HTTP error;
        the caller decides whether to ``raise_for_status()`` or inspect ``status_code``."""
        return requests.get(self._url("get"), params={"task_id": self.task_id}, timeout=timeout)

    def wait_for_terminal_state(self, timeout_s: float = 120.0, poll_s: float = 1.0) -> str:
        deadline = time.monotonic() + timeout_s
        state = self.get_status()
        while state not in TERMINAL_STATES:
            if time.monotonic() > deadline:
                raise AssertionError(f"task did not terminate within {timeout_s}s (last state: {state})")
            time.sleep(poll_s)
            state = self.get_status()
        return state


def submit_task(
    base_url: str,
    file,
    route: str = "/v1/audio/transcriptions/task",
    timeout: float = 60,
    **fields,
) -> TaskHandle:
    """Submit a task via multipart POST and return a :class:`TaskHandle`.

    Mirrors ``client.task_transcribe.submit(file=..., **fields)``. Booleans are lowercased
    so pydantic's form parsing accepts them (``str(True)`` -> ``"True"`` is fine too, but
    ``"true"`` matches the JSON/OpenAI convention the other tests use)."""
    base = base_url.rstrip("/")
    data = {k: (str(v).lower() if isinstance(v, bool) else str(v)) for k, v in fields.items()}
    with open(file, "rb") as f:
        resp = requests.post(f"{base}{route}/submit", files={"file": f}, data=data, timeout=timeout)
    resp.raise_for_status()
    return TaskHandle(base_url, route, resp.json()["task_id"])
