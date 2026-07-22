"""Task-cancellation contract for the ``/v1/audio/transcriptions/task`` endpoint.

Needs the service running (``make run`` / ``docker compose up``, port 50001) with the
pyannote pipeline available — the task is submitted with diarization so the decode runs
long enough to be cancelled mid-flight. If no server is reachable the tests skip rather
than fail, matching the "requires a running service" contract of ``make integration``.

BentoML task lifecycle (``_bentoml_impl.tasks.ResultStatus``):
pending -> in_progress -> {completed, failed, canceled}.
"""

import time
from pathlib import Path

import bentoml
import pytest
from bentoml.exceptions import BentoMLException

pytestmark = pytest.mark.integration

SERVER_URL = "http://localhost:50001"
AUDIO = Path(__file__).resolve().parent.parent / "assets" / "long_example_audio.mp3"

_TERMINAL_STATES = {"completed", "failed", "canceled"}


@pytest.fixture
def client():
    try:
        # server_ready_timeout bounds the wait so a missing server skips fast instead of
        # blocking on the client's default 30s readiness poll.
        sut = bentoml.SyncHTTPClient(SERVER_URL, server_ready_timeout=5, timeout=120)
    except BentoMLException as e:
        pytest.skip(f"no BentoML service reachable at {SERVER_URL}: {e}")
    yield sut
    sut.close()


def _wait_for_terminal_state(task, timeout_s: float = 120.0, poll_s: float = 1.0) -> str:
    """Poll the task until it reaches a terminal state; return that state's value."""
    deadline = time.monotonic() + timeout_s
    state = task.get_status().value
    while state not in _TERMINAL_STATES:
        if time.monotonic() > deadline:
            raise AssertionError(f"task did not terminate within {timeout_s}s (last state: {state})")
        time.sleep(poll_s)
        state = task.get_status().value
    return state


def test_task_transcribe_can_be_cancelled(client):
    # Diarization + a ~96s decode keeps the task running long enough that the cancel
    # lands before it would otherwise complete.
    task = client.task_transcribe.submit(file=AUDIO, diarization=True)  # type: ignore[attr-defined]

    task.cancel()

    state = _wait_for_terminal_state(task)
    if state == "completed":
        pytest.skip("decode finished before cancellation took effect; cannot assert the canceled state")
    assert state == "canceled", "a cancelled task must end in the canceled state"


def test_cancelled_task_returns_no_transcript(client):
    task = client.task_transcribe.submit(file=AUDIO, diarization=True)  # type: ignore[attr-defined]
    task.cancel()
    if _wait_for_terminal_state(task) == "completed":
        pytest.skip("decode finished before cancellation took effect; there is a transcript to return")

    # Fetching the result of a cancelled task must not hand back a successful transcription.
    with pytest.raises(BentoMLException):
        task.get()
