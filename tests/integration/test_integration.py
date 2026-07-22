from pathlib import Path

import bentoml
import pytest
import requests

from tests.integration._task_http import submit_task

pytestmark = pytest.mark.integration


class TestIntegration:
    def test_invalid_parameter(self):
        # given
        file = Path(__file__).resolve().parent.parent / "assets" / "example_audio.mp3"

        # when
        with open(file, "rb") as f:
            response = requests.post(
                "http://localhost:50001/v1/audio/transcriptions",
                files={"file": f},
                data={"schmutzgeier": "schmutzgeier"},
                timeout=30,
            )

        # Assert that the response status code is 401 because the parameter 'schmutzgeier' is invalid
        assert response.status_code == 400, response.text

    def test_transcribe_endpoint(self):
        # given
        client = bentoml.SyncHTTPClient("http://localhost:50001")
        file = Path(__file__).resolve().parent.parent / "assets" / "example_audio.mp3"

        # when
        result = client.transcribe(file=file)  # type: ignore

        # then: the json response deserializes to {"text": ...}
        assert "I am just a sample audio text." in result["text"]

    def test_transcribe_batch_endpoint(self):
        # given
        client = bentoml.SyncHTTPClient("http://localhost:50001")
        file = Path(__file__).resolve().parent.parent / "assets" / "example_audio.mp3"

        # when
        result = client.batch_transcribe(file=file)  # type: ignore

        # then: the json response deserializes to {"text": ...}
        assert "I am just a sample audio text." in result["text"]

    def test_transcribe_task_endpoint(self):
        # given
        file = Path(__file__).resolve().parent.parent / "assets" / "example_audio.mp3"

        # when: driven over plain HTTP — SyncHTTPClient.<task>.submit() is broken upstream
        task = submit_task("http://localhost:50001", file)
        assert task.wait_for_terminal_state() == "completed"
        result = task.get_result()
        result.raise_for_status()

        # then
        assert "I am just a sample audio text." in result.json()["text"]

    def test_transcribe_streaming_endpoint(self):
        # given
        client = bentoml.SyncHTTPClient("http://localhost:50001")
        file = Path(__file__).resolve().parent.parent / "assets" / "example_audio.mp3"
        data_chunks = []

        # when
        for data_chunk in client.streaming_transcribe(file=file):  # type: ignore
            data_chunks.append(data_chunk)
        client.close()

        # then
        assert data_chunks is not None
