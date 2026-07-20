import time
from pathlib import Path

import bentoml
import pytest
import requests

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

    @pytest.mark.skip(
        reason="Only for development purposes. This test only succeeds if the bentoml service is "
        "running at port 8003. Remove this comment for development."
    )
    def test_transcribe_endpoint(self):
        # given
        client = bentoml.SyncHTTPClient("http://localhost:8003")
        file = Path(__file__).resolve().parent.parent / "assets" / "example_audio.mp3"

        # when
        result = client.transcribe(file=file)  # type: ignore

        # then
        assert "I am just a sample audio text." in result

    @pytest.mark.skip(
        reason="Only for development purposes. This test only succeeds if the bentoml service is "
        "running at port 8003. Remove this comment for development."
    )
    def test_transcribe_batch_endpoint(self):
        # given
        client = bentoml.SyncHTTPClient("http://localhost:8003")
        file = Path(__file__).resolve().parent.parent / "assets" / "example_audio.mp3"

        # when
        result = client.batch_transcribe(file=file)  # type: ignore

        # then
        assert "I am just a sample audio text." in result

    @pytest.mark.skip(
        reason="Only for development purposes. This test only succeeds if the bentoml service is "
        "running at port 8003. Remove this comment for development."
    )
    def test_transcribe_task_endpoint(self):
        # given
        file = Path(__file__).resolve().parent.parent / "assets" / "example_audio.mp3"
        client = bentoml.SyncHTTPClient("http://localhost:8003")

        # when
        task = client.task_transcribe.submit(file=file)  # type: ignore
        while True:
            status = task.get_status()
            if str(status) == "ResultStatus.SUCCESS":
                break
            time.sleep(5)
        result = task.get()

        # then
        assert "I am just a sample audio text." in result

    @pytest.mark.skip(
        reason="Only for development purposes. This test only succeeds if the bentoml service is "
        "running at port 8003. Remove this comment for development."
    )
    def test_transcribe_streaming_endpoint(self):
        # given
        client = bentoml.SyncHTTPClient("http://localhost:8003")
        file = Path(__file__).resolve().parent.parent / "assets" / "example_audio.mp3"
        data_chunks = []

        # when
        for data_chunk in client.streaming_transcribe(file=file):  # type: ignore
            data_chunks.append(data_chunk)
        client.close()

        # then
        assert data_chunks is not None
