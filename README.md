<div align="center">
    <h1 align="center">Serving FasterWhisper with BentoML</h1>
</div>

[FasterWhisper](https://github.com/SYSTRAN/faster-whisper) provides fast automatic speech recognition with word-level timestamps.


## Prerequisites

- If you want to test the project locally, install FFmpeg on your system.
- Install the package manager uv ([docs](https://docs.astral.sh/uv/getting-started/installation/)).
- Python 3.13 is recommended.

## Install dependencies

```bash
git clone https://github.com/DCC-BS/bentoml-faster-whisper.git
cd bentoml-faster-whisper

uv sync
```

## Run the BentoML Service

We have defined a BentoML Service in `service.py`. Use uv to start the service in your project directory:

```bash
uv run bentoml serve service:FasterWhisper
```

2024-01-18T09:01:15+0800 [INFO] [cli] Starting production HTTP BentoServer from "service:FasterWhisper" listening on http://localhost:3000 (Press CTRL+C to quit)

The server is now active at [http://localhost:3000](http://localhost:3000/). You can interact with it using the Swagger UI or in other different ways.

#### CURL

```bash
curl -s \
     -X POST \
     -F 'audio_file=@female.wav' \
     http://localhost:3000/v1/audio/transcriptions
```

#### Python client

```python
import bentoml

with bentoml.SyncHTTPClient('http://localhost:3000') as client:
    audio_url = 'https://example.org/female.wav'
    response = client.transcribe(file=audio_url)
    print(response)
```

Further examples (task, streaming) how to programmatically interact with the faster_whisper service can be found in `test_integration.py`

### Local Development

To debug through the FasterWhisper service, you can run the service with the following script: 
```bash
uv run python launch.py
```

## Deploy

### Build an Image

For custom deployment in your own infrastructure, you can build and containerize the faster_whisper service.
```bash
docker build -t faster_whisper:latest .
```

### Run Container with NVIDIA GPU Support
You can run the prebuilt docker image with NVIDIA GPU support using the following command:
```bash
docker run --gpus all -p 50001:50001 faster_whisper:<IMAGE-TAG>
```

You can use the compose.yaml file to build an image and run the container with GPU support:
```bash
docker-compose up --build
```

Documentation: [BentoML to generate an OCI-compliant image](https://docs.bentoml.com/en/latest/guides/containerization.html)

## Observability

BentoML automatically collects a set of default metrics for each Service and exposes them via '/metrics' endpoint.

### Prometheus (local)

To run a prometheus server locally, you need to do the following:
- Install prometheus
- Start prometheus server
```bash
prometheus --config.file=/path/to/the/file/prometheus.yml
```
- Access the web UI by visiting `http://localhost:9090`

### Grafana

Tutorial link: [link](https://docs.bentoml.com/en/latest/build-with-bentoml/observability/metrics.html#create-a-grafana-dashboard)
- Install grafana
- Change http_port to a free port like 4000 in `grafana.ini` file.
- Restart grafana server
