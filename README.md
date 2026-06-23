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

### CUDA / torchcodec version coupling (maintenance note)

The stack is currently pinned to **CUDA 12.8 (cu128)**. `torch`, `torchaudio` and
`torchcodec` are all routed to the `pytorch-cu128` index in `pyproject.toml`
(`[tool.uv.sources]`). This coupling is load-bearing:

- The cu128 index only ships `torchcodec` up to **0.11.1**, which matches **torch
  2.11**. If `torchcodec` is left unpinned it resolves to a newer PyPI build
  (e.g. 0.14.0) compiled for **CUDA 13**, which fails to load at runtime with
  `libnvrtc.so.13: cannot open shared object file`. That breaks pyannote
  diarization (`NameError: name 'AudioDecoder' is not defined`).
- So torch on linux/win is effectively capped at **2.11** until we move to CUDA 13.

The Docker image uses the slim `nvidia/cuda:*-base-*` flavor rather than
`-runtime-*`. The pip cu128 wheels bundle most CUDA libs, but **NPP is not
bundled** and the `base` image doesn't ship it — torchcodec links it
(`libnppicc.so.12`). Two pieces are needed:
1. the `nvidia-npp-cu12` dependency (linux/win) installs the lib into the venv;
2. torchcodec's `.so` has no RPATH to that wheel, so the Dockerfile adds the
   wheel dir (`.../nvidia/npp/lib`) to `LD_LIBRARY_PATH`.

If you ever switch back to the `-runtime-` image, NPP is on the system loader
path there, so both the dependency and the `LD_LIBRARY_PATH` line become
redundant.

**To upgrade to CUDA 13 (cu130):**
1. Bump the base image in `Dockerfile`: `nvidia/cuda:12.8.0-base-ubuntu24.04`
   → `nvidia/cuda:13.0.0-base-ubuntu24.04`.
2. In `pyproject.toml`, change the index URL and name:
   `https://download.pytorch.org/whl/cu128` → `.../cu130`, and update the
   `pytorch-cu128` index name + all `[tool.uv.sources]` markers (`torch`,
   `torchvision`, `torchaudio`, `torchcodec`) to point at it.
3. Relax/raise the `torchcodec>=0.11` pin (cu130 ships 0.12+, paired with torch
   ≥2.11). Let `torch`/`torchaudio` resolve to their cu130 builds.
4. Swap the NPP wheel to the CUDA-13 series: `nvidia-npp-cu12` → `nvidia-npp-cu13`.
5. `uv lock`, then verify the codec loads:
   `uv run python -c "from pyannote.audio.core.io import AudioDecoder"`.
6. Ensure the deploy host has an NVIDIA driver new enough for CUDA 13.

Compatibility reference: [torchcodec version table](https://github.com/pytorch/torchcodec?tab=readme-ov-file#installing-torchcodec).

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
