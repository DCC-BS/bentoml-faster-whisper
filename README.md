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
     -F 'file=@female.wav' \
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

### Speaker diarization

The service bundles [pyannote](https://github.com/pyannote/pyannote-audio) speaker diarization
and runs it **by default** on every transcription. Set `diarization=false` in the request to
skip it. Diarization needs an `HF_AUTH_TOKEN` env var with access to the
`pyannote/speaker-diarization-community-1` model.

```bash
curl -s \
     -X POST \
     -F 'file=@meeting.wav' \
     -F 'response_format=json_diarized' \
     http://localhost:3000/v1/audio/transcriptions
```

Set `diarization_speaker_count` (1–6) to fix the number of speakers; leave it unset to let
pyannote estimate it.

**One VAD, not two.** When diarization is on, pyannote's speech turns double as the voice
activity detector: Whisper only decodes the detected speech regions (via `clip_timestamps`)
instead of running its own Silero VAD over the whole file. This avoids two independent VADs
disagreeing (Silero cutting speech that pyannote labels, or vice versa) and skips redundant
decoding of silence. Silero (`vad_filter`) is only used as a fallback when diarization is off,
or when pyannote finds no speech at all. Any input format FFmpeg can decode is accepted.

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

### CUDA version coupling (maintenance note)

This is a **mixed-CUDA stack** on a `nvidia/cuda:13.0.1-runtime-ubuntu24.04` base:

- **torch / torchaudio / torchcodec → CUDA 13 (cu130).** Routed to the `pytorch-cu130`
  index in `pyproject.toml` (`[tool.uv.sources]`). `torchcodec` from PyPI is built for
  CUDA 13 and fails to load on a CUDA-12 stack with
  `libnvrtc.so.13: cannot open shared object file`, which breaks pyannote diarization
  (`NameError: name 'AudioDecoder' is not defined`). Pinning to cu130 keeps them on a
  matched build. Note `torchaudio` on cu130 currently caps at **2.11.0**, and torch
  shares its ABI, so both are pinned to the **2.11.x** train (torchcodec 0.14 needs torch ≥2.11).
- **ctranslate2 (the faster-whisper engine) → CUDA 12.** ctranslate2 (latest 4.8.x) has
  **no CUDA-13 build**; it dlopens `libcublas.so.12` and cuDNN 12. Since the cu130 torch
  wheels bundle cuBLAS **13** and the runtime image ships CUDA 13, nothing provides the
  `.so.12` it needs — so we install the CUDA-12 libs explicitly via the
  `nvidia-cublas-cu12` + `nvidia-cudnn-cu12` pip deps. The CUDA-13 driver runs CUDA-12
  code fine (backward compatible), and the two cuBLAS sonames (`.so.12` / `.so.13`)
  coexist in one process without conflict.

**LD_LIBRARY_PATH:** the ctranslate2 cu12 wheels have no RPATH, so the `Dockerfile`
adds their dirs (`.../nvidia/cublas/lib`, `.../nvidia/cudnn/lib`) to `LD_LIBRARY_PATH`.
Torchcodec's NPP (`libnppicc.so.13`) does **not** need this — the `-runtime-` base image
ships NPP on the system loader path (that's why we use `-runtime-` over the slim `-base-`,
and why no `nvidia-npp` dep is required). If you ever switch to a `-base-` image, you must
also re-add an `nvidia-npp` dep and its wheel dir to `LD_LIBRARY_PATH`.

**To bump CUDA further:** the ceiling is set by **ctranslate2** — until it ships a
CUDA-13 build, the whisper engine stays on the cu12 cuBLAS/cuDNN wheels regardless of the
base image. For the torch side, retarget the `pytorch-cu130` index name/URL and the
`[tool.uv.sources]` markers, bump the base image, `uv lock`, then verify the codec loads
(`uv run python -c "from pyannote.audio.core.io import AudioDecoder"`) and a GPU
transcription succeeds. Ensure the deploy host driver is new enough for the target CUDA.

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
