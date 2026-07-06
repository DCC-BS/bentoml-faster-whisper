FROM nvidia/cuda:13.0.1-runtime-ubuntu24.04

ENV TZ=Europe/Zurich
ENV LANG=de_CH.UTF-8

COPY --from=ghcr.io/astral-sh/uv:0.9.13 /uv /uvx /bin/

RUN apt-get update && \
    apt-get install -y \
    ffmpeg \
    git \
    && apt clean

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project

COPY . /app

RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen

# ctranslate2 (faster-whisper engine) is CUDA-12 and dlopens libcublas.so.12 / cuDNN 12
# with no RPATH to the pip wheels. Torchcodec's NPP (libnppicc.so.13) is already on the
# system loader path via the CUDA-13 runtime base image, so only the cu12 cuBLAS/cuDNN
# wheel dirs need exposing here.
ENV LD_LIBRARY_PATH=/app/.venv/lib/python3.13/site-packages/nvidia/cublas/lib:/app/.venv/lib/python3.13/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}

ENTRYPOINT ["uv", "run", "bentoml", "serve", "service:FasterWhisper", "-p", "50001"]
