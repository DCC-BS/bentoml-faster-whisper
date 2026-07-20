# Stage 1: Build
FROM nvidia/cuda:13.3.0-runtime-ubuntu24.04 AS build

COPY --from=ghcr.io/astral-sh/uv:0.9.13 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python

WORKDIR /app

# Dependency caching layer: only invalidated when the lockfile changes
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-install-project

# Pre-download the default models into the image so the first request needs no network.
# Placed before `COPY . /app` and bind-mounting only the script, so this heavy layer is
# cached independently of application source changes (re-runs only when the script or the
# deps change). The gated pyannote weights need HF_TOKEN as a BuildKit secret; without it
# the whisper model is still baked and pyannote is skipped (see tools/download_models.py).
ARG DEFAULT_WHISPER_MODEL=large-v2
RUN --mount=type=secret,id=hf_token \
    --mount=type=bind,source=tools/download_models.py,target=/tmp/download_models.py \
    HF_HOME=/opt/models \
    DEFAULT_WHISPER_MODEL="${DEFAULT_WHISPER_MODEL}" \
    HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" \
    /app/.venv/bin/python /tmp/download_models.py

COPY . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ctranslate2 wheels are CUDA-12 builds that dlopen libcublas.so.12. cuBLAS 13 (from
# the cu130 torch wheels) is ABI-compatible for ctranslate2's use, so alias it
# (OpenNMT/CTranslate2#1933).
RUN ln -s libcublas.so.13 /app/.venv/lib/python3.13/site-packages/nvidia/cu13/lib/libcublas.so.12 && \
    ln -s libcublasLt.so.13 /app/.venv/lib/python3.13/site-packages/nvidia/cu13/lib/libcublasLt.so.12

# Stage 2: Runtime
FROM nvidia/cuda:13.3.0-runtime-ubuntu24.04

ENV TZ=Europe/Zurich
ENV LANG=de_CH.UTF-8

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Replace the default ubuntu user (member of sudo group) with a locked-down app user
RUN userdel -r ubuntu && \
    groupadd --gid 1000 app && \
    useradd --uid 1000 --gid app --create-home --shell /usr/sbin/nologin app

# uv-managed Python interpreter; the venv's python symlinks resolve here
COPY --from=build /opt/uv/python /opt/uv/python
COPY --from=build --chown=app:app /app /app

# Baked model cache. Copied to the app user's default HF cache location so the runtime
# finds it with no env var. When compose mounts the (empty) hugging_face_cache named
# volume over this path, Docker seeds the volume from these contents on first start.
COPY --from=build --chown=app:app /opt/models /home/app/.cache/huggingface

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ctranslate2 (faster-whisper engine) dlopens libcublas.so.12 (aliased to cuBLAS 13
# in the build stage) and libcudnn.so.9 with no RPATH to the pip wheels. Torchcodec's
# NPP (libnppicc.so.13) is already on the system loader path via the CUDA-13 runtime
# base image, so only the cu13 cuBLAS/cuDNN wheel dirs need exposing here.
ENV LD_LIBRARY_PATH=/app/.venv/lib/python3.13/site-packages/nvidia/cu13/lib:/app/.venv/lib/python3.13/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}

USER app

ENTRYPOINT ["bentoml", "serve", "service:FasterWhisper", "-p", "50001"]
