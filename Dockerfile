FROM nvidia/cuda:12.8.0-base-ubuntu24.04

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

# torchcodec dlopens libnppicc.so.12 with no RPATH; expose the pip NPP wheel dir.
ENV LD_LIBRARY_PATH=/app/.venv/lib/python3.13/site-packages/nvidia/npp/lib:${LD_LIBRARY_PATH}

ENTRYPOINT ["uv", "run", "bentoml", "serve", "service:FasterWhisper", "-p", "50001"]
