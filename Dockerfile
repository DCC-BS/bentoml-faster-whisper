FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV TZ=Europe/Zurich
ENV LANG=de_CH.UTF-8

COPY --from=ghcr.io/astral-sh/uv:0.8.0 /uv /uvx /bin/

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

ENTRYPOINT ["uv", "run", "bentoml", "serve", "service:FasterWhisper", "-p", "50001"]