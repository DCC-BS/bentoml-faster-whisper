FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

COPY --from=ghcr.io/astral-sh/uv:0.6.6 /uv /uvx /bin/

RUN apt-get update && \
    apt-get install -y \
    ffmpeg \
    git

ADD . /app
WORKDIR /app

RUN uv sync

ENTRYPOINT ["uv", "run", "bentoml", "serve", "service:FasterWhisper", "-p", "50001"]