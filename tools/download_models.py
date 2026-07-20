"""Pre-download the default models into the Hugging Face cache.

Run at Docker build time so the image ships with the weights and the first request
needs no network round-trip. The whisper download goes through
``faster_whisper.download_model`` so the ``large-v2`` alias resolves to the right
CTranslate2 repo; pyannote is fetched by *instantiating* the pipeline, which pulls
its sub-models (segmentation + embedding) too, not just the top-level config.

``HF_TOKEN`` is required only for the gated pyannote repo — without it the whisper
model is still cached and pyannote is skipped (a warning, not an error), so a build
without the secret still succeeds.
"""

import os
import sys


def main() -> None:
    from faster_whisper import download_model

    model = os.getenv("DEFAULT_WHISPER_MODEL", "large-v2")
    print(f"Downloading Whisper model: {model}", flush=True)
    download_model(model)

    raw_token = os.getenv("HF_TOKEN")
    token = raw_token.strip() if raw_token else None
    if not token:
        print("HF_TOKEN not set; skipping pyannote pre-download", flush=True)
        return

    from pyannote.audio import Pipeline

    print("Downloading pyannote/speaker-diarization-community-1", flush=True)
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=token)
    if pipeline is None:
        print("Failed to instantiate pyannote pipeline during pre-download", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
