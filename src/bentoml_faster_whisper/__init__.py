"""bentoml-faster-whisper package."""

from dotenv import load_dotenv

# Load .env before any submodule reads the environment at import time. config.py builds
# faster_whisper_config / language_id_config from os.getenv while it is imported, so
# DEFAULT_WHISPER_MODEL and the LID_* tunables must be in the environment by then.
# Doing it here (the package's first executed code) guarantees that. In production
# varlock/--env-file already populate the environment; load_dotenv does not override
# existing values, so it is a no-op there.
load_dotenv()
