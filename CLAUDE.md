# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A [BentoML](https://www.bentoml.com/) service that wraps [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2 Whisper) behind an OpenAI-compatible `/v1/audio/transcriptions` and `/v1/audio/translations` API, with optional [pyannote](https://github.com/pyannote/pyannote-audio) speaker diarization. Entry point: `service.py` (`FasterWhisper` class).

## Commands

- `make install` — `uv sync` + install pre-commit hooks.
- `make check` — verify `uv.lock` is in sync, run pre-commit over all files (ruff lint + format), then `uv run ty check .`.
- `make test` — fast unit tests only: `pytest -m "not integration and not model"`. No model weights are downloaded.
- `make test-model` — unit tests that load a real Whisper model (slow, GPU): `pytest -m "model and not integration"`.
- `make test-all` — every unit test, model-backed included.
- `make integration` — integration tests against a running service instance: `pytest -m integration`.
- Single test: `uv run --env-file .env python -m pytest tests/unit/test_file.py::test_name -q`.
- `make run` — serve on port 50001 via BentoML (`uv run --env-file .env bentoml serve service:FasterWhisper -p 50001`).
- `uv run python launch.py` — run the ASGI app directly under uvicorn on port 8004, for step-through debugging (BentoML's own runner obscures the debugger).
- `make docker-build` / `make docker-push` — build/push `quay.io/ktbs/fd-itbs/faster-whisper:v<version>` (version from `uv version --short`).
- `make docker-up` / `make docker-down` — run via `compose.yaml` (also brings up Prometheus + Grafana).

## Architecture

### Request flow

`service.py` defines the `FasterWhisper` BentoML service (ASGI-mounted FastAPI app at `/v1`) plus a separate `BatchFasterWhisper` service used only by the `/v1/audio/transcriptions/batch` route. Every transcription/translation route builds an `api_models.TranscriptionRequest` / `TranslationRequest` (pydantic, `extra="forbid"`) and delegates to `handlers/fast_whipser_handler.py::FasterWhisperHandler`, which owns a `WhisperModelManager` (`model_manager.py`) and a `DiarizationService` (`diarization_service.py`). Four route variants share this handler: sync (`transcribe`), batched, `bentoml.task` (async, with `progress_id` tracking), and streaming (NDJSON, one chunk per segment).

Note the intentional typos in existing filenames/identifiers — `handlers/fast_whipser_handler.py`, `helpers/whiper_diarization_merger.py`, and the `ResponseFormat.JSON_DIARZED` enum value (missing the second "I"). These are load-bearing (imported/serialized elsewhere); don't silently "fix" the spelling.

### Model lifecycle

`model_manager.py::WhisperModelManager` lazily loads Whisper models keyed by model id and wraps each in a `SelfDisposingModel`: ref-counted, and once the ref count hits zero it arms a `threading.Timer` to unload after `ttl` seconds (`ttl=-1` keeps it resident forever, `0` unloads immediately). Callers must hold the model via `with model_manager.load_model(name) as whisper:` for the full lifetime of anything consuming its lazy generators — `prepare_audio_segments` in the handler manually enters/exits this context manager because the returned segment generator outlives the calling method (see the `try/finally` and the `_held_segments()` closure there).

`DiarizationService` (pyannote) is loaded lazily on first `diarize()` call, independent of the Whisper model, so workers that never diarize never pin a pipeline to the GPU. It needs an `HF_TOKEN` env var for `pyannote/speaker-diarization-community-1`.

### Diarization doubles as VAD

When `request.diarization` is true, pyannote's speaker turns are used as the voice activity detector instead of Silero: `helpers/speech_regions.py` collapses the decoded audio down to just those speech intervals before handing it to `whisper.transcribe()` (silence never reaches the decoder), then maps segment/word timestamps back onto the original timeline and splits any segment whose words straddle two speech intervals. The collapsed speech is decoded as bounded runs, not one block: consecutive turns are grouped into runs no longer than `MAX_RUN_S` (`WHISPER_MAX_DECODE_RUN_S`, default 60s) of wall-clock span, split at the widest pause (`turns_to_language_runs`/`_split_turns_by_span`), and each run is a separate `whisper.transcribe()` call — a single very long block (continuous radio/panel speech collapses into intervals many minutes long) triggers Whisper's long-form seek drift and silently drops whole 30s windows. Silero (`vad_filter`) only runs as a fallback when diarization is off or pyannote finds no speech. `helpers/whiper_diarization_merger.py` then assigns a `speaker` to each word by finding the diarization turn with the largest time overlap, falling back to the nearest turn within `SPEECH_PAD_S` if there's no overlap at all (e.g. a word landed entirely in the turn's padding), and splits any segment whose words alternate speakers — Whisper's segmentation knows nothing about speakers, so in a fast exchange one segment can span several pyannote turns; every emitted segment carries exactly one speaker.

When diarization is on and no `language` is given, `handlers/fast_whipser_handler.py::_transcribe_language_runs` runs the turn-level language identification in `helpers/language_id.py`: batched per-turn Whisper LID (turns < `LID_MIN_TURN_S` inherit their merged interval's distribution), duration-weighted aggregation into a language inventory (relative share *or* absolute-seconds floor; `language_candidates` request param pins it), then Viterbi smoothing with a switch penalty. Consecutive same-language turns are decoded together as collapsed runs (`helpers/speech_regions.py::turns_to_language_runs`, padding clamped at run boundaries so adjacent different-language runs never decode the same audio twice; runs are further capped at `MAX_RUN_S` span, see the VAD note above) and each segment is tagged with its run's language. Both the single-language and multi-language paths share the run decode/restore loop in `handlers/fast_whipser_handler.py::_decode_language_runs`; the single-language path just passes the explicit language for every turn and leaves per-segment language unset. Tunables live in `config.py::LanguageIdConfig`, overridable via `LID_*` env vars (see README). The regression fixture for this path is `tests/unit/test_multilang_teams.py` against the gitignored internal asset `tests/assets/internal/teams_konferenz.mp3` (also records wall-clock/LID metrics as the perf reference).

### Post-decode cleanup

`helpers/transcription_cleaner.py::clean_transcription_segments` runs after decoding: drops segments that are both high `no_speech_prob` *and* low `avg_logprob` (either signal alone is unreliable), and drops segments matching `helpers/hallucinations.py`'s per-language blacklist of known Whisper training-data artifacts (Amara.org credits, etc).

### Response formats

`api_models/output_models.py::segments_to_response` / `segments_to_streaming_response` dispatch on `ResponseFormat` (`text`, `json`, `json_diarized`, `verbose_json`, `srt`, `vtt`) to the matching model in `api_models/`. `core.py` holds the internal domain types (`Segment`, `Word`, `Transcription`) that everything else is built from.

### CUDA version coupling

This is a mixed-CUDA-version stack (torch/torchaudio/torchcodec on cu130, ctranslate2 on cu12 libs aliased at build time) — see the README's "CUDA version coupling" section before touching `pyproject.toml`'s `[tool.uv.sources]`/`[[tool.uv.index]]`, the `Dockerfile`, or any torch/ctranslate2 version bump. Getting this wrong breaks either diarization (`torchcodec` `AudioDecoder`) or the Whisper decode itself (`libcublas.so.12`), not both, so a naive test pass on one path can miss a regression in the other.

## Testing

- `tests/unit/` vs `tests/integration/`, selected via the `integration`/`unit`/`model` pytest markers (`pytest.ini`); `make test`/`make test-model`/`make integration` compose these.
- `tests/conftest.py` provides session-scoped `model_manager`/`handler`/`faster_whisper_service` fixtures that share one loaded model across the whole suite (`ttl=-1`) — a per-test `ttl` would arm non-daemon `threading.Timer`s that block interpreter shutdown for 5 minutes.
- `tests/assets/` holds real audio fixtures (mp3/m4a/wav) used by both unit and integration tests, including a known-good transcript (`telefonat_transcript.json`) for regression comparison.
- Integration tests need a running service instance and, for model-backed ones, a GPU.

## Reference skill docs

`.agents/skills/` contains DCC-BS org-wide conventions (Python style, git/CI workflow, backend/UI patterns for other DCC projects). Treat them as background reference, not a spec for this repo — several conventions there (Returns-style functional error handling, Dependency Injector, structlog) are not used here; this codebase uses plain exceptions, direct construction, and `loguru`/stdlib `logging`.
