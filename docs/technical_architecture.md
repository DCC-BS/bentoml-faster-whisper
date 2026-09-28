# Technical Architecture & Implementation Notes

This document captures the architectural decisions, performance optimizations, and algorithmic design choices for `bentoml-faster-whisper`. It is the canonical home for the "why" behind non-obvious code — kept here rather than inline so the source stays lean.

---

## 1. GPU Concurrency & Resource Management

### Multi-Worker Parallel Decoding (`WHISPER_NUM_WORKERS`)
- **Problem**: Autoregressive Whisper decoding (`batch_size=1`, one token at a time) is latency-bound and leaves GPU compute units underutilized.
- **Solution**: CTranslate2 parallel worker replicas share the resident model weights in VRAM. The diarized path cuts the file into many bounded decode runs (~60s each); with `>1` worker they decode concurrently instead of one-at-a-time.
- **Performance Impact**: On a 25-minute diarized file, total wall-time dropped from ~84.8s to ~51.6s (~1.8x on the decode phase). Output is byte-identical — it is the *same* sequential decode per run, just parallelised (unlike window-batching, which loses per-window context). Decode is ~71% of pre-optimization wall-time (~52s); pyannote is ~20s and already batches internally.
- **Configuration**: `WHISPER_NUM_WORKERS` (default `4`). `4` saturates a standard accelerator (`8` gives no further gain); a bigger GPU (e.g. H100) can go higher. Weights are shared across workers, so the extra VRAM is modest.
- **Memory Bounding**: Runs are dispatched via `executor.map`, which preserves run order and only schedules ~`concurrency` runs ahead. At most that many runs' audio/mel buffers are resident at once — memory stays bounded by concurrency, not by file length. Runs cover non-overlapping spans in timeline order, so concatenating their segments in run order is already chronological.

### Batched Inference (Rejected)
- `faster_whisper.BatchedInferencePipeline` was evaluated and rejected: it regresses WER on hard German audio (its internal VAD drops speech and clips lose cross-window context). Concurrent per-run decode delivers the throughput win with no quality cost.

### Startup Warmup (`WARMUP_ON_STARTUP`)
- **Default `true`**: each worker loads the resident Whisper model and pyannote pipeline into VRAM at startup, then runs one throwaway decode on 1s of silence to compile CUDA kernels — so the first real request doesn't pay load + compile latency.
- Set `false` to keep lazy first-request loading (e.g. token-less dev without cached weights).
- `DiarizationService` loads its pyannote pipeline lazily on first `diarize()`, so a worker that never diarizes doesn't pin a pipeline to the GPU.

### Pyannote Diarization Batching & VRAM Staggering
- **Batch Size**: `segmentation_batch_size`/`embedding_batch_size` = `32` (the pyannote `speaker-diarization-community-1` default). The embedding step is ~70% of diarization time; `32` is the measured sweet spot, and lowering it only slows the run with no VRAM relief inside our budget. Kept overridable for tight GPUs.
- **Silent-bug fix**: batch size must be set on the `Pipeline` instance directly. The previous code set it on `pipeline._models` — an attribute pyannote never reads — so it silently did nothing.
- **Thread-safety**: a pyannote pipeline is not thread-safe. A lock guards both lazy loading and inference.
- **Pipeline Staggering**: diarization runs *before* Whisper, so pyannote's peak GPU allocation doesn't overlap Whisper's decode allocation. Each run's activation blocks are returned to the PyTorch allocator immediately after completion.

---

## 2. Audio Ingestion, VAD & Seeking Strategy

### Audio Normalization & Error Mapping
- Files already 16 kHz mono WAV go straight through (`_is_16k_mono_wav`); everything else is converted to WAV first. MP3/FLAC headers report duration imprecisely, so conversion gives pyannote an exact sample count without loading the whole file into RAM.
- A failed audio decode (PyAV / faster-whisper on a malformed or unsupported upload) is a client error, mapped to `InvalidArgument` → HTTP 400, not a 500.

### Whisper Seek Drift & Run Splitting (`WHISPER_MAX_DECODE_RUN_S`)
- **Problem**: Continuous speech (radio, panel discussions) collapses into intervals many minutes long. Handed to a single `whisper.transcribe()`, its long-form seek mechanism drifts and skips whole 30s windows, which are never emitted.
- **Solution**: Continuous spans are partitioned into runs no longer than `WHISPER_MAX_DECODE_RUN_S` (default `60.0`s ≈ two 30s windows — enough decode context for quality, short enough that drift doesn't accumulate; drift was measured to reappear around ~90s). The single-language and per-turn-language paths share the same run machinery.
- **Split point**: each cut falls on the *widest* silence gap between consecutive turns (ties break toward the centre for balanced recursion, not peeling one turn at a time), always on a turn boundary so no word is cut. A greedy first-fit split can land a boundary mid-sentence, where the next run's decode drifts and drops its opening words. Gaps are measured against the furthest end reached so far (not the previous turn's end) so overlapping turns still yield a real non-negative gap.

### Run-Tail Hallucinations (`WHISPER_RUN_TAIL_SLIVER_S`, `WHISPER_RUN_TAIL_ANOMALY_S`)
- **Problem**: every decode run ends with a little non-speech (the `SPEECH_PAD_S` pad plus up to half the gap to the next run). With word timestamps on (always, on the diarized path), faster-whisper moves `seek` to the last word's end, so that sub-second tail is decoded as a window of its own, zero-padded to 30s. Whisper fills it with YouTube outros ("Das war's für heute.", "Bis zum nächsten Mal.", "Tschüss.", "Vielen Dank fürs Zuschauen."). They score `no_speech_prob` ~0.6 and `avg_logprob` ~-0.7, so neither faster-whisper's skip rule nor the cleaner's dual condition catches them. Run splitting multiplied the number of run ends, so it multiplied these hallucinations.
- **Rule** (`drop_run_tail_hallucinations`, applied per run before timestamps are restored): drop a whole decode window (all segments sharing its `seek`) when it starts less than `WHISPER_RUN_TAIL_SLIVER_S` (1.0s) before the run end, or less than `WHISPER_RUN_TAIL_ANOMALY_S` (2.0s) before it and any of its segments is word-anomalous (faster-whisper's `hallucination_silence_threshold` heuristic: words near-zero length, over 2s, or with probability below 0.15). The decision is per window, because a window's normal-looking segments ("Ciao.") are hallucinated just like the anomalous ones next to them.
- **Why only the run tail**: a real closing word decoded inside a normal speech window is never touched (Telefonat "Tschüss!"), and anomalous-looking real speech in the run body is kept (a serial number read out slowly has overlong words).
- **Rejected alternatives** (measured on the 33-file German eval, the test assets and 47 hand-labelled segments in `tests/unit/test_reviewed_hallucinations.py`):
  - passing `hallucination_silence_threshold` to faster-whisper: it changes seeking for the whole decode, rewrites text, drops a real sentence at 1.0s and still leaves 6 of 14 labelled hallucinations at 2.0s;
  - stricter `no_speech_prob`/`avg_logprob` thresholds: they drop real speech, since real windows can score `no_speech_prob > 0.9`;
  - an anomaly filter without the tail restriction: it drops long real segments;
  - trimming the run-end pad: it changes the run layout and rewrites ~500 words;
  - dropping segments without diarization overlap: most outros sit inside the padded turn edge, so it misses them.
- **Result**: WER 0.4535 → 0.4516, CER 0.1549 → 0.1533, all eval outros removed, no reference word lost.
- **Known miss**: a low-confidence hallucination at a run end whose words look normal ("So it's okay.") has the same profile as real low-confidence run-end speech ("au moins."), so it is kept rather than risk dropping real speech.

### Pre-Cutting Audio vs `clip_timestamps`
- **Mechanism**: audio is cut down to pyannote/VAD speech turns *before* decoding — the same thing faster-whisper does internally for its silero VAD. Silence never reaches the decoder; `restore_and_split_segments()` maps results back onto the original timeline afterward, snapping boundaries to the speech regions so the merge lines up.
- **Rationale**: passing the regions as `clip_timestamps` is not equivalent — each clip tail gets zero-padded to a full 30s window and the model's timestamp tokens drift there, unclamped.
- **Deferred concatenate**: the speech check only verifies decodable speech *exists*; the full-file concatenate is deferred to the per-run decode paths (which re-collapse per run) and the rare LID fallback, so a long meeting never pays a whole-file `np.concatenate` nothing downstream consumes.
- **Fallback**: Silero VAD is used only when diarization is disabled or finds no speech; otherwise no speech regions would mean decoding the entire file.

---

## 3. Speaker Diarization & Segment Alignment Algorithm

### Alignment & Timestamp Jitter
- Word timestamps jitter by ~100–300ms around pyannote turn boundaries.
- A word only justifies cutting a segment when at least `_SPLIT_MIN_OVERLAP_FRACTION` of its duration lies inside its assigned turn. A border word can flip to the neighbouring turn under raw max-overlap; an unconfident border word keeps its speaker label but never *starts* a new segment.
- Words near borders with no overlap snap to the nearest turn within `SPEECH_PAD_S` (0.3s).

### Sub-Threshold Fragments
- Where pyannote emits 0.1–0.2s turn fragments (crosstalk, simultaneous greetings), word-level speaker changes are noise. A sub-threshold group folds into its neighbour (previous when possible), then neighbours that ended up with the same speaker are re-merged — preventing one-word segments.

### Boundary Splitting
- Segments are cut only at *confident* speaker changes, so a fast exchange decoded as one window keeps its alternation and segment-level speaker labels strictly match word-level labels. With no credible change, the segment stays whole and is labelled by word-duration majority (matching the pre-split behaviour).

---

## 4. Multilingual Transcription & Language Identification

### Batched Turn-Level LID (`utils/language_id.py`, `LID_*`)
- **Path**: active when diarization is enabled and no explicit `language` is given. Each diarization turn gets its own language decision instead of one language for the whole file.
- **Batched encoding**: mel windows of all turns are encoded in batches (one encoder pass per batch, not per interval). Turns longer than one 30s window use the duration-weighted average of all their window distributions, so one ambiguous stretch can't pin the turn.
- **Short-turn skip**: turns shorter than `min_turn_s` (default `1.0`s) return no distribution — Whisper LID is unreliable on very short clips; they are resolved from context by smoothing.
- **Joint resolution**: per-turn decisions are resolved over the whole file (language inventory + Viterbi smoothing with a `switch_penalty`) rather than taking each top-1 in isolation.
- **Configuration**: tunables live in `config.LanguageIdConfig`, overridable per deployment via `LID_*` env vars (e.g. `LID_SWITCH_PENALTY=3.0`). Invalid values fail at import, not silently.

### Language Reporting
- The top-level reported language is the one covering the most speech time summed across *all* runs (majority), not the language of whichever run decoded first.
- With an explicit `language`, every run decodes in it and segments keep `language=None` — per-segment language is only reported when it was auto-detected per region.
- When no turn is long enough to detect on, all speech is collapsed once and detected a single time, like the single-language path.


### Known Limitation: Short Turns in the Wrong Language
- A turn of about 1-4s can still be decoded in the wrong language of a multilingual file (teams_konferenz.mp4 at 1257s: a German question decoded as French; lichtenstein.mp3 at 14-22s: a Spanish sentence decoded as German). Pinned by `tests/unit/test_turn_language_regressions.py` as strict xfail.
- **Cause**: Whisper's own language ID on short clips, before any smoothing (the Spanish half-sentence scores de 0.64 with es outside the top 3).
- **Why not tuned away** (simulated on the real per-turn distributions, 2026-09-23): `LID_SWITCH_PENALTY=3.0` fixes the German question but flips a confirmed-real Spanish turn to English; `LID_MIN_TURN_S=1.5` fixes it but flips a confirmed-real French "D'accord." to German and changes ~20 unverifiable turns per file. `language_candidates` removes stray English/Dutch decodes but cannot choose between the meeting's own languages. None of these recovers the Spanish sentence.
- **Mitigation for clients**: pass `language` for single-language audio (skips per-turn detection entirely, e.g. Swiss German recordings otherwise partly detected as Dutch), or `language_candidates` when the language mix is known.
---

## 5. Quality & Hallucination Mitigation

### Compute Type / Quantization
- **Default**: `int8_float16` on CUDA (`default` on CPU — it is a CUDA compute type).
- **Rationale**: ~50% less VRAM and faster decoder token generation, quality-neutral on the curated German eval (WER 0.460 → 0.455, CER 0.159 → 0.157 — no regression). A resource/speed win, not a quality win. `beam_size=1` was evaluated in the same sweep and rejected (WER regression).

### Conditioning on Previous Text
- **Default**: `condition_on_previous_text = False` (per-request overridable).
- **Rationale**: on the curated German eval this improved every metric (WER 0.460 → 0.453, CER 0.159 → 0.154, BLEU 45.7 → 46.1) by avoiding the hallucination cascades that previous-text conditioning can trigger across 30s window boundaries.

### Dual-Condition Silence Filtering
- **Problem**: `no_speech_prob` alone is unreliable — whole windows of confidently decoded speech can score `> 0.9` (`head60.mp3` lost its first 23s this way, with `avg_logprob ~ -0.3`).
- **Rule**: mirroring the Whisper reference rule, a segment is treated as silence only when `no_speech_prob` fires AND the decode itself was unconfident (`avg_logprob` low).

### Hallucination Blacklists
- Blacklists are applied by the *output* language of each segment. For a multilingual file, `transcription_info.language` is the majority language; a segment decoded in another language is matched against that language's blacklist. When the task rewrites text into a fixed language (translation always emits English), the blacklist is pinned to that output language, not the detected source.

### Text/Word Normalization
- Per-word text is kept in sync with `segment.text` (e.g. ß → ss). `verbose_json` and diarized responses expose `segment.words`, so leaving an un-normalized word would contradict the same span's `segment.text`.

---

## 6. Configuration & Dependency Injection
- Two sub-configs (`faster_whisper`, `language_id`) are aliased at module import time to build pydantic model schemas — `Field` defaults and `Annotated` constraints in `models/*` and the LID util are frozen at class-definition time and can't be DI-injected.
- Runtime consumers (services, container) instead take config via the DI container / `get_config()`.
- `.env` is loaded in the package `__init__` *before* `config.py` reads the environment at import time, so `os.getenv` already reflects it. Per-field `default_factory` declarations (including `LanguageIdConfig.from_env`) mean `cls()` reads the environment fully.
- Caveat: endpoint `examples=[...]` are evaluated at class-definition time and read the import-time config global, not the injected `self.config` used by runtime checks.

---

## 7. API Design & Conformance

### Async Event Loop vs Threadpool Responsiveness
- High-frequency operational endpoints — progress polling (`/v1/audio/transcriptions/progress`) and model metadata (`/v1/models`) — are `async` handlers. They do no blocking work (progress reads a dict under a microsecond lock, no I/O, no `await`), so they answer directly on the asyncio event loop instead of queueing in Starlette's threadpool behind heavy GPU jobs.

### Response Content-Type Override
- The endpoints declare a single `application/json` return type, so `text`/`vtt`/`srt` bodies would be served mislabelled as JSON. The correct `Content-Type` is set via `ctx.response` headers, which the server applies on top of the serializer's type. `ctx` is injected per request and is `None` only for direct in-process calls (unit tests), where there is no HTTP response to label.

### Swagger / Route Cleanup
- Auto-generated task sub-routes (cancel, retry) are dropped from the OpenAPI/Swagger docs. Cancel is answered with `400 "task cancellation is not supported"` and never interrupts a running decode; retry is unused.

### OpenAI API Spec Compliance
- `verbose_json` accepts requests without explicit `timestamp_granularities` (defaulting to segment-level timestamps), preventing client errors with standard OpenAI SDKs.

---

## 8. Observability & Progress

### Unified Progress Bar
- A single monotonic `0..1` fraction folds the two GPU stages so the bar never resets: diarization occupies `0..0.3`, transcription `0.3..1.0`. Within diarization, pyannote steps get ordered weights — `embeddings` (the heavy GPU step reporting granular completed/total) gets the bulk of the band.
- The progress entry is registered *before* diarization (which runs eagerly during audio prep), so the UI sees a tracked task with live diarization progress instead of a missing/0 entry until decode. It is always removed on completion or error (the eager prep that can raise lives inside the same `try`).
- A degenerate/empty container (duration 0) holds the bar at the post-diarization floor instead of dividing by zero.
- The progress store is per-instance and lock-guarded (a class-level dict would be shared/mutated across concurrent requests). Entries are re-inserted on update to keep the dict ordered by last update — a plain key assignment keeps the original position, which would evict a long-running task that is actively reporting ahead of newer but idle entries.

### Prometheus Multiprocess Metrics
- Metrics are exposed through lazy accessors (`utils/metrics.py`), not module-level objects. Importing `prometheus_client` at module-import time (before `PROMETHEUS_MULTIPROC_DIR` is set) silently drops custom metrics under the multiprocess worker model; deferring registry construction keeps them visible. The shared label conventions and divide-by-zero guard live in these helpers rather than being reconstructed at each call site.
