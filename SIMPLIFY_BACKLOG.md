# Simplification / cleanup backlog

Findings surfaced by the whole-codebase `/simplify` pass that were **not** applied,
kept here to circle back to. Each notes why it was skipped and what fixing it
involves. Line numbers are approximate — verify before acting.

## 1. `translate_audio` discards its validated request (likely a bug, not cleanup)

- **Where:** `service.py` (~L186-190) → `handlers/fast_whipser_handler.py::translate_audio` (~L107-159)
- **What:** The `/v1/audio/translations` route builds `request = TranslationRequest.from_dict(params)`
  and calls `self._configure_vad_options(request)` (which mutates `request.vad_parameters`),
  then invokes `self.handler.translate_audio(**params)` — passing the **raw** `params` dict,
  not `request`. So the validated/configured model is dead on arrival and the VAD mutation is
  silently discarded. `translate_audio` then re-declares all 18 decode params as a flat
  positional signature and rebuilds the `whisper.transcribe()` kwargs by hand.
- **Why skipped:** This is a correctness concern, not a simplification — out of scope for
  `/simplify`, and refactoring it changes behavior. Run `/code-review` on it first.
- **Deeper fix:** `translate_audio(self, request: TranslationRequest)` mirroring
  `transcribe_audio(request)`, reusing a shared decode-options builder (the transcribe path
  already has `_decode_options`). Would also kill the 18-arg signature duplication.

## 2. Eager full-file `collapse_decoded_to_speech` that the run paths don't consume

- **Where:** `handlers/fast_whipser_handler.py` (~L208)
- **What:** Every diarized request runs `collapsed = collapse_decoded_to_speech(decoded, intervals)`,
  a `np.concatenate` over all speech chunks of the whole file (100MB+ for a long meeting). But
  both downstream decode paths re-collapse per run; the full-file `collapsed[0]` is consumed only
  in the rare LID fallback (`whisper.detect_language(audio=collapsed[0])`, taken only when no turn
  in the entire file is long enough to language-ID).
- **Why skipped:** Behavior-preserving but needs restructuring the fallback branch; medium value,
  non-trivial risk.
- **Cheaper form:** At the collapse site only compute whether any speech chunk exists; defer the
  concatenate into the LID-fallback branch (pass `intervals` down and collapse there).

## 3. Source audio decoded to 16 kHz mono twice per diarized request

- **Where:** `diarization_service.py` `_as_wav` (~L97) + `handlers/fast_whipser_handler.py` (~L206)
- **What:** `_as_wav` runs ffmpeg to produce a 16 kHz mono WAV for pyannote, then the handler
  independently calls `decode_audio(str(request.file), 16000)` on the original file again. Two full
  decodes of the same audio. The ffmpeg WAV is already what `decode_audio` would produce, but it's
  unlinked inside `diarize`'s context manager before the handler decodes.
- **Why skipped:** Cross-library, requires lifetime restructuring (keep the temp WAV alive and decode
  from it). Invasive.

## 4. `validate_timestamp_granularities` runs at two layers

- **Where:** `service.py::_prepare_transcribe` (all four transcribe routes) **and**
  `handlers/fast_whipser_handler.py::transcribe_audio` (~L93).
- **What:** Called at both the service boundary and inside the handler; only the sync route hits the
  handler-level call, so it's asymmetric across routes.
- **Why skipped:** Not pure redundancy — the handler-level call also guards direct
  `handler.transcribe_audio(...)` use in tests. Removing it would drop validation there. Decide
  whether request validation should live only at the service boundary before changing.

## 5. `DiarizationSegment.label` is redundant state

- **Where:** `diarization_service.py` (~L68-82, constructed ~L221)
- **What:** Every instance is `DiarizationSegment(turn, speaker, speaker)`, so `label` always equals
  `speaker`; it's read only in `__str__`.
- **Why skipped:** Low value; dropping it touches `__str__` and one test assertion.

## 6. Enum-coercion validators share a structure (medium)

- **Where:** `api_models/TranscriptionRequest.py::_process_empty_language` (~L20) and
  `api_models/input_models.py::_process_empty_response_format` (~L67).
- **What:** Same 4-step `BeforeValidator`: decode-bytes-or-warn → empty/None→default →
  already-an-instance→passthrough → `try Enum(value) except ValueError: warn+default`. Only the enum
  type and default differ.
- **Why skipped:** Consolidatable into a generic "coerce enum with logged fallback" validator factory,
  but more involved. Note: `_process_language_candidates` deliberately **raises** instead of falling
  back — leave that one out of any shared helper.

## 7. `segments_to_response` / `segments_to_streaming_response` repeat the format dispatch (low)

- **Where:** `api_models/output_models.py` (~L58 and ~L80).
- **What:** Both carry the full `ResponseFormat` if/elif dispatch incl. the identical
  "Unknown response format" tail. They legitimately differ (aggregate vs per-segment), so only the
  per-format payload builder is shareable — a single `_render_segment(segment, i, fmt, info)` helper.

## 8. Trivial

- `helpers/logger.py:112` — pre-existing ruff **E741** (ambiguous variable name `l`). Unrelated to the
  simplify pass; fix opportunistically.
- `diarization_service.py` (~L60) re-implements `helpers/speech_regions.py::_clamp` inline as
  `min(max(within, 0.0), 1.0)`. Would need `_clamp` made public. Trivial.
