"""Gradio UI to compare raw pyannote diarization against the full transcription
pipeline: upload an audio file and inspect pyannote's speaker turns side by side
with the pipeline's final segments (language ID + transcription + speaker merge)
and the pipeline's detected top-level language.

Not part of the served image (gradio is a dev-only dependency, this script lives
under tools/, and pytest.ini's testpaths=tests keeps it out of the test suite).

Run with:
    uv run --env-file .env python tools/diagnose_ui.py
"""

import json
import sys
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import gradio as gr

from bentoml_faster_whisper.config import WhisperModelConfig, faster_whisper_config
from bentoml_faster_whisper.models.enums import Language, ResponseFormat
from bentoml_faster_whisper.models.transcription_request import TranscriptionRequest
from bentoml_faster_whisper.services.diarization_service import DiarizationSegment, DiarizationService
from bentoml_faster_whisper.services.faster_whisper_handler import FasterWhisperHandler
from bentoml_faster_whisper.services.model_manager import WhisperModelProvider

# One resident model provider for the life of the process: the model loads once and
# stays loaded between UI interactions instead of reloading per click.
_model_manager = WhisperModelProvider(WhisperModelConfig(), faster_whisper_config.default_model_name)
_handler = FasterWhisperHandler(model_manager=_model_manager, diarization=DiarizationService())

_LANGUAGE_CHOICES = ["auto (per-region detection)"] + sorted(lang.value for lang in Language)

_TURNS_HEADERS = ["start", "end", "duration_s", "speaker"]
_SEGMENTS_HEADERS = ["start", "end", "speaker", "language", "text"]


def _mmss(seconds: float) -> str:
    minutes, secs = divmod(max(seconds, 0.0), 60)
    return f"{int(minutes)}:{secs:06.3f}"


def _run_diagnosis(
    audio_path: str,
    language: str,
    num_speakers: float | None,
    model: str,
) -> tuple[list[list], list[list], str]:
    """Run the real pipeline once and capture the pyannote turns it used along the
    way, so the "raw" side of the comparison is exactly what fed the transcription
    — not a second, independently-run diarization pass that could disagree."""
    turns: list[DiarizationSegment] = []
    original_diarize = DiarizationService.diarize

    def capturing_diarize(self, *args, **kwargs):
        for segment in original_diarize(self, *args, **kwargs):
            turns.append(segment)
            yield segment

    # capturing_diarize has the same signature as DiarizationService.diarize but ty
    # can't see that through __get__; the instance-attribute shadowing is intentional.
    _handler.diarization.diarize = cast(Any, capturing_diarize.__get__(_handler.diarization, DiarizationService))
    try:
        request = TranscriptionRequest.model_validate(
            {
                "file": audio_path,
                "model": model,
                "language": None if language.startswith("auto") else language,
                "diarization": True,
                "diarization_speaker_count": int(num_speakers) if num_speakers else None,
                "response_format": ResponseFormat.VERBOSE_JSON,
                "timestamp_granularities": ["word"],
            }
        )
        raw_response = _handler.transcribe_audio(request)
        assert isinstance(raw_response, str)  # verbose_json serializes to a JSON string
        response = json.loads(raw_response)
    finally:
        del _handler.diarization.diarize  # restore the class method (instance attr shadowed it)

    turns.sort(key=lambda t: t.start)
    turns_rows = [[_mmss(t.start), _mmss(t.end), round(t.end - t.start, 2), t.speaker] for t in turns]
    segments_rows = [
        [_mmss(s["start"]), _mmss(s["end"]), s.get("speaker") or "-", s.get("language") or "-", s["text"]]
        for s in response["segments"]
    ]
    return turns_rows, segments_rows, response.get("language") or "-"


def diagnose(
    audio_path: str | None,
    language: str,
    num_speakers: float | None,
    model: str,
) -> tuple[list[list], list[list], str]:
    if not audio_path:
        return [], [], "no file uploaded"
    return _run_diagnosis(audio_path, language, num_speakers, model)


with gr.Blocks(title="FasterWhisper diagnosis") as demo:
    gr.Markdown(
        "# Diarization / multi-language pipeline diagnosis\n"
        "Upload an audio file to compare pyannote's raw speaker turns against the "
        "full transcription pipeline's segments and detected language."
    )
    audio = gr.Audio(type="filepath", label="Audio file")
    with gr.Row():
        language = gr.Dropdown(choices=_LANGUAGE_CHOICES, value=_LANGUAGE_CHOICES[0], label="Language")
        num_speakers = gr.Number(
            value=None, precision=0, minimum=1, maximum=6, label="Number of speakers (empty = let pyannote estimate)"
        )
    with gr.Accordion("Advanced", open=False):
        model = gr.Textbox(value=faster_whisper_config.default_model_name, label="Whisper model")

    run_button = gr.Button("Run diagnosis", variant="primary")
    top_language = gr.Textbox(label="Pipeline detected language", interactive=False)

    with gr.Row():
        turns_table = gr.Dataframe(headers=_TURNS_HEADERS, label="Raw pyannote turns")
        segments_table = gr.Dataframe(headers=_SEGMENTS_HEADERS, label="Pipeline segments")

    # gradio attaches .click() dynamically from Button.EVENTS at class-creation time;
    # ty's resolution of it is flaky across whole-project runs (sometimes sees it,
    # sometimes not — confirmed both a false "unresolved-attribute" and, with a
    # matching ty:ignore, a false "unused-ignore-comment" on otherwise-identical
    # runs). cast(Any, ...) sidesteps the flake entirely instead of chasing it.
    cast(Any, run_button).click(
        diagnose,
        inputs=[audio, language, num_speakers, model],
        outputs=[turns_table, segments_table, top_language],
    )


if __name__ == "__main__":
    demo.launch()
