"""Regression test: the tools/diagnose_ui.py dev tool must construct its
WhisperModelProvider / FasterWhisperHandler with the current (DI-refactored)
constructor signatures, so importing the module does not raise TypeError."""

import importlib.util
from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")  # dev-only dependency

_DIAGNOSE_UI = Path(__file__).resolve().parents[2] / "tools" / "diagnose_ui.py"


def test_diagnose_ui_module_imports():
    spec = importlib.util.spec_from_file_location("diagnose_ui", _DIAGNOSE_UI)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)  # would raise TypeError with the old signatures

    # The handler must be wired to a real diarization service and a model provider.
    assert module._model_manager.model_id
    assert module._handler.model_manager is module._model_manager
    assert module._handler.diarization is not None
