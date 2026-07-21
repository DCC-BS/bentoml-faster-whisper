"""Regression test: values that config.py reads from the environment at import
time (DEFAULT_WHISPER_MODEL, LID_*) must respect a .env file. python-dotenv has
to run before the config module is imported, otherwise those reads are frozen to
their defaults while later os.getenv reads in service.py still pick up .env —
an inconsistent split.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[2] / "src")


def test_dotenv_applies_to_import_time_config(tmp_path):
    (tmp_path / ".env").write_text("DEFAULT_WHISPER_MODEL=regression-test-model\n")

    script = textwrap.dedent(
        """
        import bentoml_faster_whisper  # noqa: F401  (triggers package __init__ / dotenv)
        from bentoml_faster_whisper.config import get_config
        print(get_config().faster_whisper.default_model_name)
        """
    )

    env = {"PYTHONPATH": _SRC, "PATH": "/usr/bin:/bin"}
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "regression-test-model", result.stdout
