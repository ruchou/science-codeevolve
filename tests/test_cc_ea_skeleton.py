"""All three new modules import cleanly. Smoke check before logic exists."""


def test_modules_import():
    import codeevolve.lm.claude_code  # noqa: F401
    import codeevolve.lm.single_model_ensemble  # noqa: F401


def test_runner_cc_ea_alert_imports():
    import sys
    from pathlib import Path
    # Add memacc-llm to path so `runner` package is importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import runner.cc_ea_alert  # noqa: F401
