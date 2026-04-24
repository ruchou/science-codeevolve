"""Regression tests: evaluate.py must be available inside temp_cwd sandboxes.

Phase D real-LLM EA children exited with ``returncode=2, evaluate.py not found``
because ``execute()`` only *remapped* ``eval_path`` into ``temp_cwd`` — it never
ensured the file actually existed there. If ``copytree(self.cwd, temp_cwd)``
did not materialize ``evaluate.py`` (because eval_path lived outside self.cwd,
or the file wasn't in self.cwd on disk), the subprocess died before computing
fitness.

These tests exercise the paths that are supposed to be hardened by the fix:
absolute eval_path inside cwd (copytree should copy it — sanity check), and
absolute eval_path *outside* cwd (copytree skips it; the fix must copy it in,
or leave the absolute-path launch viable and its target resolvable from the
subprocess's perspective).
"""
from pathlib import Path

from codeevolve.database import Program
from codeevolve.evaluator import Evaluator


EVAL_BODY = """\
import json, sys
with open(sys.argv[2], 'w') as f:
    json.dump({'combined_score': 1.0}, f)
"""

PROG_BODY = "# no-op program\n"


def _make_program() -> Program:
    return Program(id="test_prog", code=PROG_BODY, language="python")


def test_eval_script_inside_cwd_is_available_inside_temp_cwd(tmp_path: Path) -> None:
    """Absolute eval_path living inside cwd: copytree covers it.

    This is the happy path — it should work both before and after the fix.
    Guards against regression in the ``relative_to`` remap.
    """
    cwd = tmp_path / "src_root"
    cwd.mkdir()
    eval_script = cwd / "evaluate.py"
    eval_script.write_text(EVAL_BODY)

    ev = Evaluator(
        eval_path=eval_script,
        cwd=cwd,
        timeout_s=30,
        max_mem_b=None,
        resource_check_interval_s=None,
    )
    returncode, _, _, error, eval_metrics = ev.execute(_make_program())
    assert returncode == 0, f"expected clean run, got rc={returncode} err={error!r}"
    assert eval_metrics.get("combined_score") == 1.0


def test_eval_script_outside_cwd_is_propagated_into_temp_cwd(tmp_path: Path) -> None:
    """Absolute eval_path living *outside* cwd.

    ``copytree(self.cwd, temp_cwd)`` does NOT copy files outside self.cwd.
    Under the current contract, the Evaluator always propagates the eval
    script into temp_cwd (landing it at ``temp_cwd / <basename>`` when the
    original lives outside self.cwd), so the subprocess executes the
    sandbox copy rather than the out-of-tree original. The observable
    behavior: no FileNotFoundError, clean returncode, combined_score
    written.
    """
    cwd = tmp_path / "src_root"
    cwd.mkdir()
    # Program directory has no evaluate.py — eval script lives elsewhere.
    eval_dir = tmp_path / "evaluator_root"
    eval_dir.mkdir()
    eval_script = eval_dir / "evaluate.py"
    eval_script.write_text(EVAL_BODY)

    ev = Evaluator(
        eval_path=eval_script,
        cwd=cwd,
        timeout_s=30,
        max_mem_b=None,
        resource_check_interval_s=None,
    )
    returncode, _, _, error, eval_metrics = ev.execute(_make_program())
    assert error is None, f"expected no error, got {error!r}"
    assert returncode == 0, f"expected clean run, got rc={returncode} err={error!r}"
    assert eval_metrics.get("combined_score") == 1.0


def test_eval_script_relative_but_missing_from_cwd_raises_cleanly(tmp_path: Path) -> None:
    """Relative eval_path that does not exist inside cwd.

    Reproduces the Phase D failure mode: ``eval_path = Path("evaluate.py")``
    (relative, joined to ``cwd`` by convention), but the file was never
    written into ``cwd``. Before the fix, the subprocess launches
    ``python evaluate.py ...`` with cwd=temp_cwd, which exits with
    returncode=2 (no such file) and silently poisons the fitness pipeline.

    After the fix, the Evaluator must raise FileNotFoundError (surfaced via
    the outer EvaluationError wrapping in ``execute()``) BEFORE launching
    the subprocess, so the caller sees an explicit "Evaluator script not
    found" diagnostic rather than a cryptic runtime rc=2.
    """
    cwd = tmp_path / "src_root"
    cwd.mkdir()
    # Deliberately do NOT create cwd/evaluate.py.

    ev = Evaluator(
        eval_path=Path("evaluate.py"),
        cwd=cwd,
        timeout_s=30,
        max_mem_b=None,
        resource_check_interval_s=None,
    )
    returncode, _, _, error, eval_metrics = ev.execute(_make_program())

    assert returncode != 0, "missing eval script must not be a silent success"
    assert error is not None, "missing eval script must surface a diagnostic"
    # Require the explicit pre-launch FileNotFoundError signal. The bare rc=2
    # path produces "can't open file" instead — that's the bug this guards.
    assert "Evaluator script not found" in error, (
        f"expected pre-launch FileNotFoundError diagnostic, got: {error!r}"
    )


def test_eval_script_absolute_but_missing_raises_cleanly(tmp_path: Path) -> None:
    """Absolute eval_path that does not exist on disk.

    Symmetric with ``test_eval_script_relative_but_missing_from_cwd_raises_cleanly``
    but with an absolute path outside self.cwd. The Evaluator must raise
    FileNotFoundError (surfaced via EvaluationError) BEFORE launching the
    subprocess, rather than silently passing a bogus path to python.
    """
    cwd = tmp_path / "src_root"
    cwd.mkdir()
    # Deliberately do NOT create this file.
    missing_eval = tmp_path / "nowhere" / "evaluate.py"

    ev = Evaluator(
        eval_path=missing_eval,
        cwd=cwd,
        timeout_s=30,
        max_mem_b=None,
        resource_check_interval_s=None,
    )
    returncode, _, _, error, _eval_metrics = ev.execute(_make_program())

    assert returncode != 0, "missing eval script must not be a silent success"
    assert error is not None, "missing eval script must surface a diagnostic"
    assert "Evaluator script not found" in error, (
        f"expected pre-launch FileNotFoundError diagnostic, got: {error!r}"
    )
