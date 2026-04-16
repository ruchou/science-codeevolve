"""Fork-local: format_prog_msg gains a PREVIOUS-ATTEMPT-FEEDBACK footer
when a program's legality reject, build_log, or verify_log is non-empty."""
import json

from codeevolve.database import Program
from codeevolve.prompt.sampler import format_prog_msg


def _fresh_program(**overrides) -> Program:
    """Build a Program with the minimal fields format_prog_msg requires.

    format_prog_msg asserts returncode is not None and reads
    language/code/eval_metrics/returncode/warning/error. L2 fields default to
    None and are supplied via overrides.
    """
    p = Program(
        id="test",
        code="int main(){return 0;}",
        language="cpp",
        returncode=0,
        output="",
        error="",
        warning="",
        eval_metrics={"fitness": 1.0},
    )
    for k, v in overrides.items():
        setattr(p, k, v)
    return p


def test_no_footer_on_successful_program():
    p = _fresh_program(
        legality_verdict="legal", legality_reasons=json.dumps([]),
        build_log="", verify_log="",
    )
    msg = format_prog_msg(prog=p)
    assert "PREVIOUS-ATTEMPT-FEEDBACK" not in msg


def test_legality_reject_footer():
    p = _fresh_program(
        legality_verdict="reject",
        legality_reasons=json.dumps(["removed reduction", "new write to read-only array"]),
        build_log="", verify_log="",
    )
    msg = format_prog_msg(prog=p)
    assert "# PREVIOUS-ATTEMPT-FEEDBACK" in msg
    assert "LEGALITY-REJECT:" in msg
    assert "removed reduction" in msg


def test_build_fail_footer_truncates():
    long_log = "x" * 10000
    p = _fresh_program(
        legality_verdict="legal", legality_reasons=json.dumps([]),
        build_log=long_log, verify_log="",
    )
    msg = format_prog_msg(prog=p)
    assert "BUILD-FAIL-LOG:" in msg
    footer = msg.split("# PREVIOUS-ATTEMPT-FEEDBACK", 1)[1]
    assert footer.count("x") <= 2048


def test_verify_fail_footer():
    p = _fresh_program(
        legality_verdict="legal", legality_reasons=json.dumps([]),
        build_log="", verify_log="output differs at line 5",
    )
    msg = format_prog_msg(prog=p)
    assert "VERIFY-FAIL-LOG:" in msg
    assert "output differs" in msg


def test_legality_reasons_limited_to_three():
    p = _fresh_program(
        legality_verdict="reject",
        legality_reasons=json.dumps([f"r{i}" for i in range(10)]),
        build_log="", verify_log="",
    )
    msg = format_prog_msg(prog=p)
    footer = msg.split("# PREVIOUS-ATTEMPT-FEEDBACK", 1)[1]
    assert "r0" in footer
    assert "r1" in footer
    assert "r2" in footer
    assert "r9" not in footer


def test_legality_reasons_non_json_fallback():
    """If legality_reasons is a plain string (not JSON), treat it as a single
    reason rather than crashing."""
    p = _fresh_program(
        legality_verdict="reject",
        legality_reasons="plain error message",
        build_log="", verify_log="",
    )
    msg = format_prog_msg(prog=p)
    assert "LEGALITY-REJECT:" in msg
    assert "plain error message" in msg


def test_all_three_sources_combined():
    p = _fresh_program(
        legality_verdict="reject",
        legality_reasons=json.dumps(["r1"]),
        build_log="build err", verify_log="verify err",
    )
    msg = format_prog_msg(prog=p)
    footer = msg.split("# PREVIOUS-ATTEMPT-FEEDBACK", 1)[1]
    assert "LEGALITY-REJECT:" in footer
    assert "BUILD-FAIL-LOG:" in footer
    assert "VERIFY-FAIL-LOG:" in footer
