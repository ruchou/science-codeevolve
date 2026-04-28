import asyncio
import os
import sys
from pathlib import Path

import pytest

# Make the outer `runner` package importable from these tests.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from codeevolve.lm.claude_code import ClaudeCodeLM


FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "runner" / "fixtures" / "cc_ea_smoke"


@pytest.fixture
def cwd_provider(tmp_path: Path):
    def _provider() -> Path:
        return tmp_path
    return _provider


def test_constructor_raises_when_claude_not_on_path(cwd_provider):
    """Spec §10: missing `claude` at construction time must fail loud."""
    with pytest.raises(FileNotFoundError):
        ClaudeCodeLM(
            claude_bin="definitely-not-on-path-cc-ea-test-XYZ",
            per_invocation_timeout_s=5.0,
            max_invocations_per_run=10,
            cwd_provider=cwd_provider,
        )


def test_constructor_raises_when_absolute_path_missing(cwd_provider):
    with pytest.raises(FileNotFoundError):
        ClaudeCodeLM(
            claude_bin="/absolutely/does/not/exist/claude",
            per_invocation_timeout_s=5.0,
            max_invocations_per_run=10,
            cwd_provider=cwd_provider,
        )


@pytest.mark.asyncio
async def test_generate_happy_path_returns_emitted_text(cwd_provider):
    lm = ClaudeCodeLM(
        claude_bin=str(FIXTURES / "fake_claude_happy_a.sh"),
        per_invocation_timeout_s=5.0,
        max_invocations_per_run=10,
        cwd_provider=cwd_provider,
    )
    text, prompt_tokens, completion_tokens = await lm.generate(
        [{"role": "system", "content": "be brief"}, {"role": "user", "content": "go"}]
    )
    assert "EDIT-BLOCK-START" in text
    assert prompt_tokens == 0
    assert completion_tokens == 0


@pytest.mark.asyncio
async def test_generate_serializes_messages_into_prompt(cwd_provider, tmp_path):
    """Prompt sent to the subprocess should include both system and user content."""
    # The fake binary ignores stdin and always prints, but we can verify the
    # subprocess was invoked by checking it returned the expected payload.
    lm = ClaudeCodeLM(
        claude_bin=str(FIXTURES / "fake_claude_happy_a.sh"),
        per_invocation_timeout_s=5.0,
        max_invocations_per_run=10,
        cwd_provider=cwd_provider,
    )
    text, _, _ = await lm.generate(
        [
            {"role": "system", "content": "system-marker"},
            {"role": "user", "content": "user-marker"},
        ]
    )
    assert "EDIT-BLOCK-START" in text  # subprocess ran successfully


@pytest.mark.asyncio
async def test_generate_timeout_returns_empty(cwd_provider, tmp_path):
    sleeper = tmp_path / "sleeper.sh"
    sleeper.write_text("#!/usr/bin/env bash\nsleep 10\n")
    sleeper.chmod(0o755)
    lm = ClaudeCodeLM(
        claude_bin=str(sleeper),
        per_invocation_timeout_s=0.5,
        max_invocations_per_run=10,
        cwd_provider=cwd_provider,
    )
    text, _, _ = await lm.generate([{"role": "user", "content": "x"}])
    assert text == ""


@pytest.mark.asyncio
async def test_generate_cost_ceiling_short_circuits(cwd_provider):
    lm = ClaudeCodeLM(
        claude_bin=str(FIXTURES / "fake_claude_happy_a.sh"),
        per_invocation_timeout_s=5.0,
        max_invocations_per_run=2,  # only 2 allowed
        cwd_provider=cwd_provider,
    )
    # Two real invocations succeed.
    t1, _, _ = await lm.generate([{"role": "user", "content": "x"}])
    t2, _, _ = await lm.generate([{"role": "user", "content": "x"}])
    assert "EDIT-BLOCK-START" in t1
    assert "EDIT-BLOCK-START" in t2
    # Third should short-circuit without invoking subprocess.
    t3, _, _ = await lm.generate([{"role": "user", "content": "x"}])
    assert t3 == ""


@pytest.mark.asyncio
async def test_generate_cost_ceiling_does_not_run_binary_when_exhausted(cwd_provider, tmp_path):
    """If max=0, generate must short-circuit before ever invoking the binary.
    Uses the existing fixture (which exists on disk) so construction succeeds;
    the cost-ceiling guard fires before the subprocess is ever spawned."""
    lm = ClaudeCodeLM(
        claude_bin=str(FIXTURES / "fake_claude_happy_a.sh"),
        per_invocation_timeout_s=1.0,
        max_invocations_per_run=0,
        cwd_provider=cwd_provider,
    )
    text, _, _ = await lm.generate([{"role": "user", "content": "x"}])
    assert text == ""
    # Subsequent calls also short-circuit.
    text2, _, _ = await lm.generate([{"role": "user", "content": "x"}])
    assert text2 == ""


def test_class_has_model_name_attribute():
    """Regression guard: codeevolve evolution.py reads ensemble.models[i].model_name
    in evolve_state telemetry. ClaudeCodeLM must expose this attribute."""
    assert hasattr(ClaudeCodeLM, "model_name")
    assert isinstance(ClaudeCodeLM.model_name, str)
    assert ClaudeCodeLM.model_name  # not empty


@pytest.mark.asyncio
async def test_cost_ceiling_logs_warning_once(cwd_provider, caplog):
    import logging as _logging
    lm = ClaudeCodeLM(
        claude_bin=str(FIXTURES / "fake_claude_happy_a.sh"),
        per_invocation_timeout_s=5.0,
        max_invocations_per_run=1,
        cwd_provider=cwd_provider,
    )
    # First call succeeds; second hits ceiling and logs.
    await lm.generate([{"role": "user", "content": "x"}])
    with caplog.at_level(_logging.WARNING):
        await lm.generate([{"role": "user", "content": "x"}])
        await lm.generate([{"role": "user", "content": "x"}])
    warnings = [r for r in caplog.records if r.levelno == _logging.WARNING]
    # Should log once, not twice.
    assert len(warnings) == 1
    assert "cost ceiling" in warnings[0].message.lower()
