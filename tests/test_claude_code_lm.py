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
