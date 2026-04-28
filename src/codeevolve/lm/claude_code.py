"""ClaudeCodeLM — BaseLM that wraps `claude --print` subprocess invocations
for the --ea-strategy claude_code path (spec 2026-04-27)."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

# The runner package lives outside codeevolve; add memacc-llm/ to sys.path
# at import time. This is the same trick the rest of memacc-llm uses to
# bridge the codeevolve subpackage and the outer runner.
#
# Path math (resolved):
#   parents[0] = lm/
#   parents[1] = codeevolve/ (inner src package)
#   parents[2] = src/
#   parents[3] = codeevolve/ (submodule root)
#   parents[4] = memacc-llm/   <-- we want this for PYTHONPATH
#   parents[5] = MemAcc/       (repo root)
_MEMACC_LLM_ROOT = Path(__file__).resolve().parents[4]
if str(_MEMACC_LLM_ROOT) not in sys.path:
    sys.path.insert(0, str(_MEMACC_LLM_ROOT))

from runner.onboarding.cc_ea_smoke_runner import run_cc  # type: ignore[import-not-found]

from codeevolve.lm.base import BaseLM


def _serialize_messages(messages: List[Dict[str, str]]) -> str:
    """Concat chat-style messages into a single prompt string for `claude --print`.
    System content goes first, then alternating user/assistant turns delimited
    by --- separators. The exact delimiter does not matter for `claude --print`
    because it parses the entire stdin as one user prompt; this just keeps the
    serialized form readable in transcripts.
    """
    parts: List[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"--- {role} ---\n{content}")
    return "\n\n".join(parts)


class ClaudeCodeLM(BaseLM):
    """BaseLM implementation that runs `claude --print` per `generate(...)` call.

    Token counts are returned as 0 (`claude --print` does not expose them).
    Failure modes (timeout, subprocess error, cost ceiling) all return
    ("", 0, 0); the caller treats empty text as a generation failure, same
    path as a free-tier 429.
    """

    name: str = "claude_code"
    weight: float = 1.0  # only model in the pool, weight ignored

    def __init__(
        self,
        *,
        claude_bin: str = "claude",
        permission_mode: str = "bypassPermissions",
        per_invocation_timeout_s: float = 600.0,
        max_invocations_per_run: int = 200,
        cwd_provider: Callable[[], Path],
    ):
        # Spec §10: raise immediately at construction if `claude` is not
        # resolvable. Catches the common config error of enabling
        # --ea-strategy claude_code on a machine without claude installed.
        import shutil
        if not Path(claude_bin).is_absolute() and shutil.which(claude_bin) is None:
            raise FileNotFoundError(
                f"--ea-strategy claude_code requires `claude` CLI on PATH; "
                f"got {claude_bin!r}. Install Claude Code or pass an "
                "absolute path via --claude-code-bin."
            )
        if Path(claude_bin).is_absolute() and not Path(claude_bin).exists():
            raise FileNotFoundError(
                f"--ea-strategy claude_code: --claude-code-bin path "
                f"{claude_bin!r} does not exist."
            )
        self._claude_bin = claude_bin
        self._permission_mode = permission_mode
        self._per_invocation_timeout_s = per_invocation_timeout_s
        self._max_invocations_per_run = max_invocations_per_run
        self._cwd_provider = cwd_provider
        self._invocations_used = 0

    async def generate(
        self, messages: List[Dict[str, str]]
    ) -> Tuple[str, int, int]:
        prompt = _serialize_messages(messages)
        cwd = self._cwd_provider()
        # `run_cc` is synchronous (subprocess.run); offload to a thread so we
        # don't block the asyncio loop while `claude` is computing.
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: run_cc(
                prompt=prompt,
                cwd=cwd,
                claude_bin=self._claude_bin,
                turn_budget=0,  # ignored when turn_budget_flag is None
                timeout_s=self._per_invocation_timeout_s,
                permission_mode=self._permission_mode,
                turn_budget_flag=None,
            ),
        )
        self._invocations_used += 1
        if result.status == "emitted":
            return result.stdout, 0, 0
        return "", 0, 0
