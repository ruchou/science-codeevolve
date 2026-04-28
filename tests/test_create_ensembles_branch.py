# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#

import logging
from pathlib import Path

import pytest

from codeevolve.evolution import _create_ensembles
from codeevolve.lm.openai import OpenAIEnsemble
from codeevolve.lm.single_model_ensemble import SingleModelEnsemble

# Absolute path to the fake claude binary used for unit tests.
# Path math:
#   __file__ = memacc-llm/codeevolve/tests/test_create_ensembles_branch.py
#   parents[0] = tests/
#   parents[1] = codeevolve/
#   parents[2] = memacc-llm/   <-- we want this
_FAKE_CLAUDE = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "runner"
    / "fixtures"
    / "cc_ea_smoke"
    / "fake_claude_happy_a.sh"
)


def _minimal_args() -> dict:
    return {"api_key": "x", "api_base": "https://example.invalid"}


def _minimal_config_default_strategy() -> tuple[dict, dict]:
    config = {
        "ENSEMBLE": [{"model_name": "MOCK_TEST", "weight": 1.0}],
    }
    evolve_config = {
        # No EA_STRATEGY set => default behavior (OpenAIEnsemble path).
    }
    return config, evolve_config


def _minimal_config_claude_code_strategy() -> tuple[dict, dict]:
    config = {
        # ENSEMBLE is unused on the claude_code path but is still read
        # defensively; provide a minimal stub.
        "ENSEMBLE": [{"model_name": "MOCK_TEST", "weight": 1.0}],
    }
    evolve_config = {
        "EA_STRATEGY": "claude_code",
        "CLAUDE_CODE_BIN": str(_FAKE_CLAUDE),
        "CLAUDE_CODE_MAX_INVOCATIONS": 200,
        "CLAUDE_CODE_PER_INVOCATION_TIMEOUT_S": 600.0,
    }
    return config, evolve_config


def test_create_ensembles_default_returns_openai_ensemble():
    config, evolve_config = _minimal_config_default_strategy()
    explore, exploit = _create_ensembles(
        config, evolve_config, _minimal_args(), logging.getLogger("test")
    )
    assert isinstance(explore, OpenAIEnsemble)
    assert isinstance(exploit, OpenAIEnsemble)


def test_create_ensembles_claude_code_returns_single_model_ensemble():
    config, evolve_config = _minimal_config_claude_code_strategy()
    explore, exploit = _create_ensembles(
        config, evolve_config, _minimal_args(), logging.getLogger("test")
    )
    assert isinstance(explore, SingleModelEnsemble)
    assert isinstance(exploit, SingleModelEnsemble)
    # The wrapped model is a ClaudeCodeLM — verify by name attribute.
    assert explore.select_model().name == "claude_code"
    assert exploit.select_model().name == "claude_code"


def test_create_ensembles_claude_code_shares_single_lm_across_ensembles():
    """Spec §9: cost ceiling is per-run, not per-ensemble. Both
    SingleModelEnsembles must reference the same ClaudeCodeLM instance."""
    config, evolve_config = _minimal_config_claude_code_strategy()
    explore, exploit = _create_ensembles(
        config, evolve_config, _minimal_args(), logging.getLogger("test")
    )
    assert explore.select_model() is exploit.select_model()
