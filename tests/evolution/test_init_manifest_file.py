# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#

"""Regression test for Plan E Task 2 (MemAcc-LLM) INIT_MANIFEST_FILE override.

The override block in `evolution._initialize_new_run` must leave
`init_sol.code` as the pristine baseline source and override only
`.m1b_manifest` and `.model_msg`. An earlier version overwrote `.code`
with the manifest envelope, which broke (a) `format_prog_msg` (envelope
shown where source should be) and (b) `apply_diff` (SEARCH/REPLACE
targets the envelope text instead of source). The evaluator picks up
`.m1b_manifest` and applies it against `.code` via `pipeline._closure`,
so leaving `.code` intact is the correct contract.

This test exercises the override logic in isolation. Driving
`_initialize_new_run` end-to-end requires a full evaluator stub; the
override block itself is a 3-line read-and-assign against an already
constructed `Program`, so we reproduce it faithfully here.
"""

from pathlib import Path
from uuid import uuid4

from codeevolve.database import Program


def _apply_init_manifest_override(init_sol: Program, config: dict) -> Program:
    """Reproduces the INIT_MANIFEST_FILE override block from
    `evolution._initialize_new_run`.

    Kept in sync with `src/codeevolve/evolution.py` (around lines
    1428-1437). If the production block changes, update this too.
    """
    init_manifest_path = config.get("INIT_MANIFEST_FILE")
    if init_manifest_path:
        seed_text = Path(init_manifest_path).read_text()
        init_sol.m1b_manifest = seed_text
        init_sol.model_msg = seed_text
    return init_sol


def _make_init_sol(pristine_code: str) -> Program:
    return Program(
        id=str(uuid4()),
        code=pristine_code,
        language="cpp",
        iteration_found=0,
        generation=0,
        island_found=0,
        depth=0,
    )


def test_init_manifest_leaves_code_pristine(tmp_path):
    """When INIT_MANIFEST_FILE is set, .code must remain the pristine
    baseline; only .m1b_manifest and .model_msg are overridden."""
    pristine = "int main() { return 0; }\n"
    seed_text = (
        "# M1b Manifest\n"
        "file: src/main.cpp\n"
        "<<<<<<< SEARCH\n"
        "int main() { return 0; }\n"
        "=======\n"
        "int main() { /* optimized */ return 0; }\n"
        ">>>>>>> REPLACE\n"
    )
    manifest_file = tmp_path / "seed.md"
    manifest_file.write_text(seed_text)

    init_sol = _make_init_sol(pristine)
    config = {"INIT_MANIFEST_FILE": str(manifest_file)}

    _apply_init_manifest_override(init_sol, config)

    assert init_sol.code == pristine, (
        ".code must remain the pristine baseline source; overriding it "
        "breaks format_prog_msg and apply_diff."
    )
    assert init_sol.m1b_manifest == seed_text
    assert init_sol.model_msg == seed_text


def test_init_manifest_absent_leaves_all_fields_alone(tmp_path):
    """When INIT_MANIFEST_FILE is not set, the override is a no-op."""
    pristine = "int main() { return 0; }\n"
    init_sol = _make_init_sol(pristine)
    # No INIT_MANIFEST_FILE in config.
    config: dict = {}

    _apply_init_manifest_override(init_sol, config)

    assert init_sol.code == pristine
    assert init_sol.m1b_manifest is None
    assert init_sol.model_msg is None


def test_init_manifest_override_matches_production_block():
    """Guard: the production override block in evolution.py must match
    the logic exercised here. If this fails, update the helper + tests."""
    import inspect

    from codeevolve import evolution

    src = inspect.getsource(evolution._initialize_new_run)
    # The production block must set m1b_manifest and model_msg from the
    # seed_text, and must NOT assign seed_text (or the manifest file
    # contents) to init_sol.code.
    assert "init_sol.m1b_manifest = seed_text" in src
    assert "init_sol.model_msg = seed_text" in src
    assert "init_sol.code = seed_text" not in src, (
        "Regression: init_sol.code was re-introduced as an override "
        "target. Leave .code pristine; only override .m1b_manifest and "
        ".model_msg. See evolution.py comment block and this file's "
        "module docstring for rationale."
    )
