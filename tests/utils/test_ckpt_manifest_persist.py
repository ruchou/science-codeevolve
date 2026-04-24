import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest

from codeevolve.utils.ckpt import save_ckpt


class _FakeProgDB:
    def __init__(self, code: str, manifest: str | None):
        self.best_prog_id = 0
        self.programs = {0: SimpleNamespace(code=code, m1b_manifest=manifest)}


def test_save_ckpt_writes_m1b_manifest_sibling(tmp_path):
    sol_db = _FakeProgDB(code="int main(){return 0;}", manifest="RATIONALE: try pack\nFILE: src/a.c\n")
    prompt_db = _FakeProgDB(code="prompt text", manifest=None)
    best_sol = tmp_path / "best_sol.cpp"
    best_prompt = tmp_path / "best_prompt.txt"
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()

    import logging
    save_ckpt(
        curr_epoch=3,
        prompt_db=prompt_db,
        sol_db=sol_db,
        evolve_state={},
        exploration_scheduler=None,
        best_sol_path=best_sol,
        best_prompt_path=best_prompt,
        ckpt_dir=ckpt_dir,
        logger=logging.getLogger("t"),
    )

    manifest_sibling = best_sol.with_suffix(best_sol.suffix + ".m1b_manifest.txt")
    assert manifest_sibling.exists(), "Expected manifest sibling file next to best_sol"
    assert "RATIONALE: try pack" in manifest_sibling.read_text()


def test_save_ckpt_omits_manifest_sibling_when_none(tmp_path):
    sol_db = _FakeProgDB(code="int main(){}", manifest=None)
    prompt_db = _FakeProgDB(code="p", manifest=None)
    best_sol = tmp_path / "best_sol.cpp"
    best_prompt = tmp_path / "best_prompt.txt"
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    import logging
    save_ckpt(
        curr_epoch=0,
        prompt_db=prompt_db,
        sol_db=sol_db,
        evolve_state={},
        exploration_scheduler=None,
        best_sol_path=best_sol,
        best_prompt_path=best_prompt,
        ckpt_dir=ckpt_dir,
        logger=logging.getLogger("t"),
    )
    manifest_sibling = best_sol.with_suffix(best_sol.suffix + ".m1b_manifest.txt")
    assert not manifest_sibling.exists()
