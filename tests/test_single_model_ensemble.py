from typing import Dict, List, Tuple

import pytest

from codeevolve.lm.base import BaseLM
from codeevolve.lm.single_model_ensemble import SingleModelEnsemble


class _StubLM(BaseLM):
    name: str = "stub"
    weight: float = 1.0

    def __init__(self, response: Tuple[str, int, int] = ("hello", 1, 2)):
        self._response = response

    async def generate(self, messages: List[Dict[str, str]]) -> Tuple[str, int, int]:
        return self._response


@pytest.mark.asyncio
async def test_single_model_ensemble_round_trips_response():
    ensemble = SingleModelEnsemble(_StubLM(response=("payload", 7, 13)))
    model_id, text, prompt_tokens, completion_tokens = await ensemble.generate(
        [{"role": "user", "content": "go"}]
    )
    assert model_id == 0
    assert text == "payload"
    assert prompt_tokens == 7
    assert completion_tokens == 13


def test_single_model_ensemble_models_property_lists_one_model():
    stub = _StubLM()
    ensemble = SingleModelEnsemble(stub)
    assert ensemble.models == [stub]


def test_single_model_ensemble_select_model_returns_wrapped():
    stub = _StubLM()
    ensemble = SingleModelEnsemble(stub)
    assert ensemble.select_model() is stub
