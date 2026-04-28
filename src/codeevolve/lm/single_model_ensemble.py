"""SingleModelEnsemble — minimal BaseEnsemble wrapping a single LM.

Used for the --ea-strategy claude_code path so the rest of evolution.py
can treat the CC-only configuration as just another ensemble. Required
because OpenAIEnsemble's constructor expects an OpenAI-shaped
`models_cfg: List[Dict]` and instantiates OpenAILM/MockOpenAILM
internally — we cannot pass a pre-built ClaudeCodeLM instance through it.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from codeevolve.lm.base import BaseEnsemble, BaseLM


class SingleModelEnsemble(BaseEnsemble):
    """Ensemble with exactly one model and no rotation."""

    def __init__(self, model: BaseLM, logger: Optional[logging.Logger] = None):
        self._model = model
        self.logger = logger or logging.getLogger(__name__)

    async def generate(
        self, messages: List[Dict[str, str]]
    ) -> Tuple[int, str, int, int]:
        text, prompt_tokens, completion_tokens = await self._model.generate(messages)
        return 0, text, prompt_tokens, completion_tokens

    @property
    def models(self) -> List[BaseLM]:
        return [self._model]

    def select_model(self) -> BaseLM:
        return self._model
