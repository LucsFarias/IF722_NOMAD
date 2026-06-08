from __future__ import annotations

from abc import ABC
from typing import Any, Optional

from src.llm.provider import create_llm_provider
from src.utils.llm_call import invoke_llm_with_retry


class BaseAgent(ABC):
    def __init__(
        self,
        name: str,
        llm: Optional[Any] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> None:
        self.name = name
        self.llm = llm if llm is not None else create_llm_provider(
            provider=provider,
            model=model,
            temperature=temperature,
        )

    def call_llm(self, prompt: str) -> str:
        return invoke_llm_with_retry(self.llm, prompt, agent_name=self.name)
