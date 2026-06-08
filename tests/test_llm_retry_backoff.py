from types import SimpleNamespace
from typing import Optional
from src.agents.base import BaseAgent


class RateLimitError(RuntimeError):
    def __init__(self, message: str, retry_delay: Optional[float] = None) -> None:
        super().__init__(message)
        self.retry_delay = retry_delay


class FlakyLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.prompts = []

    def invoke(self, prompt: str):
        self.calls += 1
        self.prompts.append(prompt)
        if self.calls == 1:
            raise RateLimitError("429 ResourceExhausted: You exceeded your current quota.", retry_delay=0.01)
        return SimpleNamespace(content="ok")


def test_call_llm_retries_on_rate_limit(monkeypatch):
    sleeps = []
    monkeypatch.setattr("src.utils.llm_call.time.sleep", lambda seconds: sleeps.append(seconds))

    agent = BaseAgent(name="RetryAgent", llm=FlakyLLM())
    result = agent.call_llm("hello")

    assert result == "ok"
    assert agent.llm.calls == 2
    assert sleeps == [0.01]
