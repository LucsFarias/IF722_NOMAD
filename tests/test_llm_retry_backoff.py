from types import SimpleNamespace
from typing import Optional

import pytest

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


class FakeGoogleGenerativeAI:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, prompt: str):
        self.calls += 1
        return SimpleNamespace(content="gemini-ok")


def test_call_llm_retries_on_rate_limit(monkeypatch):
    sleeps = []
    monkeypatch.setattr("src.utils.llm_call.time.sleep", lambda seconds: sleeps.append(seconds))

    agent = BaseAgent(name="RetryAgent", llm=FlakyLLM())
    result = agent.call_llm("hello")

    assert result == "ok"
    assert agent.llm.calls == 2
    assert sleeps == [0.01]


def test_call_llm_applies_gemini_delay_without_slowing_tests(monkeypatch):
    sleeps = []
    monkeypatch.setenv("GEMINI_REQUEST_DELAY_SECONDS", "15")
    monkeypatch.setattr("src.utils.llm_call.time.sleep", lambda seconds: sleeps.append(seconds))

    agent = BaseAgent(name="GeminiAgent", llm=FakeGoogleGenerativeAI())
    result = agent.call_llm("hello")

    assert result == "gemini-ok"
    assert sleeps == [15.0]


def test_call_llm_stops_after_rate_limit_attempts(monkeypatch):
    class AlwaysRateLimitedLLM:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, prompt: str):
            self.calls += 1
            raise RateLimitError("429 ResourceExhausted: You exceeded your current quota.", retry_delay=0.01)

    sleeps = []
    monkeypatch.setattr("src.utils.llm_call.time.sleep", lambda seconds: sleeps.append(seconds))

    agent = BaseAgent(name="RetryAgent", llm=AlwaysRateLimitedLLM())

    with pytest.raises(RateLimitError):
        agent.call_llm("hello")

    assert agent.llm.calls == 3
    assert sleeps == [0.01, 0.01]


def test_call_llm_reports_daily_quota_exhaustion(monkeypatch):
    class DailyQuotaLLM:
        def invoke(self, prompt: str):
            raise RateLimitError("429 ResourceExhausted: Daily quota exceeded.")

    monkeypatch.setattr("src.utils.llm_call.time.sleep", lambda seconds: None)
    agent = BaseAgent(name="DailyQuotaAgent", llm=DailyQuotaLLM())

    with pytest.raises(RuntimeError, match="daily quota"):
        agent.call_llm("hello")
