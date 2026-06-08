from types import SimpleNamespace

import pytest

import src.llm.provider as provider_module
from src.llm.provider import create_llm_provider


class FakeChatModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def invoke(self, prompt):  # pragma: no cover - not used here, but mirrors real interface
        return SimpleNamespace(content=prompt)


def test_create_llm_provider_builds_gemini_from_environment(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("LLM_TEMPERATURE", "0")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(provider_module, "ChatGoogleGenerativeAI", FakeChatModel)

    llm = create_llm_provider()

    assert isinstance(llm, FakeChatModel)
    assert llm.kwargs["model"] == "gemini-2.5-flash"
    assert llm.kwargs["temperature"] == 0.0
    assert llm.kwargs["google_api_key"] == "test-key"


def test_create_llm_provider_builds_ollama(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "llama3.1")
    monkeypatch.setenv("LLM_TEMPERATURE", "0")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setattr(provider_module, "ChatOllama", FakeChatModel)

    llm = create_llm_provider()

    assert isinstance(llm, FakeChatModel)
    assert llm.kwargs["model"] == "llama3.1"
    assert llm.kwargs["base_url"] == "http://localhost:11434"


def test_create_llm_provider_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "unknown")

    with pytest.raises(ValueError):
        create_llm_provider()
