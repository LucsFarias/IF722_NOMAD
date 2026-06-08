from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ChatGoogleGenerativeAI = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from langchain_ollama import ChatOllama  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ChatOllama = None  # type: ignore


DEFAULT_PROVIDER = "gemini"
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


def _empty_to_none(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return value or None


@dataclass(frozen=True)
class LLMConfig:
    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    google_api_key: Optional[str] = None
    ollama_base_url: Optional[str] = DEFAULT_OLLAMA_BASE_URL

    @classmethod
    def from_env(cls) -> "LLMConfig":
        # Load .env automatically if python-dotenv is installed.
        # This keeps local execution simple and does not affect tests using mocks.
        if load_dotenv is not None:
            load_dotenv()

        provider = os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower()
        model = os.getenv("LLM_MODEL", DEFAULT_MODEL).strip()

        try:
            temperature = float(
                os.getenv("LLM_TEMPERATURE", str(DEFAULT_TEMPERATURE))
            )
        except ValueError as exc:
            raise ValueError(
                "LLM_TEMPERATURE must be a valid float, e.g. 0 or 0.0."
            ) from exc

        google_api_key = _empty_to_none(
            os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        )

        ollama_base_url = _empty_to_none(
            os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
        )

        return cls(
            provider=provider,
            model=model,
            temperature=temperature,
            google_api_key=google_api_key,
            ollama_base_url=ollama_base_url,
        )


def _build_gemini_provider(config: LLMConfig, extra_kwargs: Dict[str, Any]) -> Any:
    if ChatGoogleGenerativeAI is None:
        raise RuntimeError(
            "Gemini provider requested but langchain_google_genai is not installed. "
            "Install it with: pip install langchain-google-genai"
        )

    if not config.google_api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is required when LLM_PROVIDER=gemini. "
            "Set it in your .env file or export it in your shell."
        )

    # Critical: force Gemini Developer API, not Vertex AI / Google Cloud ADC.
    # LangChain/google-genai checks GOOGLE_GENAI_USE_VERTEXAI when selecting backend.
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "false"

    # Keep both names for compatibility with google-genai/langchain versions.
    os.environ["GOOGLE_API_KEY"] = config.google_api_key
    os.environ.setdefault("GEMINI_API_KEY", config.google_api_key)

    # These parameters would force Vertex AI. Do not allow them in this project.
    forbidden_vertex_kwargs = {
        "credentials",
        "project",
        "location",
        "google_cloud_project",
        "google_cloud_location",
    }
    forbidden_used = forbidden_vertex_kwargs.intersection(extra_kwargs)
    if forbidden_used:
        raise ValueError(
            "Gemini provider in this project must use Gemini Developer API, "
            f"not Vertex AI. Remove these kwargs: {sorted(forbidden_used)}"
        )

    kwargs: Dict[str, Any] = {
        "model": config.model,
        "temperature": config.temperature,
        "vertexai": False,
        **extra_kwargs,
    }

    # Newer langchain-google-genai versions accept api_key.
    # Some older versions used google_api_key.
    # If one form is unsupported, try the other.
    try:
        return ChatGoogleGenerativeAI(
            google_api_key=config.google_api_key,
            **kwargs,
        )
    except TypeError:
        return ChatGoogleGenerativeAI(
            api_key=config.google_api_key,
            **kwargs,
        )


def _build_ollama_provider(config: LLMConfig, extra_kwargs: Dict[str, Any]) -> Any:
    if ChatOllama is None:
        raise RuntimeError(
            "Ollama provider requested but langchain_ollama is not installed. "
            "Install it with: pip install langchain-ollama"
        )

    kwargs: Dict[str, Any] = {
        "model": config.model,
        "temperature": config.temperature,
        **extra_kwargs,
    }

    if config.ollama_base_url:
        kwargs["base_url"] = config.ollama_base_url

    return ChatOllama(**kwargs)


def create_llm_provider(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    **extra_kwargs: Any,
) -> Any:
    config = LLMConfig.from_env()

    config = LLMConfig(
        provider=(provider.strip().lower() if provider is not None else config.provider),
        model=(model if model is not None else config.model),
        temperature=(temperature if temperature is not None else config.temperature),
        google_api_key=config.google_api_key,
        ollama_base_url=config.ollama_base_url,
    )

    if config.provider == "gemini":
        return _build_gemini_provider(config, extra_kwargs)

    if config.provider == "ollama":
        return _build_ollama_provider(config, extra_kwargs)

    raise ValueError(
        f"Unsupported LLM provider: {config.provider}. "
        "Expected 'gemini' or 'ollama'."
    )


def get_llm_provider(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    **extra_kwargs: Any,
) -> Any:
    return create_llm_provider(
        provider=provider,
        model=model,
        temperature=temperature,
        **extra_kwargs,
    )