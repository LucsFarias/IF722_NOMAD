from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_SLEEP_SECONDS = 10.0


def _response_to_text(response: Any) -> str:
    if isinstance(response, str):
        return response

    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content

    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text

    return str(response)


def _is_rate_limit_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        token in text
        for token in ("429", "resourceexhausted", "rate limit", "rate_limit", "quota")
    )


def _retry_delay_seconds(exc: BaseException, fallback: Optional[float]) -> float:
    retry_delay = getattr(exc, "retry_delay", None)
    if retry_delay is not None:
        try:
            value = float(retry_delay)
            if value > 0:
                return value
        except (TypeError, ValueError):
            pass

    if fallback is not None:
        return max(float(fallback), 0.0)

    env_value = os.getenv("LLM_CALL_SLEEP_SECONDS")
    if env_value is not None:
        try:
            return max(float(env_value), 0.0)
        except ValueError:
            logger.warning(
                "Invalid LLM_CALL_SLEEP_SECONDS=%r. Falling back to %.1f seconds.",
                env_value,
                DEFAULT_SLEEP_SECONDS,
            )

    return DEFAULT_SLEEP_SECONDS


def invoke_llm_with_retry(
    llm: Any,
    prompt: str,
    *,
    agent_name: str = "LLM",
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    sleep_seconds: Optional[float] = None,
) -> str:
    if llm is None:
        raise RuntimeError(f"{agent_name} does not have an LLM configured.")

    attempts = max(1, int(max_attempts))
    last_error: Optional[BaseException] = None

    for attempt in range(1, attempts + 1):
        try:
            if hasattr(llm, "invoke"):
                response = llm.invoke(prompt)
            elif callable(llm):
                response = llm(prompt)
            else:
                raise TypeError("Configured LLM does not expose invoke() or __call__().")
            return _response_to_text(response)
        except Exception as exc:  # pragma: no cover - exercised by retry tests
            last_error = exc
            if attempt >= attempts or not _is_rate_limit_error(exc):
                raise

            delay = _retry_delay_seconds(exc, sleep_seconds)
            logger.warning(
                "%s hit a rate limit (%s). Waiting %.2fs before retry %s/%s.",
                agent_name,
                exc,
                delay,
                attempt + 1,
                attempts,
            )
            time.sleep(delay)

    if last_error is not None:
        raise last_error

    raise RuntimeError(f"{agent_name} failed to invoke the LLM.")
