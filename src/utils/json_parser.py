from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _strip_markdown_fences(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text

    cleaned_lines = []
    for line in lines:
        if line.strip().startswith("```"):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def _extract_braced_json(text: str) -> Optional[str]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _ensure_dict(payload: Any, source: str) -> Dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    raise ValueError(
        f"LLM response must decode to a JSON object, got {type(payload).__name__} from {source}."
    )


def parse_llm_json(text: str) -> Dict[str, Any]:
    if not isinstance(text, str):
        raise TypeError(f"parse_llm_json expected str, got {type(text).__name__}.")

    stripped = text.strip()
    if not stripped:
        raise ValueError("LLM response is empty and cannot be parsed as JSON.")

    cleaned = _strip_markdown_fences(stripped).strip()

    try:
        return _ensure_dict(json.loads(cleaned), "direct json.loads()")
    except json.JSONDecodeError as direct_error:
        candidate = _extract_braced_json(cleaned)
        if candidate is None:
            logger.error(
                "Failed to parse LLM JSON. No JSON object delimiters found. preview=%r",
                cleaned[:500],
            )
            raise ValueError("Failed to parse LLM response as JSON object.") from direct_error

        try:
            return _ensure_dict(json.loads(candidate), "brace extraction")
        except json.JSONDecodeError as brace_error:
            logger.error(
                "Failed to parse LLM JSON after stripping fences and extracting braces. preview=%r",
                cleaned[:500],
            )
            raise ValueError("Failed to parse LLM response as JSON object.") from brace_error
