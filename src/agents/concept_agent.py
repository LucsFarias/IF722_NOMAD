from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.agents.base import BaseAgent
from src.schemas import UMLAttribute, UMLClass, UMLModel
from src.utils.json_parser import parse_llm_json

logger = logging.getLogger(__name__)


class ConceptAgent(BaseAgent):
    def __init__(
        self,
        llm: Optional[Any] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        super().__init__(
            name="ConceptAgent",
            llm=llm,
            provider=provider,
            model=model,
            temperature=temperature,
        )

    def extract_concepts(self, requirements: str) -> UMLModel:
        prompt = self._build_prompt(requirements)
        response = self.call_llm(prompt)
        try:
            payload = self._parse_response(response)
        except ValueError:
            retry_response = self.call_llm(
                prompt
                + "\n\nThe previous response was invalid. Return only valid JSON matching the requested schema."
            )
            payload = self._parse_response(retry_response)

        classes_payload = payload.get("classes", []) if isinstance(payload, dict) else []
        classes = [self._parse_class(item) for item in classes_payload if isinstance(item, dict)]
        classes = [uml_class for uml_class in classes if uml_class.name]
        return UMLModel(classes=classes, relationships=[], metadata={"source": "concept_agent"})

    def _build_prompt(self, requirements: str) -> str:
        return (
            "You are a UML concept extraction specialist.\n"
            "Extract candidate UML classes from the requirements.\n"
            "Return only JSON in the following format:\n"
            "{\n"
            '  "classes": [\n'
            "    {\n"
            '      "name": "ClassName",\n'
            '      "description": "Short description",\n'
            '      "attributes": [\n'
            '        {"name": "attributeName", "data_type": "String"}\n'
            "      ],\n"
            '      "is_abstract": false,\n'
            '      "stereotype": null\n'
            "    }\n"
            "  ]\n"
            "}\n"
            f"Requirements:\n{requirements}"
        )

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        try:
            return parse_llm_json(response_text)
        except (TypeError, ValueError) as exc:
            logger.exception("Failed to parse ConceptAgent response as JSON.")
            raise ValueError("Failed to parse ConceptAgent response as JSON.") from exc

    def _parse_class(self, payload: Dict[str, Any]) -> UMLClass:
        attributes = [
            UMLAttribute.from_dict(attribute)
            for attribute in payload.get("attributes", [])
            if isinstance(attribute, dict)
        ]
        return UMLClass(
            name=str(payload.get("name", "")).strip(),
            description=payload.get("description"),
            attributes=attributes,
            is_abstract=bool(payload.get("is_abstract", False)),
            stereotype=payload.get("stereotype"),
        )
