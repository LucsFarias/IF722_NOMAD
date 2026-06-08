from __future__ import annotations

import re
import logging
from typing import Any, Dict, List, Optional

from src.agents.base import BaseAgent
from src.schemas import UMLModel, UMLRelationship
from src.utils.json_parser import parse_llm_json

logger = logging.getLogger(__name__)


class RelationshipAgent(BaseAgent):
    def __init__(
        self,
        llm: Optional[Any] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        super().__init__(
            name="RelationshipAgent",
            llm=llm,
            provider=provider,
            model=model,
            temperature=temperature,
        )

    def add_relationships(self, model: UMLModel, requirements: str) -> UMLModel:
        prompt = self._build_prompt(model, requirements)
        response = self.call_llm(prompt)
        try:
            payload = self._parse_response(response)
        except ValueError:
            retry_response = self.call_llm(
                prompt
                + "\n\nThe previous response was invalid. Return only valid JSON matching the requested schema."
            )
            payload = self._parse_response(retry_response)

        relationships_payload = payload.get("relationships", []) if isinstance(payload, dict) else []
        relationships = [
            self._parse_relationship(item)
            for item in relationships_payload
            if isinstance(item, dict)
        ]
        relationships.extend(self._infer_relationships_from_requirements(model, requirements))
        relationships = self._filter_relationships(relationships)
        merged_relationships = self._merge_relationships(list(model.relationships), relationships)
        return UMLModel(classes=list(model.classes), relationships=merged_relationships, metadata=dict(model.metadata))

    def revise(self, model: UMLModel, requirements: str) -> UMLModel:
        prompt = self._build_prompt(model, requirements)
        response = self.call_llm(prompt)
        try:
            payload = self._parse_response(response)
        except ValueError:
            retry_response = self.call_llm(
                prompt
                + "\n\nThe previous response was invalid. Return only valid JSON matching the requested schema."
            )
            payload = self._parse_response(retry_response)

        relationships_payload = payload.get("relationships", []) if isinstance(payload, dict) else []
        relationships = [
            self._parse_relationship(item)
            for item in relationships_payload
            if isinstance(item, dict)
        ]
        relationships.extend(self._infer_relationships_from_requirements(model, requirements))
        relationships = self._filter_relationships(relationships)
        merged_relationships = self._merge_relationships(list(model.relationships), relationships)
        return UMLModel(classes=list(model.classes), relationships=merged_relationships, metadata=dict(model.metadata))

    def _build_prompt(self, model: UMLModel, requirements: str) -> str:
        class_names = ", ".join(uml_class.name for uml_class in model.classes) or "(none)"
        return (
            "You are a UML relationship extraction specialist.\n"
            "Infer relationships between the candidate classes.\n"
            "Return only JSON in the following format:\n"
            "{\n"
            '  "relationships": [\n'
            "    {\n"
            '      "source": "ClassA",\n'
            '      "target": "ClassB",\n'
            '      "relationship_type": "association",\n'
            '      "source_multiplicity": "1",\n'
            '      "target_multiplicity": "*",\n'
            '      "label": null,\n'
            '      "metadata": {}\n'
            "    }\n"
            "  ]\n"
            "}\n"
            f"Known classes: {class_names}\n"
            f"Requirements:\n{requirements}"
        )

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        try:
            return parse_llm_json(response_text)
        except (TypeError, ValueError) as exc:
            logger.exception("Failed to parse RelationshipAgent response as JSON.")
            raise ValueError("Failed to parse RelationshipAgent response as JSON.") from exc

    def _parse_relationship(self, payload: Dict[str, Any]) -> UMLRelationship:
        metadata = payload.get("metadata", {})
        return UMLRelationship(
            source=str(payload.get("source", "")).strip(),
            target=str(payload.get("target", "")).strip(),
            relationship_type=str(payload.get("relationship_type", "association")).strip(),
            source_multiplicity=payload.get("source_multiplicity"),
            target_multiplicity=payload.get("target_multiplicity"),
            source_role=payload.get("source_role"),
            target_role=payload.get("target_role"),
            label=payload.get("label"),
            metadata=dict(metadata) if isinstance(metadata, dict) else {},
        )

    def _merge_relationships(
        self,
        existing: List[UMLRelationship],
        incoming: List[UMLRelationship],
    ) -> List[UMLRelationship]:
        merged: List[UMLRelationship] = []
        index: Dict[tuple, int] = {}

        for relationship in existing:
            key = self._relationship_key(relationship)
            index[key] = len(merged)
            merged.append(relationship)

        for relationship in incoming:
            key = self._relationship_key(relationship)
            if key in index:
                merged[index[key]] = self._merge_relationship(merged[index[key]], relationship)
            else:
                index[key] = len(merged)
                merged.append(relationship)

        return merged

    def _relationship_key(self, relationship: UMLRelationship) -> tuple:
        return (
            relationship.source.strip().lower(),
            relationship.target.strip().lower(),
            relationship.relationship_type.strip().lower(),
            self._normalize_multiplicity(relationship.source_multiplicity),
            self._normalize_multiplicity(relationship.target_multiplicity),
        )

    def _merge_relationship(self, existing: UMLRelationship, incoming: UMLRelationship) -> UMLRelationship:
        return UMLRelationship(
            source=incoming.source or existing.source,
            target=incoming.target or existing.target,
            relationship_type=incoming.relationship_type or existing.relationship_type,
            source_multiplicity=self._normalize_multiplicity(incoming.source_multiplicity or existing.source_multiplicity),
            target_multiplicity=self._normalize_multiplicity(incoming.target_multiplicity or existing.target_multiplicity),
            source_role=incoming.source_role if incoming.source_role is not None else existing.source_role,
            target_role=incoming.target_role if incoming.target_role is not None else existing.target_role,
            label=incoming.label or existing.label,
            metadata={**existing.metadata, **incoming.metadata},
        )

    def _filter_relationships(self, relationships: List[UMLRelationship]) -> List[UMLRelationship]:
        filtered: List[UMLRelationship] = []
        for relationship in relationships:
            if not relationship.source or not relationship.target:
                continue
            if relationship.source.strip().lower() == relationship.target.strip().lower():
                continue
            filtered.append(relationship)
        return filtered

    def _infer_relationships_from_requirements(self, model: UMLModel, requirements: str) -> List[UMLRelationship]:
        requirements_text = " ".join(requirements.split())
        lower_text = requirements_text.lower()
        class_lookup = {self._normalize_class_name(uml_class.name): uml_class.name for uml_class in model.classes}
        inferred: List[UMLRelationship] = []

        for match in re.finditer(
            r"(?P<source>[A-Za-z][A-Za-z0-9_\-\s]+?)\s+aggregat(?:e|es)\s+any number of\s+(?P<target>[A-Za-z][A-Za-z0-9_\-\s]+)",
            requirements_text,
            flags=re.IGNORECASE,
        ):
            source = self._match_class_name(match.group("source"), class_lookup)
            target = self._match_class_name(match.group("target"), class_lookup)
            if source and target and source.lower() != target.lower():
                inferred.append(
                    UMLRelationship(
                        source=source,
                        target=target,
                        relationship_type="aggregation",
                        source_multiplicity="1",
                        target_multiplicity="*",
                    )
                )

        inferred.extend(self._infer_generalizations_from_phrases(requirements_text, class_lookup))

        if "self" in lower_text:
            for relationship in list(inferred):
                if relationship.source.lower() == relationship.target.lower():
                    inferred.remove(relationship)

        return inferred

    def _infer_generalizations_from_phrases(self, requirements_text: str, class_lookup: Dict[str, str]) -> List[UMLRelationship]:
        inferred: List[UMLRelationship] = []
        patterns = [
            r"(?P<super>[A-Za-z][A-Za-z0-9_\-\s]+?)\s+(?:can be|may be|are)\s+(?:either\s+)?(?:an?\s+)?(?P<option1>[A-Za-z][A-Za-z0-9_\-\s]+?)\s+or\s+(?:an?\s+)?(?P<option2>[A-Za-z][A-Za-z0-9_\-\s]+?)(?:[.,;]|$)",
            r"(?P<super>[A-Za-z][A-Za-z0-9_\-\s]+?)\s+elements\s+which\s+(?:can be|may be|are)\s+(?:either\s+)?(?:an?\s+)?(?P<option1>[A-Za-z][A-Za-z0-9_\-\s]+?)\s+or\s+(?:an?\s+)?(?P<option2>[A-Za-z][A-Za-z0-9_\-\s]+?)(?:[.,;]|$)",
            r"(?P<super>[A-Za-z][A-Za-z0-9_\-\s]+?)\s+elements\s+(?:can be|may be|are)\s+(?:either\s+)?(?:an?\s+)?(?P<option1>[A-Za-z][A-Za-z0-9_\-\s]+?)\s+or\s+(?:an?\s+)?(?P<option2>[A-Za-z][A-Za-z0-9_\-\s]+?)(?:[.,;]|$)",
            r"(?P<super>[A-Za-z][A-Za-z0-9_\-\s]+?)\s+(?:can be|may be|are)\s+(?:either\s+)?(?:an?\s+)?(?P<option1>[A-Za-z][A-Za-z0-9_\-\s]+?)\s+or\s+(?:an?\s+)?(?P<option2>[A-Za-z][A-Za-z0-9_\-\s]+?)(?:[.,;]|$)",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, requirements_text, flags=re.IGNORECASE):
                parent_phrase = self._clean_generalization_phrase(match.group("super"))
                parent = self._match_class_name(parent_phrase, class_lookup)
                child_one = self._match_class_name(match.group("option1"), class_lookup)
                child_two = self._match_class_name(match.group("option2"), class_lookup)
                if parent and child_one and parent.lower() != child_one.lower():
                    inferred.append(
                        UMLRelationship(
                            source=child_one,
                            target=parent,
                            relationship_type="generalization",
                        )
                    )
                if parent and child_two and parent.lower() != child_two.lower():
                    inferred.append(
                        UMLRelationship(
                            source=child_two,
                            target=parent,
                            relationship_type="generalization",
                        )
                    )
        return inferred

    def _normalize_class_name(self, text: str) -> str:
        return re.sub(r"[^a-z0-9]", "", text.lower())

    def _normalize_multiplicity(self, multiplicity: Optional[str]) -> Optional[str]:
        if multiplicity is None:
            return None
        normalized = str(multiplicity).strip().lower()
        if normalized in {"*", "0..*", "0..n", "many", "n"}:
            return "*"
        if normalized in {"1", "1..1", "one"}:
            return "1"
        return multiplicity.strip()

    def _match_class_name(self, phrase: str, class_lookup: Dict[str, str]) -> Optional[str]:
        normalized_phrase = self._normalize_class_name(phrase)
        if normalized_phrase in class_lookup:
            return class_lookup[normalized_phrase]
        singular = normalized_phrase[:-1] if normalized_phrase.endswith("s") else normalized_phrase
        if singular in class_lookup:
            return class_lookup[singular]
        return None

    def _clean_generalization_phrase(self, phrase: str) -> str:
        cleaned = phrase.strip()
        cleaned = re.sub(r"\b(which|that)\s*$", "", cleaned, flags=re.IGNORECASE).strip()
        return cleaned
