import logging
from typing import Any, Dict, List, Optional

from src.agents.base import BaseAgent
from src.prompts.attribute_specialist_prompt import build_attribute_specialist_prompt
from src.schemas import UMLAttribute
from src.types.attribute_specialist import Entity
from src.utils.json_parser import parse_llm_json

logger = logging.getLogger(__name__)


class AttributeSpecialistAgent(BaseAgent):
    def __init__(
        self,
        llm: Optional[Any] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        self.last_parse_error: Optional[str] = None
        super().__init__(
            name="AttributeSpecialistAgent",
            llm=llm,
            provider=provider,
            model=model,
            temperature=temperature,
        )

    def enrich_entities(self, entities: List[Dict[str, Any]], requirements: str) -> List[Dict[str, Any]]:
        enriched_entities: List[Dict[str, Any]] = []

        for entity_data in entities:
            entity = Entity.from_dict(entity_data)
            existing_attributes = [attr.name for attr in entity.attributes]

            if not entity.name:
                logger.warning("Skipping entity with empty name.")
                enriched_entities.append(entity_data)
                continue

            prompt = build_attribute_specialist_prompt(
                entity_name=entity.name,
                entity_description=entity.description or "",
                existing_attributes=", ".join(existing_attributes) or "(none)",
                requirements=requirements,
            )

            try:
                result = self.call_llm(prompt)
                inferred_attributes = self._parse_llm_response(result)
            except ValueError:
                retry_prompt = (
                    prompt
                    + "\n\nThe previous response was invalid. Return only valid JSON matching the requested schema."
                )
                retry_result = self.call_llm(retry_prompt)
                inferred_attributes = self._parse_llm_response(retry_result)
            except Exception as exc:
                logger.exception("AttributeSpecialist failed for entity %s", entity.name)
                self.last_parse_error = str(exc)
                raise

            merged_attributes = self._merge_attributes(entity.attributes, inferred_attributes)
            enriched_entity = {
                "name": entity.name,
                "description": entity.description,
                "attributes": [attr.to_dict() for attr in merged_attributes],
            }
            enriched_entities.append(enriched_entity)

        return enriched_entities

    def enrich_model(self, model, requirements: str):
        enriched_classes = self.enrich_entities(
            [uml_class.to_dict() for uml_class in model.classes],
            requirements=requirements,
        )
        from src.schemas import UMLClass, UMLModel

        return UMLModel(
            classes=[UMLClass.from_dict(payload) for payload in enriched_classes],
            relationships=list(model.relationships),
            metadata=dict(model.metadata),
        )

    def revise(self, model, requirements: str):
        revised_model = self.enrich_model(model, requirements)
        from src.schemas import UMLClass, UMLModel

        class_by_name = {uml_class.name.lower(): uml_class for uml_class in revised_model.classes}
        parent_map: Dict[str, List[str]] = {}
        child_map: Dict[str, List[str]] = {}
        for relationship in revised_model.relationships:
            if relationship.relationship_type.lower() not in {"inheritance", "generalization", "extends"}:
                continue
            parent_map.setdefault(relationship.source.lower(), []).append(relationship.target.lower())
            child_map.setdefault(relationship.target.lower(), []).append(relationship.source.lower())

        filtered_classes = []
        for uml_class in revised_model.classes:
            parent_attributes = set()
            for parent_name in parent_map.get(uml_class.name.lower(), []):
                parent_class = class_by_name.get(parent_name)
                if not parent_class:
                    continue
                parent_attributes.update(self._normalize_token(attribute.name) for attribute in parent_class.attributes)

            strict_requirement_filter = bool(parent_map.get(uml_class.name.lower()) or child_map.get(uml_class.name.lower()))
            filtered_classes.append(
                UMLClass(
                    name=uml_class.name,
                    description=uml_class.description,
                    attributes=[
                        attribute
                        for attribute in uml_class.attributes
                        if self._attribute_supported_by_requirements(
                            uml_class.name,
                            attribute.name,
                            requirements,
                            strict=strict_requirement_filter,
                        )
                        and self._normalize_token(attribute.name) not in parent_attributes
                    ],
                    is_abstract=uml_class.is_abstract,
                    stereotype=uml_class.stereotype,
                )
            )

        return UMLModel(
            classes=filtered_classes,
            relationships=list(revised_model.relationships),
            metadata=dict(revised_model.metadata),
        )

    def _parse_llm_response(self, response_text: str) -> List[UMLAttribute]:
        try:
            payload = parse_llm_json(response_text)
        except (TypeError, ValueError) as exc:
            logger.exception("Failed to parse AttributeSpecialist response as JSON.")
            self.last_parse_error = "Invalid JSON response"
            raise ValueError("Failed to parse AttributeSpecialist response as JSON.") from exc

        raw_attributes = payload.get("attributes", [])
        parsed_attributes: List[UMLAttribute] = []

        if not isinstance(raw_attributes, list):
            self.last_parse_error = "Missing or invalid 'attributes' list"
            raise ValueError("Missing or invalid 'attributes' list")

        for item in raw_attributes:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not name or not isinstance(name, str):
                continue
            inferred_type = item.get("inferred_type", item.get("data_type", item.get("type")))
            confidence = item.get("confidence")
            if isinstance(confidence, (int, float)):
                confidence = float(confidence)
            else:
                confidence = None

            parsed_attributes.append(
                UMLAttribute(
                    name=name.strip(),
                    data_type=inferred_type.strip() if isinstance(inferred_type, str) else None,
                    confidence=confidence,
                )
            )

        self.last_parse_error = None
        return parsed_attributes

    def _merge_attributes(self, existing: List[UMLAttribute], inferred: List[UMLAttribute]) -> List[UMLAttribute]:
        seen = {attr.name.lower() for attr in existing}
        merged = list(existing)

        for item in inferred:
            if item.name.lower() in seen:
                continue
            merged.append(item)
            seen.add(item.name.lower())

        return merged

    def _attribute_supported_by_requirements(
        self,
        class_name: str,
        attribute_name: str,
        requirements: str,
        strict: bool = False,
    ) -> bool:
        normalized_attr = self._normalize_token(attribute_name)
        normalized_requirements = self._normalize_token(requirements)
        normalized_class = self._normalize_token(class_name)
        if strict:
            for sentence in requirements.replace("\n", " ").split("."):
                normalized_sentence = self._normalize_token(sentence)
                if not normalized_sentence:
                    continue
                if normalized_class not in normalized_sentence and normalized_sentence not in normalized_class:
                    continue
                if normalized_attr in normalized_sentence:
                    return True
            return False
        return (
            normalized_attr in normalized_requirements
            or normalized_attr == normalized_class
            or normalized_attr.endswith(normalized_class)
        )

    def _normalize_token(self, text: str) -> str:
        return "".join(ch.lower() for ch in text if ch.isalnum())


AttributeSpecialist = AttributeSpecialistAgent
