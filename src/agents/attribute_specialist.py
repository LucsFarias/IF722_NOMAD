import json
import logging
from typing import Any, Dict, List

from langchain import LLMChain
from langchain.chat_models import ChatOpenAI

from src.prompts.attribute_specialist_prompt import ATTRIBUTE_SPECIALIST_PROMPT
from src.types.attribute_specialist import Attribute, Entity

logger = logging.getLogger(__name__)


class AttributeSpecialist:
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.0):
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.chain = LLMChain(llm=self.llm, prompt=ATTRIBUTE_SPECIALIST_PROMPT)

    def enrich_entities(self, entities: List[Dict[str, Any]], requirements: str) -> List[Dict[str, Any]]:
        enriched_entities: List[Dict[str, Any]] = []

        for entity_data in entities:
            entity = Entity.from_dict(entity_data)
            existing_attributes = [attr.name for attr in entity.attributes]

            if not entity.name:
                logger.warning("Skipping entity with empty name.")
                enriched_entities.append(entity_data)
                continue

            try:
                result = self.chain.run(
                    entity_name=entity.name,
                    entity_description=entity.description or "",
                    existing_attributes=", ".join(existing_attributes) or "(none)",
                    requirements=requirements,
                )
                inferred_attributes = self._parse_llm_response(result)
            except Exception as exc:
                logger.exception("AttributeSpecialist failed for entity %s", entity.name)
                inferred_attributes = []

            merged_attributes = self._merge_attributes(entity.attributes, inferred_attributes)
            enriched_entity = {
                "name": entity.name,
                "description": entity.description,
                "attributes": [attr.to_dict() for attr in merged_attributes],
            }
            enriched_entities.append(enriched_entity)

        return enriched_entities

    def _parse_llm_response(self, response_text: str) -> List[Attribute]:
        try:
            payload = json.loads(response_text.strip())
        except json.JSONDecodeError:
            logger.warning("Failed to parse AttributeSpecialist response as JSON: %s", response_text)
            return []

        raw_attributes = payload.get("attributes", [])
        parsed_attributes: List[Attribute] = []

        if not isinstance(raw_attributes, list):
            return []

        for item in raw_attributes:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not name or not isinstance(name, str):
                continue
            inferred_type = item.get("inferred_type")
            confidence = item.get("confidence")
            if isinstance(confidence, (int, float)):
                confidence = float(confidence)
            else:
                confidence = None

            parsed_attributes.append(
                Attribute(
                    name=name.strip(),
                    inferred_type=inferred_type.strip() if isinstance(inferred_type, str) else None,
                    confidence=confidence,
                )
            )

        return parsed_attributes

    def _merge_attributes(self, existing: List[Attribute], inferred: List[Attribute]) -> List[Attribute]:
        seen = {attr.name.lower() for attr in existing}
        merged = list(existing)

        for item in inferred:
            if item.name.lower() in seen:
                continue
            merged.append(item)
            seen.add(item.name.lower())

        return merged
