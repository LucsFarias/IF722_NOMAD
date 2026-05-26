import logging
from typing import Any, Dict, List, Optional

from src.agents.attribute_specialist import AttributeSpecialist

logger = logging.getLogger(__name__)


def concept_extractor_stub(requirements: str) -> List[Dict[str, Any]]:
    """
    Minimal stub for Concept Extractor for MVP purposes.
    Replace this with the project's real Concept Extractor.
    Returns a list of entities as dicts: {"name": str, "description": str, "attributes": [{...}]}
    """
    # Very small heuristic example: look for lines starting with 'Class: '
    entities: List[Dict[str, Any]] = []
    for line in requirements.splitlines():
        line = line.strip()
        if line.lower().startswith("class:"):
            name = line.split("class:", 1)[1].strip()
            entities.append({"name": name, "description": "(from stub)", "attributes": []})
    # If none found, return an example entity for quick tests
    if not entities:
        entities = [{"name": "Order", "description": "An order placed by a customer.", "attributes": [{"name": "id"}]},
                    {"name": "Customer", "description": "A buyer in the system.", "attributes": [{"name": "id"}, {"name": "email"}]}]
    return entities


def relationship_comprehender_stub(entities: List[Dict[str, Any]], requirements: str) -> List[Dict[str, Any]]:
    """Placeholder for Relationship Comprehender. Returns empty list for MVP."""
    logger.info("Relationship comprehender received %d entities", len(entities))
    return []


def run_pipeline(requirements: str, concept_entities: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    # Stage 1: Concept Extraction (or use provided entities)
    entities = concept_entities if concept_entities is not None else concept_extractor_stub(requirements)

    # Stage 1.5: Attribute Specialist
    attr_specialist = AttributeSpecialist()
    try:
        enriched = attr_specialist.enrich_entities(entities, requirements)
    except Exception as e:
        logger.exception("AttributeSpecialist failed: %s", e)
        enriched = entities  # fallback: proceed with original entities

    # Stage 2: Relationship Comprehender (stub)
    relationships = relationship_comprehender_stub(enriched, requirements)

    result = {
        "entities": enriched,
        "relationships": relationships,
    }
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample_requirements = """
    Class: Order
    Class: Customer
    Orders have a total price and a created_at timestamp. Customers have a shipping_address and phone.
    """
    output = run_pipeline(sample_requirements)
    print(output)
