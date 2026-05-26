from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Attribute:
    name: str
    inferred_type: Optional[str] = None
    confidence: Optional[float] = None
    source: str = "attribute_specialist"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "inferred_type": self.inferred_type,
            "confidence": self.confidence,
            "source": self.source,
        }


@dataclass
class Entity:
    name: str
    description: Optional[str] = None
    attributes: List[Attribute] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "attributes": [attr.to_dict() for attr in self.attributes],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        attributes_data = data.get("attributes", [])
        attributes = [
            Attribute(
                name=attr.get("name", ""),
                inferred_type=attr.get("inferred_type"),
                confidence=attr.get("confidence"),
                source=attr.get("source", "concept_extractor")
            )
            for attr in attributes_data
            if attr.get("name")
        ]
        return cls(
            name=data.get("name", ""),
            description=data.get("description"),
            attributes=attributes,
        )


@dataclass
class AttributeSpecialistInput:
    requirements: str
    entities: List[Entity]


@dataclass
class EnrichedEntity(Entity):
    pass
