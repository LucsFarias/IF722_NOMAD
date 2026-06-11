from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.schemas import UMLAttribute as Attribute
from src.schemas import UMLClass as Entity


@dataclass
class AttributeSpecialistInput:
    requirements: str
    entities: List[Entity]


@dataclass
class EnrichedEntity(Entity):
    pass
