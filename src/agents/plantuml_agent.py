from __future__ import annotations

from typing import Optional

from src.schemas import UMLAttribute, UMLClass, UMLModel, UMLRelationship


class PlantUMLAgent:
    def __init__(self, llm: Optional[object] = None) -> None:
        self.llm = llm

    def generate(self, model: UMLModel) -> str:
        lines = ["@startuml"]
        for uml_class in model.classes:
            lines.extend(self._render_class(uml_class))

        for relationship in model.relationships:
            line = self._render_relationship(relationship)
            if line:
                lines.append(line)

        lines.append("@enduml")
        return "\n".join(lines)

    def revise(self, model: UMLModel) -> str:
        return self.generate(model)

    def _render_class(self, uml_class: UMLClass):
        header = "abstract class" if uml_class.is_abstract else "class"
        if uml_class.stereotype:
            header = f'{header} "{uml_class.stereotype}"'
        lines = [f"{header} {uml_class.name} {{"]
        for attribute in uml_class.attributes:
            lines.append(self._render_attribute(attribute))
        lines.append("}")
        return lines

    def _render_attribute(self, attribute: UMLAttribute) -> str:
        visibility = attribute.visibility or "+"
        data_type = attribute.data_type or "String"
        return f"  {visibility} {attribute.name}: {data_type}"

    def _render_relationship(self, relationship: UMLRelationship) -> str:
        rel_type = relationship.relationship_type.lower().strip()
        left = relationship.source
        right = relationship.target

        if rel_type in {"inheritance", "generalization", "extends"}:
            return f"{left} --|> {right}"
        if rel_type in {"composition", "composes"}:
            connector = "*--"
        elif rel_type in {"aggregation", "aggregates"}:
            connector = "o--"
        elif rel_type in {"dependency"}:
            connector = "..>"
        else:
            connector = "--"

        left_token = f' "{relationship.source_multiplicity}"' if relationship.source_multiplicity else ""
        right_token = f' "{relationship.target_multiplicity}"' if relationship.target_multiplicity else ""
        label = f" : {relationship.label}" if relationship.label else ""
        return f"{left}{left_token} {connector}{right_token} {right}{label}".strip()
