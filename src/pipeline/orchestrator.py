from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.agents.attribute_specialist import AttributeSpecialistAgent
from src.agents.concept_agent import ConceptAgent
from src.agents.model_integrator import ModelIntegratorAgent
from src.agents.plantuml_agent import PlantUMLAgent
from src.agents.relationship_agent import RelationshipAgent
from src.schemas import UMLModel
from src.utils.llm_call import invoke_llm_with_retry


@dataclass
class SingleAgentBaseline:
    llm: Optional[Any] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: float = 0.0

    def run(self, requirements: str) -> str:
        prompt = self._build_prompt(requirements)
        llm = self.llm or ConceptAgent(provider=self.provider, model=self.model, temperature=self.temperature).llm
        response = self._call_llm(llm, prompt)
        plantuml = self._strip_markdown_fences(response).strip()
        if "@startuml" not in plantuml.lower():
            plantuml = "@startuml\n" + plantuml.strip() + "\n@enduml"
        return plantuml

    def _build_prompt(self, requirements: str) -> str:
        return (
            "You are a UML class diagram generator.\n"
            "Generate a complete PlantUML class diagram directly from the requirements.\n"
            "Return only PlantUML, including @startuml and @enduml.\n"
            f"Requirements:\n{requirements}"
        )

    def _call_llm(self, llm: Any, prompt: str) -> str:
        return invoke_llm_with_retry(llm, prompt, agent_name="SingleAgentBaseline")

    def _strip_markdown_fences(self, text: str) -> str:
        lines = text.splitlines()
        if not lines:
            return text
        cleaned_lines = [line for line in lines if not line.strip().startswith("```")]
        return "\n".join(cleaned_lines)


@dataclass
class CascadeNOMAD:
    concept_agent: Optional[ConceptAgent] = None
    attribute_specialist: Optional[AttributeSpecialistAgent] = None
    relationship_agent: Optional[RelationshipAgent] = None
    integrator: Optional[ModelIntegratorAgent] = None
    plantuml_agent: Optional[PlantUMLAgent] = None

    def __post_init__(self) -> None:
        self.concept_agent = self.concept_agent or ConceptAgent()
        self.attribute_specialist = self.attribute_specialist or AttributeSpecialistAgent()
        self.relationship_agent = self.relationship_agent or RelationshipAgent()
        self.integrator = self.integrator or ModelIntegratorAgent()
        self.plantuml_agent = self.plantuml_agent or PlantUMLAgent()

    def run(self, requirements: str) -> Dict[str, Any]:
        concept_model = self.concept_agent.extract_concepts(requirements)
        attribute_model = self.attribute_specialist.enrich_model(concept_model, requirements)
        relationship_model = self.relationship_agent.add_relationships(attribute_model, requirements)
        integrated_model = self.integrator.normalize_model(relationship_model, requirements=requirements)
        plantuml = self.plantuml_agent.generate(integrated_model)
        return {
            "concept_model": concept_model,
            "attribute_model": attribute_model,
            "relationship_model": relationship_model,
            "model": integrated_model,
            "plantuml": plantuml,
        }


def run_pipeline(requirements: str, concept_entities: Optional[Any] = None) -> Dict[str, Any]:
    cascade = CascadeNOMAD()
    result = cascade.run(requirements)
    return {
        "entities": [uml_class.to_dict() for uml_class in result["model"].classes],
        "relationships": [relationship.to_dict() for relationship in result["model"].relationships],
        "plantuml": result["plantuml"],
    }
