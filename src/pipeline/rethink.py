from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List, Optional

from src.agents.attribute_specialist import AttributeSpecialistAgent
from src.agents.concept_agent import ConceptAgent
from src.agents.model_integrator import ModelIntegratorAgent
from src.agents.plantuml_agent import PlantUMLAgent
from src.agents.relationship_agent import RelationshipAgent
from src.agents.rethink_router import RethinkRouter, RethinkRouteDecision
from src.agents.validator_agent import ValidatorAgent
from src.pipeline.orchestrator import CascadeNOMAD
from src.schemas import RethinkStep, UMLModel, ValidationReport


@dataclass
class RethinkNOMAD:
    concept_agent: Optional[ConceptAgent] = None
    attribute_specialist: Optional[AttributeSpecialistAgent] = None
    relationship_agent: Optional[RelationshipAgent] = None
    integrator: Optional[ModelIntegratorAgent] = None
    plantuml_agent: Optional[PlantUMLAgent] = None
    validator_agent: Optional[ValidatorAgent] = None
    router: Optional[RethinkRouter] = None
    max_rethink_iterations: int = 2

    def __post_init__(self) -> None:
        self.cascade = CascadeNOMAD(
            concept_agent=self.concept_agent,
            attribute_specialist=self.attribute_specialist,
            relationship_agent=self.relationship_agent,
            integrator=self.integrator,
            plantuml_agent=self.plantuml_agent,
        )
        self.concept_agent = self.cascade.concept_agent
        self.attribute_specialist = self.cascade.attribute_specialist
        self.relationship_agent = self.cascade.relationship_agent
        self.integrator = self.cascade.integrator
        self.plantuml_agent = self.cascade.plantuml_agent
        self.validator_agent = self.validator_agent or ValidatorAgent()
        self.router = self.router or RethinkRouter()

    def run(self, requirements: str) -> Dict[str, Any]:
        start = perf_counter()
        cascade_result = self.cascade.run(requirements)
        initial_model: UMLModel = cascade_result["model"]
        initial_plantuml: str = cascade_result["plantuml"]
        current_model: UMLModel = initial_model
        current_plantuml: str = initial_plantuml
        history: List[RethinkStep] = []
        llm_calls = self._estimate_llm_calls(self.cascade) + 1

        initial_report = self.validator_agent.validate(requirements, current_model, current_plantuml)
        llm_calls += 1
        current_report = initial_report
        history.append(
            RethinkStep(
                iteration=0,
                issues=list(initial_report.issues),
                routed_agents=self._serialize_routing(self.router.route(initial_report)),
                actions=[],
                validation_report=initial_report,
                llm_calls=llm_calls,
                runtime_seconds=perf_counter() - start,
            )
        )

        iteration = 0
        while iteration < self.max_rethink_iterations:
            routing = self.router.route(current_report)
            if not routing.corrections_to_apply:
                break

            current_model = self._apply_corrections(requirements, current_model, routing)
            current_plantuml = self.plantuml_agent.revise(current_model)
            llm_calls += 1
            current_report = self.validator_agent.validate(requirements, current_model, current_plantuml)
            llm_calls += 1
            iteration += 1
            history.append(
                RethinkStep(
                    iteration=iteration,
                    issues=list(current_report.issues),
                    routed_agents=self._serialize_routing(routing),
                    actions=self._build_actions(routing),
                    validation_report=current_report,
                    llm_calls=llm_calls,
                    runtime_seconds=perf_counter() - start,
                )
            )

            if current_report.is_valid:
                break

        return {
            "initial_model": initial_model,
            "initial_plantuml": initial_plantuml,
            "final_model": current_model,
            "generated_initial_plantuml": initial_plantuml,
            "generated_final_plantuml": current_plantuml,
            "model": current_model,
            "plantuml": current_plantuml,
            "validation_report_initial": initial_report,
            "validation_report": current_report,
            "validation_report_final": current_report,
            "rethink_history": history,
            "llm_calls": self._actual_llm_calls(),
            "runtime_seconds": perf_counter() - start,
        }

    def _apply_corrections(
        self,
        requirements: str,
        model: UMLModel,
        routing: RethinkRouteDecision,
    ) -> UMLModel:
        revised_model = model
        if "concept" in routing.corrections_to_apply:
            revised_model = self.concept_agent.extract_concepts(requirements)
        if "attribute" in routing.corrections_to_apply:
            revised_model = self.attribute_specialist.revise(revised_model, requirements)
        if "relationship" in routing.corrections_to_apply:
            revised_model = self.relationship_agent.revise(revised_model, requirements)
        if "integrator" in routing.corrections_to_apply:
            revised_model = self.integrator.normalize_model(revised_model, requirements=requirements)
        if "plantuml" in routing.corrections_to_apply:
            revised_model = self.integrator.normalize_model(revised_model, requirements=requirements)
        revised_model = self.integrator.normalize_model(revised_model, requirements=requirements)
        return revised_model

    def _serialize_routing(self, routing: RethinkRouteDecision) -> Dict[str, List[str]]:
        return {
            agent: [issue.issue_id for issue in issues]
            for agent, issues in routing.grouped_issues.items()
        }

    def _build_actions(self, routing: RethinkRouteDecision) -> List[Dict[str, Any]]:
        actions: List[Dict[str, Any]] = []
        for agent, issues in routing.grouped_issues.items():
            actions.append(
                {
                    "agent": agent,
                    "issue_ids": [issue.issue_id for issue in issues],
                    "should_revise": agent in routing.corrections_to_apply,
                }
            )
        return actions

    def _estimate_llm_calls(self, cascade: CascadeNOMAD) -> int:
        return 0

    def _actual_llm_calls(self) -> int:
        shared_llm = getattr(self.concept_agent, "llm", None) or getattr(self.validator_agent, "llm", None)
        if hasattr(shared_llm, "calls"):
            return int(shared_llm.calls)
        if hasattr(shared_llm, "prompts"):
            return len(shared_llm.prompts)
        return 0
