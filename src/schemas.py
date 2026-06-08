from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class UMLAttribute:
    name: str
    data_type: Optional[str] = None
    confidence: Optional[float] = None
    visibility: Optional[str] = None
    default_value: Optional[str] = None
    multiplicity: Optional[str] = None
    stereotype: Optional[str] = None
    source: str = "inferred"

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["inferred_type"] = self.data_type
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UMLAttribute":
        data_type = data.get("data_type", data.get("inferred_type", data.get("type")))
        return cls(
            name=data.get("name", ""),
            data_type=data_type,
            confidence=data.get("confidence"),
            visibility=data.get("visibility"),
            default_value=data.get("default_value"),
            multiplicity=data.get("multiplicity"),
            stereotype=data.get("stereotype"),
            source=data.get("source", "inferred"),
        )


@dataclass
class UMLClass:
    name: str
    description: Optional[str] = None
    attributes: List[UMLAttribute] = field(default_factory=list)
    is_abstract: bool = False
    stereotype: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["attributes"] = [attribute.to_dict() for attribute in self.attributes]
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UMLClass":
        attributes = [
            UMLAttribute.from_dict(attribute)
            for attribute in data.get("attributes", [])
            if isinstance(attribute, dict)
        ]
        return cls(
            name=data.get("name", ""),
            description=data.get("description"),
            attributes=attributes,
            is_abstract=bool(data.get("is_abstract", False)),
            stereotype=data.get("stereotype"),
        )


@dataclass
class UMLRelationship:
    source: str
    target: str
    relationship_type: str
    source_multiplicity: Optional[str] = None
    target_multiplicity: Optional[str] = None
    source_role: Optional[str] = None
    target_role: Optional[str] = None
    label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UMLRelationship":
        return cls(
            source=data.get("source", ""),
            target=data.get("target", ""),
            relationship_type=data.get("relationship_type", ""),
            source_multiplicity=data.get("source_multiplicity"),
            target_multiplicity=data.get("target_multiplicity"),
            source_role=data.get("source_role"),
            target_role=data.get("target_role"),
            label=data.get("label"),
            metadata=dict(data.get("metadata", {})) if isinstance(data.get("metadata", {}), dict) else {},
        )


@dataclass
class UMLModel:
    classes: List[UMLClass] = field(default_factory=list)
    relationships: List[UMLRelationship] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "classes": [uml_class.to_dict() for uml_class in self.classes],
            "relationships": [relationship.to_dict() for relationship in self.relationships],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UMLModel":
        classes = [
            UMLClass.from_dict(uml_class)
            for uml_class in data.get("classes", [])
            if isinstance(uml_class, dict)
        ]
        relationships = [
            UMLRelationship.from_dict(relationship)
            for relationship in data.get("relationships", [])
            if isinstance(relationship, dict)
        ]
        metadata = data.get("metadata", {})
        return cls(classes=classes, relationships=relationships, metadata=dict(metadata) if isinstance(metadata, dict) else {})


@dataclass
class ValidationIssue:
    issue_id: str
    severity: str
    category: str
    message: str
    responsible_agent: Optional[str] = None
    target: Optional[str] = None
    evidence: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_critical: bool = False
    id: Optional[str] = None
    target_agent: Optional[str] = None
    description: Optional[str] = None
    evidence_from_requirements: Optional[str] = None
    suggested_fix: Optional[str] = None

    def __post_init__(self) -> None:
        if self.id is None:
            self.id = self.issue_id
        if self.description is None:
            self.description = self.message
        if self.evidence_from_requirements is None:
            self.evidence_from_requirements = self.evidence
        if self.target_agent is None:
            self.target_agent = self.responsible_agent
        if self.suggested_fix is None and self.suggestions:
            self.suggested_fix = self.suggestions[0]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["issue_id"] = self.issue_id
        payload["message"] = self.message
        payload["responsible_agent"] = self.responsible_agent
        payload["evidence"] = self.evidence
        payload["suggestions"] = list(self.suggestions)
        payload["id"] = self.id or self.issue_id
        payload["target_agent"] = self.target_agent or self.responsible_agent
        payload["description"] = self.description or self.message
        payload["evidence_from_requirements"] = self.evidence_from_requirements or self.evidence
        payload["suggested_fix"] = self.suggested_fix or (self.suggestions[0] if self.suggestions else None)
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationIssue":
        suggestions = data.get("suggestions", [])
        metadata = data.get("metadata", {})
        return cls(
            issue_id=str(data.get("issue_id", data.get("id", ""))),
            severity=str(data.get("severity", "minor")),
            category=str(data.get("category", "")),
            message=str(data.get("message", data.get("description", ""))),
            responsible_agent=data.get("responsible_agent", data.get("target_agent")),
            target=data.get("target"),
            evidence=data.get("evidence", data.get("evidence_from_requirements")),
            suggestions=list(suggestions) if isinstance(suggestions, list) else [],
            metadata=dict(metadata) if isinstance(metadata, dict) else {},
            is_critical=bool(data.get("is_critical", False)),
            id=data.get("id"),
            target_agent=data.get("target_agent", data.get("responsible_agent")),
            description=data.get("description", data.get("message")),
            evidence_from_requirements=data.get("evidence_from_requirements", data.get("evidence")),
            suggested_fix=data.get("suggested_fix"),
        )


@dataclass
class ValidationReport:
    requirements: Optional[str] = None
    issues: List[ValidationIssue] = field(default_factory=list)
    validated_model: Optional[UMLModel] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requirements": self.requirements,
            "issues": [issue.to_dict() for issue in self.issues],
            "validated_model": self.validated_model.to_dict() if self.validated_model else None,
            "metadata": dict(self.metadata),
            "summary": {
                "issue_count": len(self.issues),
                "critical_issue_count": len(self.critical_issues),
                "noncritical_issue_count": len(self.noncritical_issues),
            },
            "is_valid": self.is_valid,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationReport":
        issues = [
            ValidationIssue.from_dict(issue)
            for issue in data.get("issues", [])
            if isinstance(issue, dict)
        ]
        validated_model = data.get("validated_model")
        metadata = data.get("metadata", {})
        return cls(
            requirements=data.get("requirements"),
            issues=issues,
            validated_model=UMLModel.from_dict(validated_model) if isinstance(validated_model, dict) else None,
            metadata=dict(metadata) if isinstance(metadata, dict) else {},
        )

    @property
    def critical_issues(self) -> List[ValidationIssue]:
        return [issue for issue in self.issues if issue.is_critical or issue.severity.lower() == "critical"]

    @property
    def noncritical_issues(self) -> List[ValidationIssue]:
        return [issue for issue in self.issues if issue not in self.critical_issues]

    @property
    def is_valid(self) -> bool:
        return len(self.critical_issues) == 0


@dataclass
class RethinkStep:
    iteration: int
    issues: List[ValidationIssue] = field(default_factory=list)
    routed_agents: Dict[str, List[str]] = field(default_factory=dict)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    validation_report: Optional[ValidationReport] = None
    llm_calls: int = 0
    runtime_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "issues": [issue.to_dict() for issue in self.issues],
            "routed_agents": {key: list(value) for key, value in self.routed_agents.items()},
            "actions": list(self.actions),
            "validation_report": self.validation_report.to_dict() if self.validation_report else None,
            "llm_calls": self.llm_calls,
            "runtime_seconds": self.runtime_seconds,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RethinkStep":
        issues = [
            ValidationIssue.from_dict(issue)
            for issue in data.get("issues", [])
            if isinstance(issue, dict)
        ]
        validation_report = data.get("validation_report")
        return cls(
            iteration=int(data.get("iteration", 0)),
            issues=issues,
            routed_agents={key: list(value) for key, value in data.get("routed_agents", {}).items()} if isinstance(data.get("routed_agents", {}), dict) else {},
            actions=list(data.get("actions", [])) if isinstance(data.get("actions", []), list) else [],
            validation_report=ValidationReport.from_dict(validation_report) if isinstance(validation_report, dict) else None,
            llm_calls=int(data.get("llm_calls", 0)),
            runtime_seconds=data.get("runtime_seconds"),
            metadata=dict(data.get("metadata", {})) if isinstance(data.get("metadata", {}), dict) else {},
        )


@dataclass
class ExperimentResult:
    case_name: str
    mode: str
    provider: str
    model: str
    input_requirements: str
    gold_puml: str
    generated_initial_puml: str
    generated_final_puml: str
    intermediate_model: UMLModel
    validation_report_initial: ValidationReport
    validation_report_final: ValidationReport
    rethink_history: List[RethinkStep] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    runtime_seconds: float = 0.0
    llm_calls: int = 0
    output_dir: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_name": self.case_name,
            "mode": self.mode,
            "provider": self.provider,
            "model": self.model,
            "input_requirements": self.input_requirements,
            "gold_puml": self.gold_puml,
            "generated_initial_puml": self.generated_initial_puml,
            "generated_final_puml": self.generated_final_puml,
            "intermediate_model": self.intermediate_model.to_dict(),
            "validation_report_initial": self.validation_report_initial.to_dict(),
            "validation_report_final": self.validation_report_final.to_dict(),
            "rethink_history": [step.to_dict() for step in self.rethink_history],
            "metrics": dict(self.metrics),
            "runtime_seconds": self.runtime_seconds,
            "llm_calls": self.llm_calls,
            "output_dir": self.output_dir,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentResult":
        return cls(
            case_name=data.get("case_name", ""),
            mode=data.get("mode", ""),
            provider=data.get("provider", ""),
            model=data.get("model", ""),
            input_requirements=data.get("input_requirements", ""),
            gold_puml=data.get("gold_puml", ""),
            generated_initial_puml=data.get("generated_initial_puml", ""),
            generated_final_puml=data.get("generated_final_puml", ""),
            intermediate_model=UMLModel.from_dict(data.get("intermediate_model", {})),
            validation_report_initial=ValidationReport.from_dict(data.get("validation_report_initial", {})),
            validation_report_final=ValidationReport.from_dict(data.get("validation_report_final", {})),
            rethink_history=[
                RethinkStep.from_dict(step)
                for step in data.get("rethink_history", [])
                if isinstance(step, dict)
            ],
            metrics=dict(data.get("metrics", {})) if isinstance(data.get("metrics", {}), dict) else {},
            runtime_seconds=float(data.get("runtime_seconds", 0.0) or 0.0),
            llm_calls=int(data.get("llm_calls", 0)),
            output_dir=data.get("output_dir"),
            metadata=dict(data.get("metadata", {})) if isinstance(data.get("metadata", {}), dict) else {},
        )
