from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from src.schemas import ValidationIssue, ValidationReport


TARGET_AGENT_BY_CATEGORY = {
    "class": "concept",
    "attribute": "attribute",
    "relationship": "relationship",
    "plantuml": "plantuml",
    "semantic": "integrator",
}

RELATIONSHIP_KEYWORDS = (
    "relationship",
    "relation",
    "association",
    "aggregat",
    "composition",
    "inheritance",
    "generalization",
    "multiplicity",
    "cardinality",
)

CLASS_KEYWORDS = (
    "missing class",
    "class missing",
    "missing entity",
    "class does not exist",
    "class not found",
    "unknown class",
    "undefined class",
    "nonexistent class",
    "no such class",
)


@dataclass
class RethinkRouteDecision:
    grouped_issues: Dict[str, List[ValidationIssue]] = field(default_factory=dict)
    corrections_to_apply: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "grouped_issues": {
                agent: [issue.to_dict() for issue in issues]
                for agent, issues in self.grouped_issues.items()
            },
            "corrections_to_apply": list(self.corrections_to_apply),
        }


class RethinkRouter:
    def route(self, report: ValidationReport) -> RethinkRouteDecision:
        grouped: Dict[str, List[ValidationIssue]] = {
            "concept": [],
            "attribute": [],
            "relationship": [],
            "plantuml": [],
            "integrator": [],
        }
        corrections: List[str] = []

        for issue in report.issues:
            target_agent = self._resolve_target_agent(issue)
            normalized_issue = self._with_target_agent(issue, target_agent)
            grouped.setdefault(target_agent, []).append(normalized_issue)
            if issue.severity.lower() in {"critical", "major"}:
                if target_agent not in corrections:
                    corrections.append(target_agent)

        grouped = {agent: issues for agent, issues in grouped.items() if issues}
        return RethinkRouteDecision(grouped_issues=grouped, corrections_to_apply=corrections)

    def _resolve_target_agent(self, issue: ValidationIssue) -> str:
        category = (issue.category or "").strip().lower()
        text = self._issue_text(issue)

        if category == "relationship":
            if self._mentions_class_missing(text) and not self._mentions_relationship(text):
                return "concept"
            return "relationship"

        if self._mentions_relationship(text):
            return "relationship"

        if category in TARGET_AGENT_BY_CATEGORY:
            return TARGET_AGENT_BY_CATEGORY[category]

        target_agent = (issue.target_agent or "").strip().lower()
        if target_agent in TARGET_AGENT_BY_CATEGORY.values():
            return target_agent

        return "integrator"

    def _issue_text(self, issue: ValidationIssue) -> str:
        parts = [
            issue.category or "",
            issue.message or "",
            issue.description or "",
            issue.evidence or "",
            issue.evidence_from_requirements or "",
            issue.suggested_fix or "",
            " ".join(issue.suggestions or []),
        ]
        return " ".join(part for part in parts if part).lower()

    def _mentions_relationship(self, text: str) -> bool:
        return any(keyword in text for keyword in RELATIONSHIP_KEYWORDS)

    def _mentions_class_missing(self, text: str) -> bool:
        return any(keyword in text for keyword in CLASS_KEYWORDS)

    def _with_target_agent(self, issue: ValidationIssue, target_agent: str) -> ValidationIssue:
        payload = issue.to_dict()
        payload["target_agent"] = target_agent
        payload["responsible_agent"] = target_agent
        return ValidationIssue.from_dict(payload)
