from __future__ import annotations

import json
import re
import logging
from typing import Any, Dict, List, Optional

from src.agents.base import BaseAgent
from src.schemas import UMLAttribute, UMLClass, UMLModel, UMLRelationship, ValidationIssue, ValidationReport
from src.utils.json_parser import parse_llm_json

logger = logging.getLogger(__name__)

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


class ValidatorAgent(BaseAgent):
    def __init__(
        self,
        llm: Optional[Any] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        super().__init__(
            name="ValidatorAgent",
            llm=llm,
            provider=provider,
            model=model,
            temperature=temperature,
        )

    def validate(self, requirements: str, model: UMLModel, plantuml: str) -> ValidationReport:
        prompt = self._build_prompt(requirements, model, plantuml)
        response = self.call_llm(prompt)
        try:
            payload = self._parse_response(response)
        except ValueError:
            retry_response = self.call_llm(
                prompt
                + "\n\nThe previous response was invalid. Return only valid JSON matching the requested schema."
            )
            payload = self._parse_response(retry_response)

        issues_payload = payload.get("issues", []) if isinstance(payload, dict) else []
        issues = [
            self._parse_issue(item)
            for item in issues_payload
            if isinstance(item, dict)
        ]
        issues.extend(self._heuristic_issues(requirements, model, plantuml))
        issues = self._dedupe_issues(issues)
        issues = [issue for issue in issues if issue.issue_id or issue.description]
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        return ValidationReport(
            requirements=requirements,
            issues=issues,
            validated_model=model,
            metadata=dict(metadata) if isinstance(metadata, dict) else {},
        )

    def _build_prompt(self, requirements: str, model: UMLModel, plantuml: str) -> str:
        return (
            "You are a UML validator.\n"
            "Compare the requirements, the intermediate UML model, and the PlantUML output.\n"
            "Return only JSON in the following format:\n"
            "{\n"
            '  "issues": [\n'
            "    {\n"
            '      "id": "issue-1",\n'
            '      "severity": "critical",\n'
            '      "category": "class",\n'
            '      "target_agent": "concept",\n'
            '      "description": "What is wrong",\n'
            '      "evidence_from_requirements": "Quoted evidence",\n'
            '      "suggested_fix": "What should change",\n'
            '      "target": "Optional target name"\n'
            "    }\n"
            "  ],\n"
            '  "metadata": {"validator_notes": "..."}\n'
            "}\n"
            f"Requirements:\n{requirements}\n"
            f"Intermediate model:\n{json.dumps(model.to_dict(), ensure_ascii=False)}\n"
            f"PlantUML:\n{plantuml}"
        )

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        try:
            return parse_llm_json(response_text)
        except (TypeError, ValueError) as exc:
            logger.exception("Failed to parse ValidatorAgent response as JSON.")
            raise ValueError("Failed to parse ValidatorAgent response as JSON.") from exc

    def _parse_issue(self, payload: Dict[str, Any]) -> ValidationIssue:
        suggestions = payload.get("suggestions")
        if suggestions is None and payload.get("suggested_fix"):
            suggestions = [payload.get("suggested_fix")]
        elif suggestions is None:
            suggestions = []
        issue = ValidationIssue.from_dict(
            {
                "issue_id": payload.get("id", payload.get("issue_id", "")),
                "id": payload.get("id"),
                "severity": payload.get("severity", "minor"),
                "category": payload.get("category", "semantic"),
                "message": payload.get("description", payload.get("message", "")),
                "description": payload.get("description"),
                "responsible_agent": payload.get("target_agent"),
                "target_agent": payload.get("target_agent"),
                "target": payload.get("target"),
                "evidence": payload.get("evidence_from_requirements"),
                "evidence_from_requirements": payload.get("evidence_from_requirements"),
                "suggestions": suggestions,
                "suggested_fix": payload.get("suggested_fix"),
                "is_critical": str(payload.get("severity", "minor")).lower() == "critical",
                "metadata": payload.get("metadata", {}),
            }
        )
        return self._normalize_issue_semantics(issue)

    def _normalize_issue_semantics(self, issue: ValidationIssue) -> ValidationIssue:
        text = self._issue_text(issue)
        target_agent = (issue.target_agent or "").strip().lower()
        category = (issue.category or "").strip().lower()

        if self._mentions_relationship(text):
            issue.category = "relationship"
            issue.target_agent = "relationship"
            issue.responsible_agent = "relationship"
            return issue

        if target_agent == "concept" and category == "class":
            issue.target_agent = "concept"
            issue.responsible_agent = "concept"
        return issue

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

    def _heuristic_issues(self, requirements: str, model: UMLModel, plantuml: str) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        normalized_requirements = self._normalize_text(requirements)

        for relationship in model.relationships:
            if relationship.source.strip().lower() == relationship.target.strip().lower():
                issues.append(
                    self._make_issue(
                        issue_id=f"heuristic-self-relationship-{relationship.source}-{relationship.target}",
                        severity="major",
                        category="relationship",
                        target_agent="relationship",
                        description=f"Unjustified self-relationship detected for {relationship.source}.",
                        evidence=f"{relationship.source} relates to itself in the model.",
                        suggested_fix="Remove the self-relationship unless explicitly required.",
                        target=relationship.source,
                    )
                )

        for uml_class in model.classes:
            issues.extend(self._extra_attribute_issues(uml_class, normalized_requirements))

        issues.extend(self._missing_generalization_issues(requirements, model))
        issues.extend(self._suspicious_plantuml_issues(plantuml))
        return issues

    def _extra_attribute_issues(self, uml_class: UMLClass, normalized_requirements: str) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        for attribute in uml_class.attributes:
            if not isinstance(attribute, UMLAttribute):
                continue
            normalized_attr = self._normalize_text(attribute.name)
            if not normalized_attr:
                continue
            if normalized_attr in normalized_requirements:
                continue
            if not self._looks_like_derived_or_structural_attribute(normalized_attr):
                continue
            issues.append(
                self._make_issue(
                    issue_id=f"heuristic-extra-attribute-{uml_class.name}-{attribute.name}",
                    severity="major",
                    category="attribute",
                    target_agent="attribute",
                    description=f"Extra attribute {uml_class.name}.{attribute.name} is not supported by the requirements.",
                    evidence=f"Attribute name '{attribute.name}' is not mentioned in the requirements.",
                    suggested_fix=f"Remove or justify {uml_class.name}.{attribute.name}.",
                    target=f"{uml_class.name}.{attribute.name}",
                )
            )
        return issues

    def _looks_like_derived_or_structural_attribute(self, normalized_attr: str) -> bool:
        suspicious_tokens = (
            "parent",
            "path",
            "filesystem",
            "fileystem",
            "sizeinbytes",
            "foldername",
            "parentfolder",
            "parentfoldername",
        )
        if normalized_attr in {"sizeinbytes", "parentfolder", "parentfoldername", "filesystempath"}:
            return True
        return any(token in normalized_attr for token in suspicious_tokens)

    def _missing_generalization_issues(self, requirements: str, model: UMLModel) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        text = " ".join(requirements.split())
        class_lookup = {self._normalize_text(uml_class.name): uml_class.name for uml_class in model.classes}

        for sentence in re.split(r"[.!?]\s*", text):
            lowered = sentence.lower().strip()
            if "can be" not in lowered:
                continue
            prefix, suffix = sentence.split("can be", 1)
            parent_phrase = prefix.strip(" ,:-")
            option_text = suffix.strip(" ,.;")
            option_parts = re.split(r"\bor\b", option_text, maxsplit=1, flags=re.IGNORECASE)
            if len(option_parts) < 2:
                continue

            option1_phrase = option_parts[0].strip(" ,.;")
            option2_phrase = option_parts[1].strip(" ,.;")

            parent_key = self._normalize_text(parent_phrase)
            parent = class_lookup.get(parent_key) or self._fuzzy_match(parent_key, class_lookup)
            child_one = class_lookup.get(self._normalize_text(option1_phrase)) or self._fuzzy_match(self._normalize_text(option1_phrase), class_lookup)
            child_two = class_lookup.get(self._normalize_text(option2_phrase)) or self._fuzzy_match(self._normalize_text(option2_phrase), class_lookup)

            expected_pairs = [(child_one, parent), (child_two, parent)]
            for child, parent_name in expected_pairs:
                if not child or not parent_name:
                    continue
                if self._normalize_text(child) == self._normalize_text(parent_name):
                    continue
                if not self._has_generalization(model.relationships, child, parent_name):
                    issues.append(
                        self._make_issue(
                            issue_id=f"heuristic-missing-generalization-{child}-{parent_name}",
                            severity="critical",
                            category="relationship",
                            target_agent="relationship",
                            description=f"Missing generalization from {child} to {parent_name}.",
                            evidence=f"The requirements state that {parent_phrase.strip()} can be {option1_phrase.strip()} or {option2_phrase.strip()}.",
                            suggested_fix=f"Add {child} --|> {parent_name}.",
                            target=f"{child}->{parent_name}",
                        )
                    )
        return issues

    def _suspicious_plantuml_issues(self, plantuml: str) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        for line in plantuml.splitlines():
            if re.match(r'^\s*"[^\"]+"\s+[A-Za-z_]\w*\s+(?:o--|\*--|--|<\|--|\.\.>)', line):
                issues.append(
                    self._make_issue(
                        issue_id=f"heuristic-plantuml-multiplicity-{self._normalize_text(line)[:32]}",
                        severity="major",
                        category="plantuml",
                        target_agent="plantuml",
                        description="Suspicious PlantUML syntax: multiplicity appears before the source class.",
                        evidence=line.strip(),
                        suggested_fix="Move the source class before the multiplicity token.",
                        target=line.strip(),
                    )
                )
            if re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*--\s*\1\b", line):
                issues.append(
                    self._make_issue(
                        issue_id=f"heuristic-self-association-{self._normalize_text(line)[:32]}",
                        severity="major",
                        category="relationship",
                        target_agent="relationship",
                        description="Self-relationship detected in PlantUML.",
                        evidence=line.strip(),
                        suggested_fix="Remove the self-relationship unless explicitly required.",
                        target=line.strip(),
                    )
                )
        return issues

    def _has_generalization(self, relationships: List[UMLRelationship], child: str, parent: str) -> bool:
        child_norm = self._normalize_text(child)
        parent_norm = self._normalize_text(parent)
        for relationship in relationships:
            if relationship.relationship_type.lower() not in {"inheritance", "generalization", "extends"}:
                continue
            if self._normalize_text(relationship.source) == child_norm and self._normalize_text(relationship.target) == parent_norm:
                return True
        return False

    def _fuzzy_match(self, phrase: str, class_lookup: Dict[str, str]) -> Optional[str]:
        normalized_phrase = self._normalize_text(phrase)
        if normalized_phrase in class_lookup:
            return class_lookup[normalized_phrase]
        singular = normalized_phrase[:-1] if normalized_phrase.endswith("s") else normalized_phrase
        if singular in class_lookup:
            return class_lookup[singular]
        return None

    def _dedupe_issues(self, issues: List[ValidationIssue]) -> List[ValidationIssue]:
        deduped: List[ValidationIssue] = []
        seen = set()
        for issue in issues:
            key = (
                issue.category.lower(),
                issue.target_agent.lower() if issue.target_agent else "",
                self._normalize_text(issue.target or ""),
                self._normalize_text(issue.description or issue.message or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(issue)
        return deduped

    def _make_issue(
        self,
        *,
        issue_id: str,
        severity: str,
        category: str,
        target_agent: str,
        description: str,
        evidence: str,
        suggested_fix: str,
        target: str,
    ) -> ValidationIssue:
        return ValidationIssue.from_dict(
            {
                "issue_id": issue_id,
                "id": issue_id,
                "severity": severity,
                "category": category,
                "message": description,
                "description": description,
                "responsible_agent": target_agent,
                "target_agent": target_agent,
                "target": target,
                "evidence": evidence,
                "evidence_from_requirements": evidence,
                "suggestions": [suggested_fix],
                "suggested_fix": suggested_fix,
                "is_critical": severity.lower() == "critical",
                "metadata": {"source": "heuristic"},
            }
        )

    def _normalize_text(self, text: str) -> str:
        return "".join(ch.lower() for ch in text if ch.isalnum())
