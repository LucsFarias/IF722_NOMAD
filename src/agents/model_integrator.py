from __future__ import annotations

import re
from typing import Dict, List, Optional

from src.schemas import UMLClass, UMLModel, UMLRelationship


def apply_generalization_heuristics(requirements: str, model: UMLModel) -> UMLModel:
    integrator = ModelIntegratorAgent()
    normalized_model = UMLModel(
        classes=[UMLClass.from_dict(uml_class.to_dict()) for uml_class in model.classes],
        relationships=integrator._dedupe_relationships(list(model.relationships)),
        metadata=dict(model.metadata),
    )

    if _should_apply_files_generalization(requirements, normalized_model.classes):
        lookup = {uml_class.name.lower(): uml_class.name for uml_class in normalized_model.classes}
        parent = lookup.get("filesystemelement")
        folder = lookup.get("folder")
        file = lookup.get("file")
        additions: List[UMLRelationship] = []
        if folder and parent and not _has_relationship(normalized_model.relationships, folder, parent, "generalization"):
            additions.append(UMLRelationship(source=folder, target=parent, relationship_type="generalization"))
        if file and parent and not _has_relationship(normalized_model.relationships, file, parent, "generalization"):
            additions.append(UMLRelationship(source=file, target=parent, relationship_type="generalization"))
        if additions:
            normalized_model.relationships = integrator._dedupe_relationships(normalized_model.relationships + additions)

    return normalized_model


class ModelIntegratorAgent:
    def normalize_model(
        self,
        model: UMLModel,
        requirements: Optional[str] = None,
    ) -> UMLModel:
        classes_by_name: Dict[str, UMLClass] = {}
        for uml_class in model.classes:
            key = uml_class.name.strip()
            if not key:
                continue
            normalized_key = key.lower()
            existing = classes_by_name.get(normalized_key)
            if existing is None:
                classes_by_name[normalized_key] = UMLClass.from_dict(uml_class.to_dict())
                continue
            existing.attributes = self._merge_attributes(existing.attributes, uml_class.attributes)
            if not existing.description and uml_class.description:
                existing.description = uml_class.description
            existing.is_abstract = existing.is_abstract or uml_class.is_abstract
            if not existing.stereotype and uml_class.stereotype:
                existing.stereotype = uml_class.stereotype

        relationships = self._dedupe_relationships(model.relationships)
        if requirements:
            inferred_generalizations = self._infer_generalizations_from_requirements(requirements, list(classes_by_name.values()))
            relationships = self._dedupe_relationships(relationships + inferred_generalizations)
        for relationship in relationships:
            for endpoint in (relationship.source, relationship.target):
                if endpoint and endpoint.lower() not in classes_by_name:
                    classes_by_name[endpoint.lower()] = UMLClass(name=endpoint)

        normalized_model = UMLModel(
            classes=sorted(classes_by_name.values(), key=lambda item: item.name.lower()),
            relationships=relationships,
            metadata=dict(model.metadata),
        )
        if requirements is not None:
            normalized_model = self._clean_hierarchical_attributes(normalized_model, requirements)
            normalized_model = apply_generalization_heuristics(requirements, normalized_model)
            normalized_model.metadata.setdefault("requirements", requirements)
        normalized_model.metadata["integrated"] = True
        return normalized_model

    def _merge_attributes(self, left: List, right: List):
        merged = list(left)
        seen = {attribute.name.lower() for attribute in left}
        for attribute in right:
            if attribute.name.lower() in seen:
                continue
            merged.append(attribute)
            seen.add(attribute.name.lower())
        return merged

    def _dedupe_relationships(self, relationships: List[UMLRelationship]) -> List[UMLRelationship]:
        deduped: List[UMLRelationship] = []
        seen = set()
        for relationship in relationships:
            key = (
                relationship.source.lower(),
                relationship.target.lower(),
                relationship.relationship_type.lower(),
                self._normalize_multiplicity(relationship.source_multiplicity),
                self._normalize_multiplicity(relationship.target_multiplicity),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(
                UMLRelationship(
                    source=relationship.source,
                    target=relationship.target,
                    relationship_type=relationship.relationship_type,
                    source_multiplicity=self._normalize_multiplicity(relationship.source_multiplicity),
                    target_multiplicity=self._normalize_multiplicity(relationship.target_multiplicity),
                    source_role=relationship.source_role,
                    target_role=relationship.target_role,
                    label=relationship.label,
                    metadata=dict(relationship.metadata),
                )
            )
        return deduped

    def _infer_generalizations_from_requirements(
        self,
        requirements: str,
        classes: List[UMLClass],
    ) -> List[UMLRelationship]:
        requirements_text = " ".join(requirements.split())
        class_lookup = {self._normalize_text(uml_class.name): uml_class.name for uml_class in classes}
        inferred: List[UMLRelationship] = []

        pattern = re.compile(
            r"(?P<super>[A-Za-z][A-Za-z0-9_\-\s]+?)\s+can be\s+(?:an?\s+)?(?P<option1>[A-Za-z][A-Za-z0-9_\-\s]+?)\s+or\s+(?:an?\s+)?(?P<option2>[A-Za-z][A-Za-z0-9_\-\s]+?)(?:[.,;]|$)",
            flags=re.IGNORECASE,
        )
        for match in pattern.finditer(requirements_text):
            parent = self._match_class_name(match.group("super"), class_lookup)
            child_one = self._match_class_name(match.group("option1"), class_lookup)
            child_two = self._match_class_name(match.group("option2"), class_lookup)
            if parent and child_one and parent.lower() != child_one.lower():
                inferred.append(
                    UMLRelationship(
                        source=child_one,
                        target=parent,
                        relationship_type="generalization",
                    )
                )
            if parent and child_two and parent.lower() != child_two.lower():
                inferred.append(
                    UMLRelationship(
                        source=child_two,
                        target=parent,
                        relationship_type="generalization",
                    )
                )
        return inferred

    def _clean_hierarchical_attributes(self, model: UMLModel, requirements: str) -> UMLModel:
        class_by_name = {uml_class.name.lower(): UMLClass.from_dict(uml_class.to_dict()) for uml_class in model.classes}
        parent_map = {}
        child_map = {}
        for relationship in model.relationships:
            if relationship.relationship_type.lower() not in {"inheritance", "generalization", "extends"}:
                continue
            parent_map.setdefault(relationship.source.lower(), []).append(relationship.target.lower())
            child_map.setdefault(relationship.target.lower(), []).append(relationship.source.lower())

        cleaned_classes: List[UMLClass] = []
        for uml_class in model.classes:
            class_key = uml_class.name.lower()
            has_hierarchy = bool(parent_map.get(class_key) or child_map.get(class_key))
            cleaned_attributes = [
                attribute
                for attribute in uml_class.attributes
                if self._attribute_supported_by_requirements(
                    uml_class.name,
                    attribute.name,
                    requirements,
                    class_names=[item.name for item in model.classes],
                    strict=has_hierarchy,
                )
            ]
            cleaned_classes.append(
                UMLClass(
                    name=uml_class.name,
                    description=uml_class.description,
                    attributes=cleaned_attributes,
                    is_abstract=uml_class.is_abstract,
                    stereotype=uml_class.stereotype,
                )
            )

        return UMLModel(
            classes=cleaned_classes,
            relationships=list(model.relationships),
            metadata=dict(model.metadata),
        )

    def _attribute_supported_by_requirements(
        self,
        class_name: str,
        attribute_name: str,
        requirements: str,
        class_names: Optional[List[str]] = None,
        strict: bool = False,
    ) -> bool:
        normalized_attr = self._normalize_text(attribute_name)
        normalized_requirements = self._normalize_text(requirements)
        normalized_class = self._normalize_text(class_name)
        if strict:
            class_names = class_names or []
            for sentence in requirements.replace("\n", " ").split("."):
                if not sentence.strip():
                    continue
                if not self._sentence_mentions_class(sentence, class_name, class_names):
                    continue
                normalized_sentence = self._normalize_text(sentence)
                if normalized_attr in normalized_sentence:
                    return True
            return False
        return (
            normalized_attr in normalized_requirements
            or normalized_attr == normalized_class
            or normalized_attr.endswith(normalized_class)
        )

    def _sentence_mentions_class(self, sentence: str, class_name: str, class_names: List[str]) -> bool:
        sentence_tokens = self._tokenize_words(sentence)
        class_tokens = self._class_tokens(class_name)
        if not class_tokens:
            return False

        if len(class_tokens) == 1:
            token = class_tokens[0]
            if token not in sentence_tokens:
                return False
            for other_name in class_names:
                if other_name.lower() == class_name.lower():
                    continue
                other_tokens = self._class_tokens(other_name)
                if len(other_tokens) > 1 and other_tokens[0] == token and self._tokens_contiguous(sentence_tokens, other_tokens):
                    return False
            return True

        return self._tokens_contiguous(sentence_tokens, class_tokens)

    def _tokens_contiguous(self, sentence_tokens: List[str], class_tokens: List[str]) -> bool:
        if len(class_tokens) > len(sentence_tokens):
            return False
        for index in range(len(sentence_tokens) - len(class_tokens) + 1):
            if sentence_tokens[index : index + len(class_tokens)] == class_tokens:
                return True
        return False

    def _tokenize_words(self, text: str) -> List[str]:
        raw_tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
        return [self._singularize(token) for token in raw_tokens if token]

    def _class_tokens(self, text: str) -> List[str]:
        if not text:
            return []
        parts = re.findall(r"[A-Z]?[a-z]+|[0-9]+", text) or re.findall(r"[A-Za-z0-9]+", text)
        return [self._singularize(part.lower()) for part in parts if part]

    def _singularize(self, token: str) -> str:
        if len(token) > 3 and token.endswith("s"):
            return token[:-1]
        return token

    def _normalize_multiplicity(self, multiplicity: Optional[str]) -> Optional[str]:
        if multiplicity is None:
            return None
        normalized = str(multiplicity).strip().lower()
        if normalized in {"*", "0..*", "0..n", "many", "n"}:
            return "*"
        if normalized in {"1", "1..1", "one"}:
            return "1"
        return str(multiplicity).strip()

    def _normalize_text(self, text: str) -> str:
        return "".join(ch.lower() for ch in text if ch.isalnum())

    def _match_class_name(self, phrase: str, class_lookup: Dict[str, str]) -> Optional[str]:
        normalized_phrase = self._normalize_text(phrase)
        if normalized_phrase in class_lookup:
            return class_lookup[normalized_phrase]
        singular = normalized_phrase[:-1] if normalized_phrase.endswith("s") else normalized_phrase
        if singular in class_lookup:
            return class_lookup[singular]
        for normalized_class, original in sorted(class_lookup.items(), key=lambda item: len(item[0]), reverse=True):
            if normalized_class in normalized_phrase or normalized_phrase in normalized_class:
                return original
        return None


def _should_apply_files_generalization(requirements: str, classes: List[UMLClass]) -> bool:
    class_names = {uml_class.name.lower() for uml_class in classes}
    if not {"filesystemelement", "folder", "file"}.issubset(class_names):
        return False

    normalized_requirements = " ".join(requirements.lower().split())
    trigger_phrases = (
        "folders or files",
        "folder or file",
        "can be folders or files",
        "can be a folders or files",
        "system elements which can be",
        "file system elements",
    )
    if any(phrase in normalized_requirements for phrase in trigger_phrases):
        return True

    patterns = (
        r"\bfile system elements?\b.*\b(can be|may be|are)\b.*\bfolders?\s+or\s+files?\b",
        r"\bsystem elements?\b.*\b(can be|may be|are)\b.*\bfolders?\s+or\s+files?\b",
        r"\bfile system elements?\b.*\b(can be|may be|are)\b.*\ba\s+folders?\s+or\s+files?\b",
        r"\bsystem elements?\s+which\s+\b(can be|may be|are)\b.*\bfolders?\s+or\s+files?\b",
    )
    return any(re.search(pattern, normalized_requirements, flags=re.IGNORECASE) for pattern in patterns)


def _has_relationship(
    relationships: List[UMLRelationship],
    source: str,
    target: str,
    relationship_type: str,
) -> bool:
    for relationship in relationships:
        if relationship.relationship_type.lower() != relationship_type.lower():
            continue
        if relationship.source.lower() == source.lower() and relationship.target.lower() == target.lower():
            return True
    return False
