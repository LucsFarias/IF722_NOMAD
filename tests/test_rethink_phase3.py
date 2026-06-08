from types import SimpleNamespace

from src.agents.attribute_specialist import AttributeSpecialistAgent
from src.agents.concept_agent import ConceptAgent
from src.agents.model_integrator import ModelIntegratorAgent, apply_generalization_heuristics
from src.agents.relationship_agent import RelationshipAgent
from src.agents.rethink_router import RethinkRouter
from src.agents.validator_agent import ValidatorAgent
from src.pipeline import RethinkNOMAD
from src.schemas import ValidationIssue, ValidationReport


class SequencedFakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        response = self.responses.pop(0)
        return SimpleNamespace(content=response)


def test_validator_agent_parses_fake_report():
    fake_llm = SequencedFakeLLM(
        [
            "```json\n"
            '{"issues": ['
            '{'
            '"id": "i-1", "severity": "critical", "category": "relationship", "target_agent": "relationship", '
            '"description": "Missing relationship", "evidence_from_requirements": "Customers place orders", '
            '"suggested_fix": "Add Customer to Order association"'
            '}'
            '], "metadata": {"source": "fake"}}'
            "\n```"
        ]
    )

    validator = ValidatorAgent(llm=fake_llm)
    report = validator.validate("Customers place orders.", _fake_model(), "@startuml\n@enduml")

    assert isinstance(report, ValidationReport)
    assert len(report.issues) == 1
    issue = report.issues[0]
    assert issue.issue_id == "i-1"
    assert issue.target_agent == "relationship"
    assert issue.description == "Missing relationship"
    assert issue.suggested_fix == "Add Customer to Order association"


def test_validator_agent_normalizes_misclassified_relationship_issue_origin():
    fake_llm = SequencedFakeLLM(
        [
            "```json\n"
            '{"issues": ['
            '{'
            '"id": "i-rel", "severity": "critical", "category": "class", "target_agent": "concept", '
            '"description": "Missing relationship between Folder and FileSystemElement", '
            '"evidence_from_requirements": "Folders aggregate file system elements", '
            '"suggested_fix": "Add Folder aggregation FileSystemElement relationship"'
            '}'
            '], "metadata": {"source": "fake"}}'
            "\n```"
        ]
    )

    validator = ValidatorAgent(llm=fake_llm)
    report = validator.validate("Folders aggregate file system elements.", _fake_model(), "@startuml\n@enduml")

    assert len(report.issues) == 1
    issue = report.issues[0]
    assert issue.category == "relationship"
    assert issue.target_agent == "relationship"
    assert issue.responsible_agent == "relationship"


def test_rethink_router_groups_issues_and_prioritizes_major_or_critical():
    report = ValidationReport(
        issues=[
            ValidationIssue(
                issue_id="i-1",
                severity="critical",
                category="relationship",
                target_agent="relationship",
                message="Missing relationship",
            ),
            ValidationIssue(
                issue_id="i-2",
                severity="minor",
                category="plantuml",
                target_agent="plantuml",
                message="Stylistic issue",
            ),
            ValidationIssue(
                issue_id="i-3",
                severity="major",
                category="attribute",
                target_agent="attribute",
                message="Wrong attribute",
            ),
        ]
    )

    decision = RethinkRouter().route(report)

    assert set(decision.grouped_issues) == {"relationship", "plantuml", "attribute"}
    assert decision.corrections_to_apply == ["relationship", "attribute"]
    assert [issue.issue_id for issue in decision.grouped_issues["plantuml"]] == ["i-2"]


def test_rethink_router_overrides_misclassified_relationship_issue():
    report = ValidationReport(
        issues=[
            ValidationIssue(
                issue_id="i-rel",
                severity="critical",
                category="class",
                target_agent="concept",
                message="Missing relationship between Folder and FileSystemElement",
                suggested_fix="Add Folder aggregation FileSystemElement relationship",
                evidence_from_requirements="Folders contain files and relate to the filesystem element hierarchy",
            )
        ]
    )

    decision = RethinkRouter().route(report)

    assert "relationship" in decision.grouped_issues
    routed_issue = decision.grouped_issues["relationship"][0]
    assert routed_issue.target_agent == "relationship"
    assert routed_issue.responsible_agent == "relationship"
    assert decision.corrections_to_apply == ["relationship"]


def test_validator_agent_detects_suspicious_plantuml_syntax():
    fake_llm = SequencedFakeLLM(
        [
            "```json\n"
            '{"issues": [], "metadata": {"source": "fake"}}'
            "\n```"
        ]
    )

    validator = ValidatorAgent(llm=fake_llm)
    from src.schemas import UMLClass, UMLModel

    report = validator.validate(
        "Folder aggregates FileSystemElement.",
        UMLModel(classes=[UMLClass(name="Folder"), UMLClass(name="FileSystemElement")], relationships=[]),
        '@startuml\n"1" Folder o-- "*" FileSystemElement\n@enduml',
    )

    assert any(issue.category == "plantuml" for issue in report.issues)
    assert any("multiplicity" in issue.description.lower() for issue in report.issues)


def test_validator_agent_reports_missing_generalizations_for_files_case():
    fake_llm = SequencedFakeLLM(
        [
            "```json\n"
            '{"issues": [], "metadata": {"source": "fake"}}'
            "\n```"
        ]
    )

    validator = ValidatorAgent(llm=fake_llm)
    from src.schemas import UMLAttribute, UMLClass, UMLModel, UMLRelationship

    model = UMLModel(
        classes=[
            UMLClass(
                name="FileSystemElement",
                attributes=[UMLAttribute(name="name", data_type="String")],
            ),
            UMLClass(
                name="File",
                attributes=[
                    UMLAttribute(name="extension", data_type="String"),
                    UMLAttribute(name="size", data_type="Integer"),
                    UMLAttribute(name="name", data_type="String"),
                ],
            ),
            UMLClass(name="Folder"),
        ],
        relationships=[
            UMLRelationship(
                source="Folder",
                target="FileSystemElement",
                relationship_type="aggregation",
                source_multiplicity="1",
                target_multiplicity="*",
            )
        ],
    )

    report = validator.validate(
        "File system elements can be folders or files. "
        "All file system elements have a name. "
        "Every file system element belongs to a folder. "
        "Folders aggregate any number of file system elements. "
        "Files have extension and size.",
        model,
        '@startuml\nFolder "1" o-- "*" FileSystemElement\n@enduml',
    )

    generalization_issues = [issue for issue in report.issues if "generalization" in issue.description.lower()]
    assert len(generalization_issues) >= 2
    assert all(issue.category == "relationship" for issue in generalization_issues)
    assert all(issue.target_agent == "relationship" for issue in generalization_issues)
    assert not any("missing relationship" in issue.description.lower() and issue.category == "class" for issue in report.issues)


def test_rethink_nomad_handles_files_case_and_preserves_expected_structure(monkeypatch):
    concept_llm = SequencedFakeLLM(
        [
            "```json\n"
            '{"classes": ['
            '{"name": "FileSystemElement", "description": "Base element", "attributes": []},'
            '{"name": "Folder", "description": "A folder", "attributes": []},'
            '{"name": "File", "description": "A file", "attributes": []}'
            ']}'
            "\n```",
        ]
    )
    attribute_llm = SequencedFakeLLM(
        [
            "```json\n"
            '{"attributes": [{"name": "name", "data_type": "String", "confidence": 0.8}, {"name": "parent_folder_name", "data_type": "String", "confidence": 0.6}, {"name": "size_in_bytes", "data_type": "Integer", "confidence": 0.6}]}'
            "\n```",
            "```json\n"
            '{"attributes": []}'
            "\n```",
            "```json\n"
            '{"attributes": [{"name": "extension", "data_type": "String", "confidence": 0.9}, {"name": "size", "data_type": "Integer", "confidence": 0.9}, {"name": "parent_folder", "data_type": "String", "confidence": 0.4}, {"name": "file_system_path", "data_type": "String", "confidence": 0.4}]}'
            "\n```",
            "```json\n"
            '{"attributes": [{"name": "extension", "data_type": "String", "confidence": 0.9}, {"name": "size", "data_type": "Integer", "confidence": 0.9}]}'
            "\n```",
            "```json\n"
            '{"attributes": [{"name": "name", "data_type": "String", "confidence": 0.8}]}'
            "\n```",
            "```json\n"
            '{"attributes": []}'
            "\n```",
            "```json\n"
            '{"attributes": []}'
            "\n```",
            "```json\n"
            '{"attributes": []}'
            "\n```",
            "```json\n"
            '{"attributes": []}'
            "\n```",
        ]
    )
    relationship_llm = SequencedFakeLLM(
        [
            "```json\n"
            '{"relationships": []}'
            "\n```",
            "```json\n"
            '{"relationships": []}'
            "\n```",
            "```json\n"
            '{"relationships": []}'
            "\n```",
            "```json\n"
            '{"relationships": []}'
            "\n```",
        ]
    )
    validator_llm = SequencedFakeLLM(
        [
            "```json\n"
            '{"issues": [], "metadata": {}}'
            "\n```",
            "```json\n"
            '{"issues": [], "metadata": {"status": "clean"}}'
            "\n```",
            "```json\n"
            '{"issues": [], "metadata": {"status": "clean"}}'
            "\n```",
            "```json\n"
            '{"issues": [], "metadata": {"status": "clean"}}'
            "\n```",
        ]
    )

    rethink = RethinkNOMAD(
        concept_agent=ConceptAgent(llm=concept_llm),
        attribute_specialist=AttributeSpecialistAgent(llm=attribute_llm),
        relationship_agent=RelationshipAgent(llm=relationship_llm),
        validator_agent=ValidatorAgent(llm=validator_llm),
        max_rethink_iterations=2,
    )

    requirements = (
        "File system elements can be folders or files. "
        "All file system elements have a name. "
        "Every file system element belongs to a folder. "
        "Folders aggregate any number of file system elements. "
        "Files have extension and size."
    )

    result = rethink.run(requirements)

    assert len(result["rethink_history"]) >= 1
    assert result["validation_report"].is_valid is True
    assert result["model"].relationships
    assert 'Folder --|> FileSystemElement' in result["generated_final_plantuml"]
    assert 'File --|> FileSystemElement' in result["generated_final_plantuml"]
    assert 'Folder "1" o-- "*" FileSystemElement' in result["generated_final_plantuml"]
    assert result["generated_final_plantuml"].count("o--") == 1
    assert "File -- File" not in result["generated_final_plantuml"]
    assert "File.name" not in result["generated_final_plantuml"]
    assert "FileSystemElement.extension" not in result["generated_final_plantuml"]
    assert "parent_folder" not in result["generated_final_plantuml"]
    assert "file_system_path" not in result["generated_final_plantuml"]
    assert "parent_folder_name" not in result["generated_final_plantuml"]
    assert "size_in_bytes" not in result["generated_final_plantuml"]
    assert "contains" not in result["generated_final_plantuml"]
    file_class = next(uml_class for uml_class in result["model"].classes if uml_class.name == "File")
    fs_class = next(uml_class for uml_class in result["model"].classes if uml_class.name == "FileSystemElement")
    folder_class = next(uml_class for uml_class in result["model"].classes if uml_class.name == "Folder")
    assert sorted(attribute.name for attribute in file_class.attributes) == ["extension", "size"]
    assert [attribute.name for attribute in fs_class.attributes] == ["name"]
    assert folder_class.attributes == []
    assert len(concept_llm.prompts) == 1
    assert len(attribute_llm.prompts) >= 1
    assert len(relationship_llm.prompts) >= 1
    assert len(validator_llm.prompts) >= 1
    assert any(relationship.relationship_type == "aggregation" for relationship in result["model"].relationships)
    assert sum(1 for relationship in result["model"].relationships if relationship.relationship_type == "generalization") >= 2


def test_model_integrator_dedupes_relationships_and_cleans_files_hierarchy():
    integrator = ModelIntegratorAgent()
    from src.schemas import UMLAttribute, UMLClass, UMLModel, UMLRelationship

    model = UMLModel(
        classes=[
            UMLClass(
                name="FileSystemElement",
                attributes=[
                    UMLAttribute(name="name", data_type="String"),
                    UMLAttribute(name="extension", data_type="String"),
                ],
            ),
            UMLClass(
                name="File",
                attributes=[
                    UMLAttribute(name="extension", data_type="String"),
                    UMLAttribute(name="size", data_type="Integer"),
                    UMLAttribute(name="name", data_type="String"),
                ],
            ),
            UMLClass(name="Folder"),
        ],
        relationships=[
            UMLRelationship(
                source="Folder",
                target="FileSystemElement",
                relationship_type="aggregation",
                source_multiplicity="1",
                target_multiplicity="*",
            ),
            UMLRelationship(
                source="Folder",
                target="FileSystemElement",
                relationship_type="aggregation",
                source_multiplicity="1",
                target_multiplicity="*",
                label="contains",
            ),
        ],
    )

    normalized = integrator.normalize_model(
        model,
        requirements=(
            "File system elements can be folders or files. "
            "All file system elements have a name. "
            "Every file system element belongs to a folder. "
            "Folders aggregate any number of file system elements. "
            "Files have extension and size."
        ),
    )

    file_class = next(uml_class for uml_class in normalized.classes if uml_class.name == "File")
    fs_class = next(uml_class for uml_class in normalized.classes if uml_class.name == "FileSystemElement")
    assert sorted(attribute.name for attribute in file_class.attributes) == ["extension", "size"]
    assert [attribute.name for attribute in fs_class.attributes] == ["name"]
    assert sum(1 for relationship in normalized.relationships if relationship.relationship_type == "aggregation") == 1
    assert sum(1 for relationship in normalized.relationships if relationship.relationship_type == "generalization") >= 2


def test_apply_generalization_heuristics_adds_files_generalizations_without_duplicates():
    from src.schemas import UMLClass, UMLModel, UMLRelationship

    model = UMLModel(
        classes=[
            UMLClass(name="FileSystemElement"),
            UMLClass(name="Folder"),
            UMLClass(name="File"),
        ],
        relationships=[
            UMLRelationship(
                source="Folder",
                target="FileSystemElement",
                relationship_type="aggregation",
                source_multiplicity="1",
                target_multiplicity="*",
            )
        ],
    )

    processed = apply_generalization_heuristics(
        "It represents oily system elements which can be a folders or files.",
        model,
    )

    generalizations = [
        (relationship.source, relationship.target)
        for relationship in processed.relationships
        if relationship.relationship_type.lower() == "generalization"
    ]

    assert ("Folder", "FileSystemElement") in generalizations
    assert ("File", "FileSystemElement") in generalizations
    assert len(generalizations) == 2
    assert sum(1 for relationship in processed.relationships if relationship.relationship_type.lower() == "aggregation") == 1


def test_plantuml_agent_renders_files_generalizations_and_single_aggregation():
    from src.agents.plantuml_agent import PlantUMLAgent
    from src.schemas import UMLAttribute, UMLClass, UMLModel, UMLRelationship

    model = UMLModel(
        classes=[
            UMLClass(name="FileSystemElement", attributes=[UMLAttribute(name="name", data_type="String")]),
            UMLClass(name="Folder"),
            UMLClass(
                name="File",
                attributes=[
                    UMLAttribute(name="extension", data_type="String"),
                    UMLAttribute(name="size", data_type="Integer"),
                ],
            ),
        ],
        relationships=[
            UMLRelationship(
                source="Folder",
                target="FileSystemElement",
                relationship_type="aggregation",
                source_multiplicity="1",
                target_multiplicity="*",
            ),
            UMLRelationship(
                source="Folder",
                target="FileSystemElement",
                relationship_type="generalization",
            ),
            UMLRelationship(
                source="File",
                target="FileSystemElement",
                relationship_type="generalization",
            ),
        ],
    )

    plantuml = PlantUMLAgent().generate(model)

    assert "Folder --|> FileSystemElement" in plantuml
    assert "File --|> FileSystemElement" in plantuml
    assert 'Folder "1" o-- "*" FileSystemElement' in plantuml
    assert plantuml.count("o--") == 1
    assert "File.name" not in plantuml
    assert "FileSystemElement.extension" not in plantuml


def _fake_model():
    from src.schemas import UMLClass, UMLModel

    return UMLModel(classes=[UMLClass(name="Order"), UMLClass(name="Customer")], relationships=[])


def test_relationship_agent_preserves_existing_relationships_when_revision_is_empty():
    fake_llm = SequencedFakeLLM(
        [
            "```json\n"
            '{"relationships": []}'
            "\n```",
        ]
    )

    agent = RelationshipAgent(llm=fake_llm)
    from src.schemas import UMLRelationship

    model = _file_model_with_relationships(
        [
            UMLRelationship(
                source="Folder",
                target="FileSystemElement",
                relationship_type="generalization",
            )
        ]
    )

    revised = agent.revise(model, "Folders contain files.")

    assert len(revised.relationships) == 1
    assert revised.relationships[0].source == "Folder"
    assert revised.relationships[0].target == "FileSystemElement"


def test_relationship_agent_infers_files_generalization_deterministically():
    fake_llm = SequencedFakeLLM(
        [
            "```json\n"
            '{"relationships": []}'
            "\n```",
        ]
    )

    agent = RelationshipAgent(llm=fake_llm)
    from src.schemas import UMLClass, UMLModel

    model = UMLModel(
        classes=[
            UMLClass(name="FileSystemElement"),
            UMLClass(name="Folder"),
            UMLClass(name="File"),
        ],
        relationships=[],
    )

    revised = agent.add_relationships(
        model,
        "File system elements can be folders or files.",
    )

    generalizations = [
        (relationship.source, relationship.target)
        for relationship in revised.relationships
        if relationship.relationship_type.lower() in {"inheritance", "generalization", "extends"}
    ]

    assert ("Folder", "FileSystemElement") in generalizations
    assert ("File", "FileSystemElement") in generalizations
    assert len(generalizations) == 2


def _file_model_with_relationships(relationships):
    from src.schemas import UMLClass, UMLModel

    return UMLModel(
        classes=[
            UMLClass(name="Folder"),
            UMLClass(name="File"),
            UMLClass(name="FileSystemElement"),
        ],
        relationships=list(relationships),
    )
