from src.schemas import (
    ExperimentResult,
    RethinkStep,
    UMLAttribute,
    UMLClass,
    UMLModel,
    UMLRelationship,
    ValidationIssue,
    ValidationReport,
)


def test_schema_round_trip_and_experiment_result_serialization():
    uml_attribute = UMLAttribute(
        name="id",
        data_type="String",
        visibility="private",
        default_value=None,
        multiplicity="1",
        stereotype=None,
        source="inferred",
    )
    uml_class = UMLClass(name="Order", description="An order", attributes=[uml_attribute])
    relationship = UMLRelationship(
        source="Customer",
        target="Order",
        relationship_type="association",
        source_multiplicity="1",
        target_multiplicity="*",
        label="places",
    )
    model = UMLModel(classes=[uml_class], relationships=[relationship], metadata={"source": "test"})

    issue = ValidationIssue(
        issue_id="issue-1",
        severity="critical",
        category="missing_class",
        message="Missing class Customer",
        responsible_agent="ConceptAgent",
        target="Customer",
        suggestions=["Add Customer"],
        is_critical=True,
    )
    report = ValidationReport(requirements="req", issues=[issue], validated_model=model)
    rethink_step = RethinkStep(iteration=1, issues=[issue], llm_calls=2, runtime_seconds=1.25)
    result = ExperimentResult(
        case_name="car",
        mode="rethink",
        provider="gemini",
        model="gemini-2.5-flash",
        input_requirements="Cars belong to models.",
        gold_puml="@startuml\n@enduml",
        generated_initial_puml="@startuml\n@enduml",
        generated_final_puml="@startuml\n@enduml",
        intermediate_model=model,
        validation_report_initial=report,
        validation_report_final=report,
        rethink_history=[rethink_step],
        metrics={"f1": 1.0},
        runtime_seconds=3.5,
        llm_calls=2,
        output_dir="results/car",
    )

    restored_model = UMLModel.from_dict(model.to_dict())
    restored_report = ValidationReport.from_dict(report.to_dict())
    restored_result = ExperimentResult.from_dict(result.to_dict())

    assert restored_model.to_dict() == model.to_dict()
    assert restored_report.to_dict()["is_valid"] is False
    assert restored_report.critical_issues[0].issue_id == "issue-1"
    assert restored_result.to_dict()["case_name"] == "car"
    assert restored_result.rethink_history[0].iteration == 1


def test_validation_report_flags_validity_by_critical_issues():
    report = ValidationReport(
        issues=[
            ValidationIssue(
                issue_id="issue-1",
                severity="minor",
                category="style",
                message="Minor formatting issue",
                is_critical=False,
            )
        ]
    )

    assert report.is_valid is True
    assert len(report.noncritical_issues) == 1
