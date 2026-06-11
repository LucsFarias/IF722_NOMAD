from types import SimpleNamespace
import json

from scripts.evaluate_results import main as evaluate_main
from scripts.run_experiment import run_experiment


class SequencedFakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        response = self.responses.pop(0)
        return SimpleNamespace(content=response)


def _write_dataset(tmp_path):
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "name": "sales",
                        "description": "Customers place orders.",
                        "golden_diagram": "@startuml\nclass Customer\nclass Order\nCustomer -- Order\n@enduml",
                    }
                )
            ]
        ),
        encoding="utf-8",
    )
    return dataset_path


def test_run_experiment_single_mode_writes_artifacts(tmp_path):
    dataset_path = _write_dataset(tmp_path)
    runs_root = tmp_path / "runs"
    fake_llm = SequencedFakeLLM(
        [
            "@startuml\nclass Customer\nclass Order\nCustomer -- Order\n@enduml",
            '{"issues": [], "metadata": {"status": "clean"}}',
        ]
    )

    results = run_experiment(
        dataset_path=dataset_path,
        mode="single",
        provider="ollama",
        model="fake-model",
        output_root=runs_root,
        llm_factory=lambda **kwargs: fake_llm,
    )

    case_dir = runs_root / "single" / "sales"
    assert len(results) == 1
    assert (case_dir / "input_requirements.txt").exists()
    assert (case_dir / "gold.puml").exists()
    assert (case_dir / "generated_initial.puml").exists()
    assert (case_dir / "generated_final.puml").exists()
    assert (case_dir / "intermediate_model.json").exists()
    assert (case_dir / "validation_report_initial.json").exists()
    assert (case_dir / "validation_report_final.json").exists()
    assert (case_dir / "rethink_history.json").exists()
    assert (case_dir / "metrics.json").exists()

    metrics = json.loads((case_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["mode"] == "single"
    assert metrics["initial_issue_count"] == 0
    assert metrics["final_issue_count"] == 0
    assert metrics["llm_calls"] == 2


def test_run_experiment_cascade_mode_writes_artifacts(tmp_path):
    dataset_path = _write_dataset(tmp_path)
    runs_root = tmp_path / "runs"
    fake_llm = SequencedFakeLLM(
        [
            '{"classes": ['
            '{"name": "Customer", "description": "A customer", "attributes": []},'
            '{"name": "Order", "description": "An order", "attributes": []}'
            ']}',
            '{"attributes": [{"name": "name", "data_type": "String", "confidence": 0.9}]}',
            '{"attributes": [{"name": "number", "data_type": "String", "confidence": 0.9}]}',
            '{"relationships": ['
            '{"source": "Customer", "target": "Order", "relationship_type": "association", '
            '"source_multiplicity": "1", "target_multiplicity": "*", "label": "places", "metadata": {}}'
            ']}',
            '{"issues": [], "metadata": {"status": "clean"}}',
        ]
    )

    results = run_experiment(
        dataset_path=dataset_path,
        mode="cascade",
        provider="ollama",
        model="fake-model",
        output_root=runs_root,
        llm_factory=lambda **kwargs: fake_llm,
    )

    case_dir = runs_root / "cascade" / "sales"
    assert len(results) == 1
    metrics = json.loads((case_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["mode"] == "cascade"
    assert metrics["final_issue_count"] == 0
    assert metrics["llm_calls"] == 5
    intermediate = json.loads((case_dir / "intermediate_model.json").read_text(encoding="utf-8"))
    assert len(intermediate["classes"]) == 2


def test_run_experiment_rethink_and_evaluate_summary(tmp_path):
    dataset_path = _write_dataset(tmp_path)
    runs_root = tmp_path / "runs"
    single_llm = SequencedFakeLLM(
        [
            "@startuml\nclass Customer\nclass Order\nCustomer -- Order\n@enduml",
            '{"issues": [], "metadata": {"status": "clean"}}',
        ]
    )
    rethink_llm = SequencedFakeLLM(
        [
            '{"classes": ['
            '{"name": "Customer", "description": "A customer", "attributes": []},'
            '{"name": "Order", "description": "An order", "attributes": []}'
            ']}',
            '{"attributes": [{"name": "name", "data_type": "String", "confidence": 0.9}]}',
            '{"attributes": [{"name": "number", "data_type": "String", "confidence": 0.9}]}',
            '{"relationships": []}',
            '{"issues": ['
            '{'
            '"id": "i-1", "severity": "critical", "category": "relationship", "target_agent": "relationship", '
            '"description": "Missing Customer-Order relation", '
            '"evidence_from_requirements": "Customers place orders", '
            '"suggested_fix": "Add Customer places Order"'
            '}'
            '], "metadata": {}}',
            '{"relationships": ['
            '{"source": "Customer", "target": "Order", "relationship_type": "association", '
            '"source_multiplicity": "1", "target_multiplicity": "*", "label": "places", "metadata": {}}'
            ']}',
            '{"issues": [], "metadata": {"status": "clean"}}',
        ]
    )

    run_experiment(
        dataset_path=dataset_path,
        mode="single",
        provider="ollama",
        model="fake-model",
        output_root=runs_root,
        llm_factory=lambda **kwargs: single_llm,
    )
    run_experiment(
        dataset_path=dataset_path,
        mode="rethink",
        provider="ollama",
        model="fake-model",
        output_root=runs_root,
        llm_factory=lambda **kwargs: rethink_llm,
    )

    summary_csv = tmp_path / "results" / "summary.csv"
    evaluate_main(["--runs", str(runs_root), "--output", str(summary_csv)])

    assert summary_csv.exists()
    summary_json = summary_csv.with_suffix(".json")
    assert summary_json.exists()

    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["aggregate"]["count"] == 2
    assert len(payload["cases"]) == 2

    rethink_metrics = json.loads((runs_root / "rethink" / "sales" / "metrics.json").read_text(encoding="utf-8"))
    rethink_history = json.loads((runs_root / "rethink" / "sales" / "rethink_history.json").read_text(encoding="utf-8"))
    assert rethink_metrics["initial_issue_count"] == 1
    assert rethink_metrics["final_issue_count"] == 0
    assert rethink_metrics["resolved_issue_count"] == 1
    assert len(rethink_history) == 2
