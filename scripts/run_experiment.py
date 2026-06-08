from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.attribute_specialist import AttributeSpecialistAgent
from src.agents.concept_agent import ConceptAgent
from src.agents.model_integrator import ModelIntegratorAgent
from src.agents.plantuml_agent import PlantUMLAgent
from src.agents.relationship_agent import RelationshipAgent
from src.agents.rethink_router import RethinkRouter
from src.agents.validator_agent import ValidatorAgent
from src.data_loader import DatasetCase, load_dataset
from src.eval_helpers import evaluate_uml, plantuml_to_model
from src.llm.provider import create_llm_provider
from src.pipeline import CascadeNOMAD, RethinkNOMAD, SingleAgentBaseline
from src.schemas import ExperimentResult, RethinkStep, UMLModel, ValidationReport


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class CachedLLM:
    def __init__(self, llm: Any, cache_path: Path):
        self.llm = llm
        self.cache_path = cache_path
        self.cache = self._load_cache()
        self.calls = 0

    def _load_cache(self) -> Dict[str, str]:
        if not self.cache_path.exists():
            return {}
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")

    def invoke(self, prompt: str):
        self.calls += 1
        if prompt in self.cache:
            return self.cache[prompt]

        if hasattr(self.llm, "invoke"):
            response = self.llm.invoke(prompt)
        else:
            response = self.llm(prompt)

        content = response if isinstance(response, str) else getattr(response, "content", None)
        if content is None:
            content = getattr(response, "text", None)
        if content is None:
            content = str(response)
        self.cache[prompt] = content
        self._save_cache()
        return content


class CountingLLM:
    def __init__(self, llm: Any):
        self.llm = llm
        self.calls = 0

    def invoke(self, prompt: str):
        self.calls += 1
        if hasattr(self.llm, "invoke"):
            return self.llm.invoke(prompt)
        return self.llm(prompt)


@dataclass
class RunArtifacts:
    case_name: str
    output_dir: Path
    result: ExperimentResult


def build_shared_llm(provider: str, model: Optional[str], use_cache: bool, cache_path: Path):
    llm = create_llm_provider(provider=provider, model=model, temperature=0.0)
    if use_cache:
        return CachedLLM(llm, cache_path)
    return llm


def build_single_runner(shared_llm: Any) -> SingleAgentBaseline:
    return SingleAgentBaseline(llm=shared_llm)


def build_cascade_runner(shared_llm: Any) -> CascadeNOMAD:
    return CascadeNOMAD(
        concept_agent=ConceptAgent(llm=shared_llm),
        attribute_specialist=AttributeSpecialistAgent(llm=shared_llm),
        relationship_agent=RelationshipAgent(llm=shared_llm),
        integrator=ModelIntegratorAgent(),
        plantuml_agent=PlantUMLAgent(),
    )


def build_rethink_runner(shared_llm: Any, max_rethink_iterations: int) -> RethinkNOMAD:
    return RethinkNOMAD(
        concept_agent=ConceptAgent(llm=shared_llm),
        attribute_specialist=AttributeSpecialistAgent(llm=shared_llm),
        relationship_agent=RelationshipAgent(llm=shared_llm),
        integrator=ModelIntegratorAgent(),
        plantuml_agent=PlantUMLAgent(),
        validator_agent=ValidatorAgent(llm=shared_llm),
        router=RethinkRouter(),
        max_rethink_iterations=max_rethink_iterations,
    )


def run_case(
    case: DatasetCase,
    mode: str,
    provider: str,
    model: Optional[str],
    output_root: Path,
    max_rethink_iterations: int,
    use_cache: bool,
    llm_factory: Callable[..., Any] = create_llm_provider,
) -> RunArtifacts:
    case_dir = output_root / mode / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_root / ".cache" / "llm_cache.json"
    shared_llm = llm_factory(provider=provider, model=model, temperature=0.0)
    if use_cache:
        shared_llm = CachedLLM(shared_llm, cache_path)
    shared_llm = CountingLLM(shared_llm)

    start = perf_counter()
    if mode == "single":
        runner = build_single_runner(shared_llm)
        generated_initial = runner.run(case.description)
        generated_final = generated_initial
        intermediate_model = plantuml_to_model(generated_final)
        validator = ValidatorAgent(llm=shared_llm)
        validation_report_initial = validator.validate(case.description, intermediate_model, generated_initial)
        validation_report_final = validation_report_initial
        rethink_history: List[Dict[str, Any]] = []
        llm_calls = getattr(shared_llm, "calls", 0) or len(getattr(shared_llm, "prompts", [])) or 0
    elif mode == "cascade":
        runner = build_cascade_runner(shared_llm)
        cascade_result = runner.run(case.description)
        generated_initial = cascade_result["plantuml"]
        generated_final = generated_initial
        intermediate_model = cascade_result["model"]
        validator = ValidatorAgent(llm=shared_llm)
        validation_report_initial = validator.validate(case.description, intermediate_model, generated_initial)
        validation_report_final = validation_report_initial
        rethink_history = []
        llm_calls = getattr(shared_llm, "calls", 0) or len(getattr(shared_llm, "prompts", [])) or 0
    elif mode == "rethink":
        runner = build_rethink_runner(shared_llm, max_rethink_iterations=max_rethink_iterations)
        rethink_result = runner.run(case.description)
        generated_initial = rethink_result["generated_initial_plantuml"]
        generated_final = rethink_result["generated_final_plantuml"]
        intermediate_model = rethink_result["final_model"]
        validation_report_initial = rethink_result["validation_report_initial"]
        validation_report_final = rethink_result["validation_report_final"]
        rethink_history = [step.to_dict() for step in rethink_result["rethink_history"]]
        llm_calls = rethink_result["llm_calls"]
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    metrics = build_case_metrics(
        case=case,
        mode=mode,
        provider=provider,
        model=model or "",
        generated_initial=generated_initial,
        generated_final=generated_final,
        intermediate_model=intermediate_model,
        validation_report_initial=validation_report_initial,
        validation_report_final=validation_report_final,
        runtime_seconds=perf_counter() - start,
        llm_calls=llm_calls,
        rethink_history=rethink_history,
    )

    save_case_artifacts(
        case_dir=case_dir,
        case=case,
        generated_initial=generated_initial,
        generated_final=generated_final,
        intermediate_model=intermediate_model,
        validation_report_initial=validation_report_initial,
        validation_report_final=validation_report_final,
        rethink_history=rethink_history,
        metrics=metrics,
    )

    result = ExperimentResult(
        case_name=case.name,
        mode=mode,
        provider=provider,
        model=model or "",
        input_requirements=case.description,
        gold_puml=case.golden_diagram,
        generated_initial_puml=generated_initial,
        generated_final_puml=generated_final,
        intermediate_model=intermediate_model,
        validation_report_initial=validation_report_initial,
        validation_report_final=validation_report_final,
        rethink_history=[RethinkStep.from_dict(step) for step in rethink_history],
        metrics=metrics,
        runtime_seconds=metrics["runtime_seconds"],
        llm_calls=metrics["llm_calls"],
        output_dir=str(case_dir),
    )
    return RunArtifacts(case_name=case.name, output_dir=case_dir, result=result)


def build_case_metrics(
    case: DatasetCase,
    mode: str,
    provider: str,
    model: str,
    generated_initial: str,
    generated_final: str,
    intermediate_model: UMLModel,
    validation_report_initial: ValidationReport,
    validation_report_final: ValidationReport,
    runtime_seconds: float,
    llm_calls: int,
    rethink_history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    initial_eval = evaluate_uml(case.golden_diagram, generated_initial, normalize=True)
    final_eval = evaluate_uml(case.golden_diagram, generated_final, normalize=True)
    initial_issue_ids = {issue.issue_id for issue in validation_report_initial.issues}
    final_issue_ids = {issue.issue_id for issue in validation_report_final.issues}
    resolved_issue_ids = initial_issue_ids - final_issue_ids

    metrics = {
        "case_name": case.name,
        "mode": mode,
        "provider": provider,
        "model": model,
        "runtime_seconds": runtime_seconds,
        "llm_calls": llm_calls,
        "initial_issue_count": len(validation_report_initial.issues),
        "final_issue_count": len(validation_report_final.issues),
        "resolved_issue_count": len(resolved_issue_ids),
        "initial_evaluation": initial_eval,
        "final_evaluation": final_eval,
        "average_f1": _average_f1(final_eval),
        "rethink_iterations": len(rethink_history),
        "rethink_history": rethink_history,
    }
    return metrics


def _average_f1(evaluation: Dict[str, Any]) -> float:
    return (
        evaluation["classes"]["f1"]
        + evaluation["attributes"]["f1"]
        + evaluation["relationships_strict"]["f1"]
        + evaluation["relationships_relaxed"]["f1"]
    ) / 4.0


def save_case_artifacts(
    case_dir: Path,
    case: DatasetCase,
    generated_initial: str,
    generated_final: str,
    intermediate_model: UMLModel,
    validation_report_initial: ValidationReport,
    validation_report_final: ValidationReport,
    rethink_history: List[Dict[str, Any]],
    metrics: Dict[str, Any],
) -> None:
    (case_dir / "input_requirements.txt").write_text(case.description, encoding="utf-8")
    (case_dir / "gold.puml").write_text(case.golden_diagram, encoding="utf-8")
    (case_dir / "generated_initial.puml").write_text(generated_initial, encoding="utf-8")
    (case_dir / "generated_final.puml").write_text(generated_final, encoding="utf-8")
    _json_dump(case_dir / "intermediate_model.json", intermediate_model.to_dict())
    _json_dump(case_dir / "validation_report_initial.json", validation_report_initial.to_dict())
    _json_dump(case_dir / "validation_report_final.json", validation_report_final.to_dict())
    _json_dump(case_dir / "rethink_history.json", rethink_history)
    _json_dump(case_dir / "metrics.json", metrics)


def run_experiment(
    dataset_path: Path,
    mode: str,
    provider: str,
    model: Optional[str],
    output_root: Path,
    max_rethink_iterations: int = 2,
    limit_cases: Optional[int] = None,
    use_cache: bool = False,
    llm_factory: Callable[..., Any] = create_llm_provider,
) -> List[RunArtifacts]:
    cases = load_dataset(dataset_path, limit=limit_cases)
    output_root.mkdir(parents=True, exist_ok=True)
    runs: List[RunArtifacts] = []
    for case in cases:
        runs.append(
            run_case(
                case=case,
                mode=mode,
                provider=provider,
                model=model,
                output_root=output_root,
                max_rethink_iterations=max_rethink_iterations,
                use_cache=use_cache,
                llm_factory=llm_factory,
            )
        )
    return runs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run NOMAD experiments.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mode", choices=["single", "cascade", "rethink"], required=True)
    parser.add_argument("--provider", choices=["gemini", "ollama"], required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-rethink-iterations", type=int, default=2)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--use-cache", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_experiment(
        dataset_path=Path(args.dataset),
        mode=args.mode,
        provider=args.provider,
        model=args.model,
        output_root=Path(args.output),
        max_rethink_iterations=args.max_rethink_iterations,
        limit_cases=args.limit_cases,
        use_cache=args.use_cache,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
