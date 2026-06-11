from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_metrics_files(runs_root: Path) -> List[Dict[str, Any]]:
    metrics_files = sorted(runs_root.rglob("metrics.json"))
    runs: List[Dict[str, Any]] = []
    for metrics_file in metrics_files:
        with metrics_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["metrics_path"] = str(metrics_file)
        runs.append(payload)
    return runs


def flatten_run(payload: Dict[str, Any]) -> Dict[str, Any]:
    final_eval = payload.get("final_evaluation", {})
    initial_eval = payload.get("initial_evaluation", {})
    row = {
        "case_name": payload.get("case_name"),
        "mode": payload.get("mode"),
        "provider": payload.get("provider"),
        "model": payload.get("model"),
        "runtime_seconds": payload.get("runtime_seconds", 0.0),
        "llm_calls": payload.get("llm_calls", 0),
        "initial_issue_count": payload.get("initial_issue_count", 0),
        "final_issue_count": payload.get("final_issue_count", 0),
        "resolved_issue_count": payload.get("resolved_issue_count", 0),
        "average_f1": payload.get("average_f1", 0.0),
        "initial_class_precision": _metric(initial_eval, "classes", "precision"),
        "initial_class_recall": _metric(initial_eval, "classes", "recall"),
        "initial_class_f1": _metric(initial_eval, "classes", "f1"),
        "initial_attribute_precision": _metric(initial_eval, "attributes", "precision"),
        "initial_attribute_recall": _metric(initial_eval, "attributes", "recall"),
        "initial_attribute_f1": _metric(initial_eval, "attributes", "f1"),
        "initial_relationship_strict_f1": _metric(initial_eval, "relationships_strict", "f1"),
        "initial_relationship_relaxed_f1": _metric(initial_eval, "relationships_relaxed", "f1"),
        "final_class_precision": _metric(final_eval, "classes", "precision"),
        "final_class_recall": _metric(final_eval, "classes", "recall"),
        "final_class_f1": _metric(final_eval, "classes", "f1"),
        "final_attribute_precision": _metric(final_eval, "attributes", "precision"),
        "final_attribute_recall": _metric(final_eval, "attributes", "recall"),
        "final_attribute_f1": _metric(final_eval, "attributes", "f1"),
        "final_relationship_strict_precision": _metric(final_eval, "relationships_strict", "precision"),
        "final_relationship_strict_recall": _metric(final_eval, "relationships_strict", "recall"),
        "final_relationship_strict_f1": _metric(final_eval, "relationships_strict", "f1"),
        "final_relationship_relaxed_precision": _metric(final_eval, "relationships_relaxed", "precision"),
        "final_relationship_relaxed_recall": _metric(final_eval, "relationships_relaxed", "recall"),
        "final_relationship_relaxed_f1": _metric(final_eval, "relationships_relaxed", "f1"),
    }
    return row


def _metric(payload: Dict[str, Any], section: str, metric: str) -> float:
    return float(payload.get(section, {}).get(metric, 0.0))


def aggregate_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"count": 0}

    numeric_fields = [
        "runtime_seconds",
        "llm_calls",
        "initial_issue_count",
        "final_issue_count",
        "resolved_issue_count",
        "average_f1",
        "final_class_f1",
        "final_attribute_f1",
        "final_relationship_strict_f1",
        "final_relationship_relaxed_f1",
    ]

    aggregates: Dict[str, Any] = {"count": len(rows)}
    for field in numeric_fields:
        aggregates[f"avg_{field}"] = sum(float(row.get(field, 0.0)) for row in rows) / len(rows)
    return aggregates


def write_summary_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_json(rows: List[Dict[str, Any]], output_path: Path) -> None:
    payload = {
        "cases": rows,
        "aggregate": aggregate_rows(rows),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate NOMAD experiment results.")
    parser.add_argument("--runs", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    runs_root = Path(args.runs)
    rows = [flatten_run(payload) for payload in load_metrics_files(runs_root)]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_summary_csv(rows, output_path)
    write_summary_json(rows, output_path.with_suffix(".json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
