from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "small_models" / "data.jsonl"


@dataclass
class DatasetCase:
    name: str
    description: str
    golden_diagram: str
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = dict(self.raw)
        payload.update(
            {
                "name": self.name,
                "description": self.description,
                "golden_diagram": self.golden_diagram,
            }
        )
        return payload


def load_dataset(path: Union[str, Path] = DEFAULT_DATASET_PATH, limit: Optional[int] = None) -> List[DatasetCase]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    cases: List[DatasetCase] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {dataset_path} at line {line_number}") from exc

            missing_fields = [field_name for field_name in ("name", "description", "golden_diagram") if field_name not in payload]
            if missing_fields:
                raise ValueError(
                    f"Missing required field(s) {missing_fields} in {dataset_path} at line {line_number}"
                )

            cases.append(
                DatasetCase(
                    name=str(payload["name"]),
                    description=str(payload["description"]),
                    golden_diagram=str(payload["golden_diagram"]),
                    raw=dict(payload),
                )
            )

            if limit is not None and len(cases) >= limit:
                break

    return cases
