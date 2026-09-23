from __future__ import annotations

import json
from pathlib import Path

from biojev.datasets.base import DatasetAdapter
from biojev.schemas import NLIExample
from biojev.utils.io import write_jsonl


class MedNLIAdapter(DatasetAdapter):
    """Adapter for a manually obtained MedNLI directory.

    MedNLI is credentialed through PhysioNet and therefore is never downloaded by this project.
    Pass a directory containing the official JSONL files.
    """

    name = "mednli"

    def __init__(self, source_dir: str | Path, raw_root="data/raw", processed_root="data/processed"):
        super().__init__(raw_root, processed_root)
        self.source_dir = Path(source_dir)

    def prepare(self, force: bool = False):
        if not self.source_dir.exists():
            raise FileNotFoundError(
                f"MedNLI source directory not found: {self.source_dir}. "
                "Download it manually after obtaining PhysioNet access."
            )
        output: dict[str, list[NLIExample]] = {}
        for path in sorted(self.source_dir.glob("*.jsonl")):
            lower = path.name.lower()
            split = "test" if "test" in lower else "dev" if "dev" in lower else "train"
            rows: list[NLIExample] = []
            with path.open("r", encoding="utf-8") as handle:
                for i, line in enumerate(handle):
                    raw = json.loads(line)
                    premise = raw.get("sentence1") or raw.get("premise")
                    hypothesis = raw.get("sentence2") or raw.get("hypothesis")
                    label = raw.get("gold_label") or raw.get("label")
                    if premise is None or hypothesis is None or label is None:
                        continue
                    rows.append(
                        NLIExample(
                            id=f"mednli-{split}-{i}", dataset=self.name, split=split,
                            premise=str(premise), hypothesis=str(hypothesis), label=str(label).lower(),
                            metadata={"source_file": path.name},
                        )
                    )
            output.setdefault(split, []).extend(rows)
        if not output:
            raise RuntimeError("No MedNLI JSONL records found in the supplied directory.")
        for split, rows in output.items():
            write_jsonl(self.processed_root / self.name / f"{split}.jsonl", rows)
        return output
