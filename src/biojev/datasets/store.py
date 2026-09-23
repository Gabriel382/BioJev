from __future__ import annotations

from pathlib import Path

from biojev.schemas import NLIExample, RelationExample
from biojev.utils.io import read_jsonl


def load_processed(dataset: str, split: str, root: str | Path = "data/processed"):
    path = Path(root) / dataset / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Processed split not found: {path}. Run scripts/download_datasets.py first.")
    examples = []
    for row in read_jsonl(path):
        task = row.get("task")
        examples.append(NLIExample(**row) if task == "nli" else RelationExample(**row))
    return examples
