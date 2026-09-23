from __future__ import annotations

from datasets import load_dataset

from biojev.datasets.base import DatasetAdapter
from biojev.schemas import NLIExample
from biojev.utils.io import write_jsonl


class BioNLIAdapter(DatasetAdapter):
    name = "bionli"

    def prepare(self, force: bool = False):
        ds = load_dataset("presencesw/bionli")
        source = ds["train"]
        # The published HF copy contains one 36k split. Keep it intact by default;
        # deterministic train/dev/test splitting is done only when requested by config.
        examples = [
            NLIExample(
                id=f"bionli-{i}", dataset=self.name, split="train", task="nli",
                premise=row["sentence1"], hypothesis=row["sentence2"],
                label=str(row["gold_label"]).lower(), metadata={"source_index": i}
            )
            for i, row in enumerate(source)
        ]
        write_jsonl(self.processed_root / self.name / "train.jsonl", examples)
        return {"train": examples}
