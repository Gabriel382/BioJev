from __future__ import annotations

from dataclasses import dataclass

from datasets import load_dataset

from biojev.datasets.base import DatasetAdapter
from biojev.schemas import NLIExample
from biojev.utils.io import write_jsonl


LABELS = {0: "entailment", 1: "neutral", 2: "contradiction"}


@dataclass(frozen=True)
class GeneralNLISpec:
    repo: str
    split_map: dict[str, str]


SPECS = {
    "snli": GeneralNLISpec(
        repo="stanfordnlp/snli",
        split_map={"train": "train", "validation": "dev", "test": "test"},
    ),
    "mnli": GeneralNLISpec(
        repo="nyu-mll/multi_nli",
        split_map={"train": "train", "validation_matched": "dev", "validation_mismatched": "test"},
    ),
    "anli_r1": GeneralNLISpec(
        repo="facebook/anli",
        split_map={"train_r1": "train", "dev_r1": "dev", "test_r1": "test"},
    ),
    "anli_r2": GeneralNLISpec(
        repo="facebook/anli",
        split_map={"train_r2": "train", "dev_r2": "dev", "test_r2": "test"},
    ),
    "anli_r3": GeneralNLISpec(
        repo="facebook/anli",
        split_map={"train_r3": "train", "dev_r3": "dev", "test_r3": "test"},
    ),
}


class GeneralNLIAdapter(DatasetAdapter):
    """Normalize public general-domain NLI corpora to BioJev's canonical schema.

    The three public corpora used here expose the same label order:
    0=entailment, 1=neutral, 2=contradiction. Rows with label=-1 are omitted.
    """

    def __init__(self, name: str, *args, **kwargs):
        if name not in SPECS:
            raise KeyError(f"Unsupported general NLI dataset: {name}")
        super().__init__(*args, **kwargs)
        self.name = name
        self.spec = SPECS[name]

    def prepare(self, force: bool = False):
        ds = load_dataset(self.spec.repo)
        output: dict[str, list[NLIExample]] = {}
        for source_split, target_split in self.spec.split_map.items():
            if source_split not in ds:
                continue
            rows: list[NLIExample] = []
            for i, row in enumerate(ds[source_split]):
                raw_label = row.get("label", -1)
                try:
                    raw_label = int(raw_label)
                except (TypeError, ValueError):
                    continue
                if raw_label not in LABELS:
                    continue
                premise = str(row.get("premise", "")).strip()
                hypothesis = str(row.get("hypothesis", "")).strip()
                if not premise or not hypothesis:
                    continue
                rows.append(
                    NLIExample(
                        id=f"{self.name}-{source_split}-{i}",
                        dataset=self.name,
                        split=target_split,
                        premise=premise,
                        hypothesis=hypothesis,
                        label=LABELS[raw_label],
                        metadata={"source_split": source_split, "source_index": i},
                    )
                )
            if rows:
                output[target_split] = rows
                write_jsonl(self.processed_root / self.name / f"{target_split}.jsonl", rows)
        return output
