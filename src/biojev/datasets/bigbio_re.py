from __future__ import annotations

import random
from typing import Any

from datasets import load_dataset

from biojev.datasets.base import DatasetAdapter
from biojev.datasets.common import (
    allowed_type_pairs,
    entity_text,
    flatten_passages,
    negative_entity_pairs,
    observed_relation_labels,
    relation_pairs,
)
from biojev.schemas import RelationExample
from biojev.utils.io import write_jsonl


BIGBIO_CONFIGS = {
    "chemprot": ("bigbio/chemprot", "chemprot_bigbio_kb"),
    "ddi2013": ("bigbio/ddi_corpus", "ddi_corpus_bigbio_kb"),
    "biored": ("bigbio/biored", "biored_bigbio_kb"),
}


class BigBioRelationAdapter(DatasetAdapter):
    def __init__(
        self,
        name: str,
        raw_root="data/raw",
        processed_root="data/processed",
        include_no_relation: bool = False,
        negative_ratio: float = 1.0,
        seed: int = 42,
    ):
        if name not in BIGBIO_CONFIGS:
            raise ValueError(f"Unsupported BigBio dataset: {name}")
        self.name = name
        self.include_no_relation = include_no_relation
        self.negative_ratio = negative_ratio
        self.seed = seed
        super().__init__(raw_root, processed_root)

    def _load(self):
        repo, config = BIGBIO_CONFIGS[self.name]
        if self.name == "biored":
            try:
                import bioc  # noqa: F401
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "BioRED requires the optional parser dependency 'bioc'. "
                    "Install/reinstall the project dependencies before preparing BioRED."
                ) from exc
        # BioRED is still script-backed on the Hub and requires trusted remote code.
        if self.name == "biored":
            return load_dataset(repo, config, trust_remote_code=True)

        # Other Sprint-1 BigBio repositories expose parquet-backed configs.
        # Keep a remote-code fallback for compatibility with older cached revisions.
        try:
            return load_dataset(repo, config)
        except (ValueError, FileNotFoundError):
            return load_dataset(repo, config, trust_remote_code=True)

    def prepare(self, force: bool = False):
        ds = self._load()
        output: dict[str, list[RelationExample]] = {}
        rng = random.Random(self.seed)
        for split_name, split_ds in ds.items():
            rows: list[dict[str, Any]] = [dict(row) for row in split_ds]
            labels = observed_relation_labels(rows)
            candidates = labels + (["NO_RELATION"] if self.include_no_relation else [])
            allowed = allowed_type_pairs(rows) if self.include_no_relation else set()
            examples: list[RelationExample] = []
            for row_index, row in enumerate(rows):
                context = flatten_passages(row.get("passages", []))
                for a, b, label, relation_id in relation_pairs(row):
                    examples.append(
                        RelationExample(
                            id=f"{self.name}-{split_name}-{relation_id or row_index}",
                            dataset=self.name,
                            split=split_name,
                            context=context,
                            subject=entity_text(a),
                            object=entity_text(b),
                            subject_type=str(a.get("type", "")) or None,
                            object_type=str(b.get("type", "")) or None,
                            label=label,
                            candidates=candidates,
                            metadata={"document_id": row.get("document_id") or row.get("id")},
                        )
                    )
                if self.include_no_relation:
                    negatives = negative_entity_pairs(row, allowed)
                    positives = max(1, len(relation_pairs(row)))
                    rng.shuffle(negatives)
                    negatives = negatives[: int(positives * self.negative_ratio)]
                    for neg_i, (a, b) in enumerate(negatives):
                        examples.append(
                            RelationExample(
                                id=f"{self.name}-{split_name}-{row_index}-neg-{neg_i}",
                                dataset=self.name,
                                split=split_name,
                                context=context,
                                subject=entity_text(a), object=entity_text(b),
                                subject_type=str(a.get("type", "")) or None,
                                object_type=str(b.get("type", "")) or None,
                                label="NO_RELATION", candidates=candidates,
                                metadata={"document_id": row.get("document_id") or row.get("id")},
                            )
                        )
            output[split_name] = examples
            write_jsonl(self.processed_root / self.name / f"{split_name}.jsonl", examples)
        return output
