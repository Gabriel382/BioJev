from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from biojev.datasets.splitting import stratified_split
from biojev.datasets.store import load_processed
from biojev.schemas import NLIExample


CANONICAL_LABELS = ("contradiction", "entailment", "neutral")
LABEL2ID = {label: i for i, label in enumerate(CANONICAL_LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

_LABEL_ALIASES = {
    "contradiction": "contradiction",
    "contradictory": "contradiction",
    "entailment": "entailment",
    "entails": "entailment",
    "entailed": "entailment",
    "neutral": "neutral",
}


def normalize_nli_label(label: str) -> str:
    key = str(label).strip().lower()
    if key not in _LABEL_ALIASES:
        raise ValueError(f"Unsupported NLI label: {label!r}")
    return _LABEL_ALIASES[key]


def normalize_example(example: NLIExample) -> NLIExample:
    return example.model_copy(update={"label": normalize_nli_label(example.label)})


def _resolve_dataset_splits(name: str, seed: int) -> dict[str, list[NLIExample]]:
    """Load train/dev/test without leaking a one-split corpus into evaluation."""
    try:
        train = [normalize_example(x) for x in load_processed(name, "train")]
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Decision dataset {name!r} is not prepared. Run scripts/download_datasets.py first."
        ) from exc
    try:
        dev = [normalize_example(x) for x in load_processed(name, "dev")]
    except FileNotFoundError:
        try:
            dev = [normalize_example(x) for x in load_processed(name, "validation")]
        except FileNotFoundError:
            dev = []
    try:
        test = [normalize_example(x) for x in load_processed(name, "test")]
    except FileNotFoundError:
        test = []

    if not dev and not test:
        return stratified_split(train, seed=seed, dev_size=0.1, test_size=0.1)
    if not dev:
        parts = stratified_split(train, seed=seed, dev_size=0.1, test_size=0.0)
        train, dev = parts["train"], parts["dev"]
    return {"train": train, "dev": dev, "test": test}


def _sample(examples: list[NLIExample], n: int, rng: np.random.Generator) -> list[NLIExample]:
    if n <= 0 or not examples:
        return []
    replace = n > len(examples)
    indices = rng.choice(len(examples), size=n, replace=replace)
    return [examples[int(i)] for i in indices]


@dataclass
class DecisionMixture:
    train: list[NLIExample]
    dev: list[NLIExample]
    manifest: dict[str, Any]


def build_decision_mixture(
    dataset_cfgs: list[dict[str, Any]],
    *,
    seed: int,
    epoch_examples: int | None = None,
    dev_max_per_dataset: int | None = 2000,
) -> DecisionMixture:
    """Build a deterministic weighted NLI mixture.

    `weight` controls the share of examples in an epoch. Sampling with replacement is
    intentional when a small biomedical dataset is assigned a large weight.
    """
    if not dataset_cfgs:
        raise ValueError("At least one decision dataset is required")
    rng = np.random.default_rng(seed)
    prepared = []
    total_available = 0
    for item in dataset_cfgs:
        name = str(item["name"])
        splits = _resolve_dataset_splits(name, seed)
        train = list(splits["train"])
        dev = list(splits["dev"])
        max_train = item.get("max_train_examples")
        if max_train is not None and len(train) > int(max_train):
            train = _sample(train, int(max_train), rng)
        prepared.append((item, train, dev))
        total_available += len(train)

    target_total = int(epoch_examples or total_available)
    if target_total <= 0:
        raise ValueError("Decision mixture has no training examples")
    weights = np.array([float(item.get("weight", 1.0)) for item, _, _ in prepared], dtype=float)
    if np.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError("Dataset weights must be non-negative and sum to > 0")
    weights = weights / weights.sum()

    counts = np.floor(weights * target_total).astype(int)
    remainder = target_total - int(counts.sum())
    fractional = weights * target_total - counts
    for idx in np.argsort(-fractional)[:remainder]:
        counts[int(idx)] += 1

    train_mix: list[NLIExample] = []
    dev_mix: list[NLIExample] = []
    manifest_sources = []
    for (item, train, dev), count in zip(prepared, counts):
        sampled_train = _sample(train, int(count), rng)
        if dev_max_per_dataset is not None and len(dev) > int(dev_max_per_dataset):
            sampled_dev = _sample(dev, int(dev_max_per_dataset), rng)
        else:
            sampled_dev = dev
        train_mix.extend(sampled_train)
        dev_mix.extend(sampled_dev)
        manifest_sources.append(
            {
                "name": item["name"],
                "weight": float(item.get("weight", 1.0)),
                "available_train": len(train),
                "sampled_train": len(sampled_train),
                "dev": len(sampled_dev),
            }
        )

    rng.shuffle(train_mix)
    rng.shuffle(dev_mix)
    return DecisionMixture(
        train=train_mix,
        dev=dev_mix,
        manifest={
            "epoch_examples": len(train_mix),
            "dev_examples": len(dev_mix),
            "sources": manifest_sources,
        },
    )
