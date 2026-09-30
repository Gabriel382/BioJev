from __future__ import annotations

import hashlib
from typing import Any

from biojev.datasets.splitting import stratified_split
from biojev.datasets.store import load_processed


SPLIT_POLICY_VERSION = "sprint4-baseline-v3"


def _load_optional(dataset: str, *names: str):
    for name in names:
        try:
            return load_processed(dataset, name), name
        except FileNotFoundError:
            continue
    return None, None


def _id_hash(examples) -> str:
    payload = "\n".join(str(x.id) for x in examples).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _manifest(
    *,
    dataset: str,
    seed: int,
    train,
    dev,
    test,
    dev_source: str,
    test_source: str,
    original_train_count: int,
) -> dict[str, Any]:
    return {
        "split_policy_version": SPLIT_POLICY_VERSION,
        "dataset": dataset,
        "seed": int(seed),
        "dev_source": dev_source,
        "test_source": test_source,
        "original_train_count": int(original_train_count),
        "train_count": len(train),
        "dev_count": len(dev),
        "test_count": len(test),
        "train_ids_sha256": _id_hash(train),
        "dev_ids_sha256": _id_hash(dev),
        "test_ids_sha256": _id_hash(test),
    }


def resolve_baseline_splits(dataset: str, seed: int):
    """Resolve supervised-baseline train/dev/test partitions safely.

    Split policy:
      1. Always use the official training split as the source of training data.
      2. For development, prefer official ``dev`` then official ``validation``.
      3. For test, always preserve official ``test`` when it exists.
      4. Synthesize *only* the partition that is genuinely missing.

    Consequences for the current Sprint-4 datasets:
      - BioNLI: no official dev/test -> deterministic synthetic dev+test from train.
      - NLI4CT: official dev, no labeled test -> synthetic test from train; dev preserved.
      - ChemProt: official validation + test -> both preserved.
      - DDI2013: official test, no dev -> synthetic dev from train; official test preserved.
      - BioRED: official validation + test -> both preserved.
    """
    original_train = load_processed(dataset, "train")
    original_train_count = len(original_train)
    dev, dev_name = _load_optional(dataset, "dev", "validation")
    test, test_name = _load_optional(dataset, "test")

    if dev is not None and test is not None:
        train = original_train
        dev_source = str(dev_name)
        test_source = str(test_name)

    elif dev is None and test is not None:
        parts = stratified_split(original_train, seed=seed, dev_size=0.1, test_size=0.0)
        train, dev = parts["train"], parts["dev"]
        dev_source = "synthetic_from_train"
        test_source = str(test_name)

    elif dev is not None and test is None:
        parts = stratified_split(original_train, seed=seed, dev_size=0.0, test_size=0.1)
        train, test = parts["train"], parts["test"]
        dev_source = str(dev_name)
        test_source = "synthetic_from_train"

    else:
        parts = stratified_split(original_train, seed=seed, dev_size=0.1, test_size=0.1)
        train, dev, test = parts["train"], parts["dev"], parts["test"]
        dev_source = "synthetic_from_train"
        test_source = "synthetic_from_train"

    manifest = _manifest(
        dataset=dataset,
        seed=seed,
        train=train,
        dev=dev,
        test=test,
        dev_source=dev_source,
        test_source=test_source,
        original_train_count=original_train_count,
    )
    return train, dev, test, manifest
