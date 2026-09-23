from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
from sklearn.model_selection import train_test_split


def stratified_split(
    examples: Sequence,
    seed: int = 42,
    dev_size: float = 0.1,
    test_size: float = 0.1,
):
    """Deterministically split a one-split dataset while preserving labels when possible."""
    examples = list(examples)
    if not examples:
        return {"train": [], "dev": [], "test": []}
    if dev_size < 0 or test_size < 0 or dev_size + test_size >= 1:
        raise ValueError("dev_size and test_size must be >=0 and sum to <1")
    labels = [x.label for x in examples]
    indices = np.arange(len(examples))
    holdout = dev_size + test_size
    if holdout == 0:
        return {"train": examples, "dev": [], "test": []}
    stratify = labels if len(set(labels)) > 1 else None
    train_idx, hold_idx = train_test_split(
        indices, test_size=holdout, random_state=seed, stratify=stratify
    )
    if dev_size == 0:
        dev_idx, test_idx = [], hold_idx
    elif test_size == 0:
        dev_idx, test_idx = hold_idx, []
    else:
        hold_labels = [labels[int(i)] for i in hold_idx]
        dev_fraction = dev_size / holdout
        stratify_hold = hold_labels if len(set(hold_labels)) > 1 else None
        dev_idx, test_idx = train_test_split(
            hold_idx, train_size=dev_fraction, random_state=seed, stratify=stratify_hold
        )
    result = defaultdict(list)
    for split, idxs in (("train", train_idx), ("dev", dev_idx), ("test", test_idx)):
        for i in idxs:
            result[split].append(examples[int(i)].model_copy(update={"split": split}))
    return {"train": result["train"], "dev": result["dev"], "test": result["test"]}
