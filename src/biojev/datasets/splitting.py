from __future__ import annotations

from collections import Counter, defaultdict
from typing import Sequence
import warnings

import numpy as np
from sklearn.model_selection import train_test_split


def _split_indices_safely(indices, labels, *, seed: int, test_size=None, train_size=None):
    """Split indices with stratification when feasible.

    Singleton-label examples are protected in the first partition so a synthetic
    holdout cannot remove the only available training example for a class. This is
    primarily a safety net: official dev/validation/test splits are preferred by
    Sprint 4 and are never replaced merely because a class is rare.
    """
    indices = np.asarray(indices)
    labels = list(labels)
    counts = Counter(labels)

    protected_mask = np.array([counts[label] < 2 for label in labels], dtype=bool)
    protected = indices[protected_mask]
    pool = indices[~protected_mask]
    pool_labels = [label for label, is_protected in zip(labels, protected_mask) if not is_protected]

    if len(pool) == 0:
        return indices, np.array([], dtype=indices.dtype)

    stratify = pool_labels if len(set(pool_labels)) > 1 else None
    kwargs = {"random_state": seed, "stratify": stratify}
    if test_size is not None:
        kwargs["test_size"] = test_size
    if train_size is not None:
        kwargs["train_size"] = train_size

    try:
        first, second = train_test_split(pool, **kwargs)
    except ValueError as exc:
        warnings.warn(
            f"Stratified split is not feasible ({exc}); falling back to a deterministic "
            "unstratified split while keeping singleton classes in training.",
            RuntimeWarning,
            stacklevel=2,
        )
        kwargs["stratify"] = None
        first, second = train_test_split(pool, **kwargs)

    if len(protected):
        first = np.concatenate([np.asarray(first), protected])
    return np.asarray(first), np.asarray(second)


def stratified_split(
    examples: Sequence,
    seed: int = 42,
    dev_size: float = 0.1,
    test_size: float = 0.1,
):
    """Deterministically split a one-split dataset while preserving labels when possible.

    This helper is used only when an official partition is genuinely unavailable.
    Singleton classes stay in training rather than causing sklearn's stratifier to
    crash. On ordinary well-populated datasets the behavior is unchanged from the
    original stratified splitter.
    """
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

    train_idx, hold_idx = _split_indices_safely(
        indices,
        labels,
        seed=seed,
        test_size=holdout,
    )

    if dev_size == 0:
        dev_idx, test_idx = np.array([], dtype=int), hold_idx
    elif test_size == 0:
        dev_idx, test_idx = hold_idx, np.array([], dtype=int)
    elif len(hold_idx) == 0:
        dev_idx, test_idx = np.array([], dtype=int), np.array([], dtype=int)
    else:
        hold_labels = [labels[int(i)] for i in hold_idx]
        dev_fraction = dev_size / holdout
        stratify_hold = (
            hold_labels
            if len(set(hold_labels)) > 1 and min(Counter(hold_labels).values()) >= 2
            else None
        )
        try:
            dev_idx, test_idx = train_test_split(
                hold_idx,
                train_size=dev_fraction,
                random_state=seed,
                stratify=stratify_hold,
            )
        except ValueError as exc:
            warnings.warn(
                f"Holdout stratification is not feasible ({exc}); using deterministic "
                "unstratified dev/test split.",
                RuntimeWarning,
                stacklevel=2,
            )
            dev_idx, test_idx = train_test_split(
                hold_idx,
                train_size=dev_fraction,
                random_state=seed,
                stratify=None,
            )

    result = defaultdict(list)
    for split, idxs in (("train", train_idx), ("dev", dev_idx), ("test", test_idx)):
        for i in idxs:
            result[split].append(examples[int(i)].model_copy(update={"split": split}))
    return {"train": result["train"], "dev": result["dev"], "test": result["test"]}
