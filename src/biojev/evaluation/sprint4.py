from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from biojev.datasets.splitting import stratified_split
from biojev.datasets.store import load_processed
from biojev.evaluation.evaluator import evaluate_records


def resolve_eval_split(dataset: str, split: str, seed: int = 42):
    """Resolve a deterministic evaluation split without changing Sprint-1 storage.

    BioNLI is distributed as one split in the current mirror. The same deterministic
    stratified splitter used by Sprint 3 is therefore reused here so its test partition
    stays held out from decision training.
    """
    try:
        return load_processed(dataset, split)
    except FileNotFoundError:
        if split not in {"dev", "test"}:
            raise
        train = load_processed(dataset, "train")
        parts = stratified_split(train, seed=seed, dev_size=0.1, test_size=0.1)
        return parts[split]


def detailed_metrics(records) -> dict[str, Any]:
    metrics: dict[str, Any] = dict(evaluate_records(records))
    if not records:
        return metrics
    gold = [r.gold for r in records]
    pred = [r.prediction for r in records]
    labels = sorted(set(gold) | set(pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        gold, pred, labels=labels, average=None, zero_division=0
    )
    metrics["labels"] = labels
    metrics["per_class"] = {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    }
    metrics["confusion_matrix"] = confusion_matrix(gold, pred, labels=labels).tolist()
    metrics["gold_distribution"] = dict(Counter(gold))
    metrics["pred_distribution"] = dict(Counter(pred))
    return metrics


def prediction_method(examples) -> str:
    if not examples:
        raise RuntimeError("No examples available")
    return "nli" if examples[0].task == "nli" else "relation_classification"
