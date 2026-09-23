from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_recall_fscore_support

from biojev.schemas import PredictionRecord


def classification_metrics(records: list[PredictionRecord]) -> dict[str, float]:
    if not records:
        return {}
    gold = [r.gold for r in records]
    pred = [r.prediction for r in records]
    labels = sorted(set(gold) | set(pred))
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        gold, pred, labels=labels, average="macro", zero_division=0
    )
    return {
        "n": len(records),
        "accuracy": float(accuracy_score(gold, pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "f1_micro": float(f1_score(gold, pred, labels=labels, average="micro", zero_division=0)),
    }
