from __future__ import annotations

import numpy as np

from biojev.schemas import PredictionRecord


def calibration_metrics(records: list[PredictionRecord], n_bins: int = 15) -> dict[str, float]:
    if not records:
        return {}
    confidences, correct, briers, nlls = [], [], [], []
    for row in records:
        labels = sorted(row.probabilities)
        if row.gold not in row.probabilities:
            continue
        probs = np.array([row.probabilities[x] for x in labels], dtype=float)
        probs = probs / max(probs.sum(), 1e-12)
        gold = np.array([1.0 if x == row.gold else 0.0 for x in labels])
        confidences.append(float(probs.max()))
        correct.append(float(row.prediction == row.gold))
        briers.append(float(np.square(probs - gold).sum()))
        nlls.append(float(-np.log(max(row.probabilities[row.gold], 1e-12))))
    if not confidences:
        return {}
    confidences = np.asarray(confidences)
    correct = np.asarray(correct)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidences[mask].mean())
    return {
        "ece": float(ece),
        "brier": float(np.mean(briers)),
        "nll": float(np.mean(nlls)),
        "mean_confidence": float(confidences.mean()),
    }
