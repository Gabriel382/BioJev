from __future__ import annotations

from biojev.evaluation.calibration import calibration_metrics
from biojev.evaluation.metrics import classification_metrics


def evaluate_records(records):
    return {**classification_metrics(records), **calibration_metrics(records)}
