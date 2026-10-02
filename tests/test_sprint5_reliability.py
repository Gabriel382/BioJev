import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_sprint5_reliability.py"
spec = importlib.util.spec_from_file_location("s5", SCRIPT)
s5 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = s5
spec.loader.exec_module(s5)


def test_perfect_calibration_and_risk():
    conf = np.array([1.0, 1.0, 1.0, 1.0])
    correct = np.array([True, True, True, True])
    _, ece = s5.fixed_reliability(conf, correct, 10)
    _, aurc = s5.risk_coverage(conf, correct, 20)
    assert ece == 0.0
    assert aurc == 0.0


def test_normalize_from_probabilities():
    raw = pd.DataFrame({
        "label": [0, 1, 1],
        "probabilities": [[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]],
    })
    norm, probs, classes, warnings = s5.normalize_predictions(raw)
    assert list(norm.pred) == [0, 1, 0]
    assert np.allclose(norm.confidence, [0.9, 0.8, 0.6])
    assert probs.shape == (3, 2)


def test_brier_nll_binary():
    gold = np.array([0, 1])
    probs = np.array([[0.8, 0.2], [0.25, 0.75]])
    brier, nll, warning = s5.probabilistic_scores(gold, probs, None)
    assert warning is None
    assert 0 < brier < 1
    assert 0 < nll < 1


def test_threshold_coverage_monotone():
    n = pd.DataFrame({
        "gold": [0, 0, 1, 1], "pred": [0, 1, 1, 1],
        "confidence": [0.95, 0.8, 0.7, 0.55], "correct": [True, False, True, True],
    })
    t = s5.threshold_table(n, [0.5, 0.7, 0.9])
    assert list(t.accepted_n) == [4, 3, 1]
