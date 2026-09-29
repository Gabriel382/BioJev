from pathlib import Path

from biojev.config import load_yaml
from biojev.evaluation.sprint4 import detailed_metrics
from biojev.schemas import PredictionRecord


def test_sprint4_configs_reference_expected_datasets():
    cfg = load_yaml("configs/sprint4/frozen_4b.yaml")
    assert [x["name"] for x in cfg["datasets"]] == [
        "bionli", "nli4ct", "chemprot", "ddi2013", "biored"
    ]
    assert any(x["name"] == "biojev-4b" for x in cfg["models"])


def test_cross_configs_preserve_original_source_exposure():
    bio = load_yaml("configs/decision/4b_bionli_only.yaml")
    ct = load_yaml("configs/decision/4b_nli4ct_only.yaml")
    assert bio["stages"][1]["epoch_examples"] == 30000
    assert ct["stages"][1]["epoch_examples"] == 10000
    assert bio["stages"][0]["epoch_examples"] == 100000
    assert ct["stages"][0]["epoch_examples"] == 100000


def test_detailed_metrics_contains_per_class_and_confusion_matrix():
    rows = [
        PredictionRecord(id="1", dataset="x", split="test", task="nli", gold="entailment", prediction="entailment", probabilities={"entailment":0.8,"contradiction":0.2}),
        PredictionRecord(id="2", dataset="x", split="test", task="nli", gold="contradiction", prediction="entailment", probabilities={"entailment":0.7,"contradiction":0.3}),
    ]
    metrics = detailed_metrics(rows)
    assert "per_class" in metrics
    assert len(metrics["confusion_matrix"]) == 2
    assert metrics["n"] == 2


def test_sprint4_walkthroughs_exist():
    assert Path("notebooks/04_sprint4_walkthrough.py").exists()
    assert Path("notebooks/04_sprint4_walkthrough.ipynb").exists()
