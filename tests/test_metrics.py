from biojev.evaluation.evaluator import evaluate_records
from biojev.schemas import PredictionRecord


def test_metrics_perfect():
    rows = [
        PredictionRecord(id="1", dataset="x", split="test", task="nli", gold="entailment",
                         prediction="entailment", probabilities={"entailment": 0.9, "contradiction": 0.1}),
        PredictionRecord(id="2", dataset="x", split="test", task="nli", gold="contradiction",
                         prediction="contradiction", probabilities={"entailment": 0.1, "contradiction": 0.9}),
    ]
    metrics = evaluate_records(rows)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_macro"] == 1.0
    assert metrics["brier"] > 0
