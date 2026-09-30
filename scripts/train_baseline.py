#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from biojev.datasets.baseline_splits import resolve_baseline_splits
from biojev.evaluation.evaluator import evaluate_records
from biojev.models.seqcls import SequenceClassifierModel
from biojev.training.seqcls import train_sequence_classifier
from biojev.utils.environment import snapshot_environment
from biojev.utils.io import ensure_dir, write_json, write_jsonl
from biojev.utils.seed import set_seed


def _splits(dataset: str, seed: int):
    """Backward-compatible wrapper returning only train/dev/test."""
    train, dev, test, _ = resolve_baseline_splits(dataset, seed)
    return train, dev, test


def main():
    p = argparse.ArgumentParser(description="Train and evaluate a conventional supervised encoder baseline.")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=512)
    args = p.parse_args()

    set_seed(args.seed)
    train, dev, test, manifest = resolve_baseline_splits(args.dataset, args.seed)

    print(
        "  splits: "
        f"train={len(train):,} | dev={len(dev):,} ({manifest['dev_source']}) | "
        f"test={len(test):,} ({manifest['test_source']})"
    )

    # Write provenance before expensive training. A failed run therefore still records
    # which split policy it attempted, while completion is determined by test_metrics.json.
    out = ensure_dir(args.output)
    write_json(Path(out) / "split_manifest.json", manifest)

    result = train_sequence_classifier(
        model_id=args.model,
        train_examples=train,
        dev_examples=dev,
        output_dir=out,
        seed=args.seed,
        epochs=args.epochs,
        per_device_batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
    )
    model = SequenceClassifierModel(str(out), max_length=args.max_length)
    records = (
        model.predict_nli(test, args.batch_size)
        if test and test[0].task == "nli"
        else model.predict_relations(test, args.batch_size)
    )
    metrics = evaluate_records(records)
    write_jsonl(Path(out) / "test_predictions.jsonl", records)
    write_json(Path(out) / "test_metrics.json", metrics)
    write_json(Path(out) / "environment.json", snapshot_environment())
    print({"checkpoint": result.checkpoint, "split_manifest": manifest, "test_metrics": metrics})


if __name__ == "__main__":
    main()
