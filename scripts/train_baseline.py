#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from biojev.datasets.splitting import stratified_split
from biojev.datasets.store import load_processed
from biojev.evaluation.evaluator import evaluate_records
from biojev.models.seqcls import SequenceClassifierModel
from biojev.training.seqcls import train_sequence_classifier
from biojev.utils.environment import snapshot_environment
from biojev.utils.io import write_json, write_jsonl
from biojev.utils.seed import set_seed


def _splits(dataset: str, seed: int):
    train = load_processed(dataset, "train")
    try:
        dev = load_processed(dataset, "dev")
    except FileNotFoundError:
        parts = stratified_split(train, seed=seed, dev_size=0.1, test_size=0.1)
        return parts["train"], parts["dev"], parts["test"]
    try:
        test = load_processed(dataset, "test")
    except FileNotFoundError:
        # Do not contaminate dev: split training data into a smaller train/test partition.
        parts = stratified_split(train, seed=seed, dev_size=0.0, test_size=0.1)
        train, test = parts["train"], parts["test"]
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
    train, dev, test = _splits(args.dataset, args.seed)

    result = train_sequence_classifier(
        model_id=args.model,
        train_examples=train,
        dev_examples=dev,
        output_dir=args.output,
        seed=args.seed,
        epochs=args.epochs,
        per_device_batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
    )
    model = SequenceClassifierModel(str(args.output), max_length=args.max_length)
    records = (
        model.predict_nli(test, args.batch_size)
        if test and test[0].task == "nli"
        else model.predict_relations(test, args.batch_size)
    )
    write_jsonl(Path(args.output) / "test_predictions.jsonl", records)
    write_json(Path(args.output) / "test_metrics.json", evaluate_records(records))
    write_json(Path(args.output) / "environment.json", snapshot_environment())
    print({"checkpoint": result.checkpoint, "test_metrics": evaluate_records(records)})


if __name__ == "__main__":
    main()
