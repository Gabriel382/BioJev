#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from biojev.config import load_yaml
from biojev.datasets.splitting import stratified_split
from biojev.datasets.store import load_processed
from biojev.evaluation.evaluator import evaluate_records
from biojev.models.registry import build_model
from biojev.utils.environment import snapshot_environment
from biojev.utils.io import ensure_dir, write_json, write_jsonl
from biojev.utils.seed import set_seed


def resolve_split(dataset: str, split: str, seed: int):
    try:
        return load_processed(dataset, split)
    except FileNotFoundError:
        if split not in {"dev", "test"}:
            raise
        train = load_processed(dataset, "train")
        parts = stratified_split(train, seed=seed, dev_size=0.1, test_size=0.1)
        return parts[split]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-config", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--output", default=None)
    args = p.parse_args()
    set_seed(args.seed)

    model_cfg = load_yaml(args.model_config)
    examples = resolve_split(args.dataset, args.split, args.seed)
    if args.max_examples:
        examples = examples[: args.max_examples]
    model = build_model(model_cfg)
    if not examples:
        raise RuntimeError("No examples available")
    if examples[0].task == "nli":
        records = model.predict_nli(examples, args.batch_size)
    else:
        records = model.predict_relations(examples, args.batch_size)
    metrics = evaluate_records(records)
    out = Path(args.output or f"runs/{model.name}__{args.dataset}__{args.split}__seed{args.seed}")
    ensure_dir(out)
    write_jsonl(out / "predictions.jsonl", records)
    write_json(out / "metrics.json", metrics)
    write_json(out / "environment.json", snapshot_environment())
    write_json(out / "model_config.json", model_cfg)
    print(metrics)


if __name__ == "__main__":
    main()
