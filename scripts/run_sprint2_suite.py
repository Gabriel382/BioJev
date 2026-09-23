#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

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
    p = argparse.ArgumentParser(
        description="Memory-efficient Sprint-2 matrix: load each 4B model once, then evaluate all datasets/seeds."
    )
    p.add_argument("--config", required=True)
    p.add_argument("--max-examples", type=int, default=None)
    args = p.parse_args()

    suite = load_yaml(args.config)
    output_root = ensure_dir(suite.get("output_root", "results/sprint2"))
    max_examples = args.max_examples or suite.get("max_examples")
    summary = []

    for model_entry in suite["models"]:
        model_cfg = load_yaml(model_entry["config"])
        print(f"\n[load model] {model_entry['name']}")
        model = build_model(model_cfg)
        batch_size = int(model_entry.get("batch_size", 1))

        for dataset in suite["datasets"]:
            for seed in suite.get("seeds", [42]):
                set_seed(seed)
                split = dataset.get("split", "test")
                examples = resolve_split(dataset["name"], split, seed)
                if max_examples:
                    examples = examples[: int(max_examples)]
                run_name = f"{model_entry['name']}__{dataset['name']}__{split}__seed{seed}"
                out = Path(output_root) / run_name
                ensure_dir(out)
                print(f"[evaluate] {run_name} n={len(examples):,}")

                if examples[0].task == "nli":
                    records = model.predict_nli(examples, batch_size)
                else:
                    records = model.predict_relations(examples, batch_size)
                metrics = evaluate_records(records)
                write_jsonl(out / "predictions.jsonl", records)
                write_json(out / "metrics.json", metrics)
                write_json(out / "environment.json", snapshot_environment())
                write_json(out / "model_config.json", model_cfg)
                summary.append({"run": run_name, "status": "ok", **metrics})
                print(metrics)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_json(Path(output_root) / "summary.json", summary)
    print(f"\nWrote {Path(output_root) / 'summary.json'}")


if __name__ == "__main__":
    main()
