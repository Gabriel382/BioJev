#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import torch

from biojev.config import load_yaml
from biojev.evaluation.sprint4 import detailed_metrics, prediction_method, resolve_eval_split
from biojev.models.registry import build_model
from biojev.utils.environment import snapshot_environment
from biojev.utils.io import ensure_dir, write_json, write_jsonl
from biojev.utils.seed import set_seed


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    scalar_keys = []
    seen = set()
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                if key not in seen:
                    scalar_keys.append(key); seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in scalar_keys})


def main():
    p = argparse.ArgumentParser(description="Sprint 4 frozen multi-dataset benchmark.")
    p.add_argument("--config", required=True)
    p.add_argument("--model", action="append", default=[], help="Run only named model(s).")
    p.add_argument("--dataset", action="append", default=[], help="Run only named dataset(s).")
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    root = ensure_dir(cfg.get("output_root", "results/sprint4/frozen"))
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    requested_models = set(args.model)
    requested_datasets = set(args.dataset)
    rows: list[dict] = []

    for model_spec in cfg["models"]:
        model_name = model_spec["name"]
        if requested_models and model_name not in requested_models:
            continue
        print(f"\n[Sprint 4] loading model once: {model_name}")
        model_cfg = load_yaml(model_spec["config"])
        model = build_model(model_cfg)
        try:
            for ds_spec in cfg["datasets"]:
                ds_name = ds_spec["name"]
                if requested_datasets and ds_name not in requested_datasets:
                    continue
                split = ds_spec.get("split", "test")
                role = ds_spec.get("role", "evaluation")
                run_dir = root / f"{model_name}__{ds_name}__{split}__seed{seed}"
                metric_path = run_dir / "metrics.json"
                if args.skip_existing and metric_path.exists():
                    print(f"  [skip] {ds_name}/{split}")
                    metrics = json.loads(metric_path.read_text(encoding="utf-8"))
                else:
                    examples = resolve_eval_split(ds_name, split, seed)
                    max_examples = ds_spec.get("max_examples", args.max_examples)
                    if max_examples:
                        examples = examples[: int(max_examples)]
                    mode = prediction_method(examples)
                    batch_size = int(ds_spec.get("batch_size", model_spec.get("batch_size", 8)))
                    print(f"  [eval] {ds_name}/{split}: {len(examples):,} examples ({mode})")
                    if mode == "nli":
                        records = model.predict_nli(examples, batch_size)
                    else:
                        records = model.predict_relations(examples, batch_size)
                    metrics = detailed_metrics(records)
                    ensure_dir(run_dir)
                    write_jsonl(run_dir / "predictions.jsonl", records)
                    write_json(metric_path, metrics)
                    write_json(run_dir / "model_config.json", model_cfg)
                    write_json(run_dir / "environment.json", snapshot_environment())
                    write_json(run_dir / "protocol.json", {
                        "sprint": 4,
                        "regime": "frozen_task_general",
                        "model": model_name,
                        "dataset": ds_name,
                        "split": split,
                        "role": role,
                        "target_task_gradient_updates": False,
                        "relation_scope": (
                            "gold_entity_pair_relation_typing" if mode == "relation_classification" else "nli"
                        ),
                    })
                rows.append({
                    "model": model_name, "dataset": ds_name, "split": split, "role": role,
                    **{k: v for k, v in metrics.items() if isinstance(v, (int, float, str, bool))},
                })
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_json(root / "summary.json", rows)
    _write_csv(root / "summary.csv", rows)
    print(f"\nWrote {root / 'summary.csv'}")


if __name__ == "__main__":
    main()
