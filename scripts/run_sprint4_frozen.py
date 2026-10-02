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


def _verify_backfill_metrics(existing: dict, recomputed: dict, tol: float = 1e-6) -> list[str]:
    """Return human-readable scalar metric mismatches beyond tolerance."""
    keys = (
        "n", "accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_micro",
        "ece", "brier", "nll", "mean_confidence",
    )
    diffs: list[str] = []
    for key in keys:
        if key not in existing or key not in recomputed:
            continue
        a, b = existing[key], recomputed[key]
        if isinstance(a, bool) or isinstance(b, bool):
            continue
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if key == "n":
                if int(a) != int(b):
                    diffs.append(f"{key}: existing={a} recomputed={b}")
            elif abs(float(a) - float(b)) > tol:
                diffs.append(f"{key}: existing={a:.12g} recomputed={b:.12g}")
    return diffs


def main():
    p = argparse.ArgumentParser(description="Sprint 4 frozen multi-dataset benchmark.")
    p.add_argument("--config", required=True)
    p.add_argument("--model", action="append", default=[], help="Run only named model(s).")
    p.add_argument("--dataset", action="append", default=[], help="Run only named dataset(s).")
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument(
        "--backfill-metric-tolerance",
        type=float,
        default=1e-6,
        help="Maximum absolute scalar-metric difference allowed when backfilling missing predictions for an existing run.",
    )
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
                prediction_path = run_dir / "predictions.jsonl"
                has_metrics = metric_path.exists()
                has_predictions = prediction_path.exists()
                backfill_predictions = bool(args.skip_existing and has_metrics and not has_predictions)

                if args.skip_existing and has_metrics and has_predictions:
                    print(f"  [skip] {ds_name}/{split} (metrics + predictions present)")
                    metrics = json.loads(metric_path.read_text(encoding="utf-8"))
                else:
                    if backfill_predictions:
                        print(f"  [backfill] {ds_name}/{split}: metrics exist, predictions missing")
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
                    recomputed_metrics = detailed_metrics(records)
                    ensure_dir(run_dir)

                    if backfill_predictions:
                        existing_metrics = json.loads(metric_path.read_text(encoding="utf-8"))
                        diffs = _verify_backfill_metrics(
                            existing_metrics, recomputed_metrics, tol=float(args.backfill_metric_tolerance)
                        )
                        if diffs:
                            candidate_predictions = run_dir / "predictions.backfill_candidate.jsonl"
                            candidate_metrics = run_dir / "metrics.backfill_candidate.json"
                            write_jsonl(candidate_predictions, records)
                            write_json(candidate_metrics, recomputed_metrics)
                            details = "\n      ".join(diffs)
                            raise RuntimeError(
                                f"Backfill verification failed for {model_name}/{ds_name}/{split}. "
                                f"Existing Sprint-4 metrics disagree with recomputed inference:\n      {details}\n"
                                f"Candidate artifacts were written to {candidate_predictions} and {candidate_metrics}; "
                                "the canonical predictions.jsonl was NOT created."
                            )
                        write_jsonl(prediction_path, records)
                        metrics = existing_metrics
                        write_json(run_dir / "prediction_backfill_verification.json", {
                            "status": "verified",
                            "metric_tolerance": float(args.backfill_metric_tolerance),
                            "existing_metrics_preserved": True,
                        })
                        print(f"  [backfill-ok] {ds_name}/{split}: predictions restored; existing metrics preserved")
                    else:
                        metrics = recomputed_metrics
                        write_jsonl(prediction_path, records)
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
