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


PROVENANCE = {
    "general_only": {"dapt": False, "general_nli": True, "biomedical_nli": False, "biomedical_source": "none"},
    "biomedical_only": {"dapt": False, "general_nli": False, "biomedical_nli": True, "biomedical_source": "bionli+nli4ct"},
    "dapt_general": {"dapt": True, "general_nli": True, "biomedical_nli": False, "biomedical_source": "none"},
    "dapt_biomedical": {"dapt": True, "general_nli": False, "biomedical_nli": True, "biomedical_source": "bionli+nli4ct"},
    "general_biomedical": {"dapt": False, "general_nli": True, "biomedical_nli": True, "biomedical_source": "bionli+nli4ct"},
    "full": {"dapt": True, "general_nli": True, "biomedical_nli": True, "biomedical_source": "bionli+nli4ct"},
    "bionli_only": {"dapt": True, "general_nli": True, "biomedical_nli": True, "biomedical_source": "bionli"},
    "nli4ct_only": {"dapt": True, "general_nli": True, "biomedical_nli": True, "biomedical_source": "nli4ct"},
}


def seen_status(variant: str, dataset: str) -> str:
    p = PROVENANCE[variant]
    if dataset in {"chemprot", "ddi2013", "biored"}:
        return "unseen_target_task"
    if dataset == "bionli":
        return "seen_in_decision_training" if p["biomedical_source"] in {"bionli+nli4ct", "bionli"} else "unseen_target_task"
    if dataset == "nli4ct":
        return "seen_in_decision_training" if p["biomedical_source"] in {"bionli+nli4ct", "nli4ct"} else "unseen_target_task"
    return "unknown"


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k, v in row.items()
                              if isinstance(v, (str, int, float, bool)) or v is None))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in keys})


def main():
    p = argparse.ArgumentParser(description="Sprint 6: frozen evaluation of Nano ablation variants.")
    p.add_argument("--config", default="configs/sprint6/eval_nano.yaml")
    p.add_argument("--model", action="append", default=[])
    p.add_argument("--dataset", action="append", default=[])
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--audit-only", action="store_true")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    root = ensure_dir(cfg.get("output_root", "results/sprint6/nano_ablation_eval"))
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    selected_models = set(args.model)
    selected_datasets = set(args.dataset)

    audit_rows = []
    for model_spec in cfg["models"]:
        name = model_spec["name"]
        if selected_models and name not in selected_models:
            continue
        mcfg = load_yaml(model_spec["config"])
        checkpoint = Path(mcfg["params"]["checkpoint"])
        audit_rows.append({
            "variant": name,
            "status": "ready" if checkpoint.exists() else "missing",
            "checkpoint": str(checkpoint),
        })
    _write_csv(Path(root) / "input_audit.csv", audit_rows)
    missing = [r for r in audit_rows if r["status"] != "ready"]
    print(f"Sprint 6 eval audit: ready={len(audit_rows)-len(missing)} | missing={len(missing)}")
    for row in missing:
        print(f"  [missing] {row['variant']}: {row['checkpoint']}")
    if args.audit_only:
        if missing:
            raise SystemExit(2)
        return
    if missing:
        raise SystemExit("Evaluation audit failed; finish missing training variants first.")

    rows = []
    for model_spec in cfg["models"]:
        name = model_spec["name"]
        if selected_models and name not in selected_models:
            continue
        print(f"\n[Sprint 6] loading {name}")
        model_cfg = load_yaml(model_spec["config"])
        model = build_model(model_cfg)
        try:
            for ds_spec in cfg["datasets"]:
                ds_name = ds_spec["name"]
                if selected_datasets and ds_name not in selected_datasets:
                    continue
                split = ds_spec.get("split", "test")
                run_dir = Path(root) / f"{name}__{ds_name}__{split}__seed{seed}"
                metric_path = run_dir / "metrics.json"
                pred_path = run_dir / "predictions.jsonl"
                if args.skip_existing and metric_path.exists() and pred_path.exists():
                    metrics = json.loads(metric_path.read_text(encoding="utf-8"))
                    print(f"  [skip] {ds_name}/{split}")
                else:
                    examples = resolve_eval_split(ds_name, split, seed)
                    mode = prediction_method(examples)
                    batch_size = int(ds_spec.get("batch_size", model_spec.get("batch_size", 16)))
                    print(f"  [eval] {ds_name}/{split}: {len(examples):,} examples ({mode})")
                    records = model.predict_nli(examples, batch_size) if mode == "nli" else model.predict_relations(examples, batch_size)
                    metrics = detailed_metrics(records)
                    ensure_dir(run_dir)
                    write_jsonl(pred_path, records)
                    write_json(metric_path, metrics)
                    write_json(run_dir / "model_config.json", model_cfg)
                    write_json(run_dir / "environment.json", snapshot_environment())
                    write_json(run_dir / "protocol.json", {
                        "sprint": 6,
                        "regime": "nano_ablation",
                        "variant": name,
                        "dataset": ds_name,
                        "split": split,
                        "seed": seed,
                        "seen_status": seen_status(name, ds_name),
                        **PROVENANCE[name],
                    })
                rows.append({
                    "variant": name,
                    "dataset": ds_name,
                    "split": split,
                    "seed": seed,
                    "seen_status": seen_status(name, ds_name),
                    **PROVENANCE[name],
                    **{k: v for k, v in metrics.items() if isinstance(v, (int, float, str, bool))},
                })
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_json(Path(root) / "summary.json", rows)
    _write_csv(Path(root) / "summary.csv", rows)
    print(f"\nWrote {Path(root) / 'summary.csv'}")


if __name__ == "__main__":
    main()
