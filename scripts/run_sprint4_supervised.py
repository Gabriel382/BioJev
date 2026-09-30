#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from biojev.config import load_yaml
from biojev.datasets.baseline_splits import resolve_baseline_splits
from biojev.utils.io import ensure_dir, write_json


def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _expected_manifest(cache: dict, dataset: str, seed: int):
    key = (dataset, int(seed))
    if key not in cache:
        _, _, _, manifest = resolve_baseline_splits(dataset, int(seed))
        cache[key] = manifest
    return cache[key]


def _existing_state(out: Path, dataset: str, seed: int, expected_cache: dict):
    """Return (state, reason, metrics) for an existing baseline directory.

    Patched runs are validated against split_manifest.json. Legacy pre-patch runs do
    not have that file, so migration validation compares their recorded test ``n``
    against the test size implied by the corrected split policy. This preserves the
    already-valid BioNLI/NLI4CT runs while detecting the old DDI2013 n=403 runs as
    stale (correct official test n=979).
    """
    metric_path = out / "test_metrics.json"
    metrics = _load_json(metric_path)
    if metrics is None:
        return "missing", "no test_metrics.json", {}

    expected = _expected_manifest(expected_cache, dataset, seed)
    expected_n = int(expected["test_count"])
    actual_n = metrics.get("n")
    try:
        actual_n_int = int(actual_n)
    except (TypeError, ValueError):
        return "stale", f"invalid/missing metric n={actual_n!r}; expected {expected_n}", metrics

    if actual_n_int != expected_n:
        return "stale", f"test n={actual_n_int}, corrected policy expects n={expected_n}", metrics

    manifest = _load_json(out / "split_manifest.json")
    if manifest is not None:
        # Exact deterministic provenance comparison for all patched runs.
        comparable_keys = [
            "split_policy_version", "dataset", "seed", "dev_source", "test_source",
            "original_train_count", "train_count", "dev_count", "test_count",
            "train_ids_sha256", "dev_ids_sha256", "test_ids_sha256",
        ]
        mismatches = [
            key for key in comparable_keys if manifest.get(key) != expected.get(key)
        ]
        if mismatches:
            return "stale", "split manifest mismatch: " + ", ".join(mismatches), metrics
        return "valid", "split manifest matches corrected policy", metrics

    # Legacy migration path. For the known pre-patch suite, matching the corrected
    # test cardinality is sufficient to retain BioNLI/NLI4CT and reject DDI2013.
    return "valid", f"legacy result with corrected test n={expected_n}", metrics


def main():
    p = argparse.ArgumentParser(description="Sprint 4 target-supervised encoder baselines.")
    p.add_argument("--config", default="configs/sprint4/supervised_baselines.yaml")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip existing runs only when their test split is compatible with the corrected policy.",
    )
    p.add_argument(
        "--audit-only",
        action="store_true",
        help="Audit all 45 runs and write audit.csv without training or deleting anything.",
    )
    args = p.parse_args()

    cfg = load_yaml(args.config)
    root = ensure_dir(cfg.get("checkpoint_root", "checkpoints/sprint4"))
    summary_root = ensure_dir(cfg.get("summary_root", "results/sprint4/supervised_baselines"))
    expected_cache: dict = {}
    audit_rows = []

    # First audit the complete matrix before mutating anything.
    for model in cfg["models"]:
        for dataset in cfg["datasets"]:
            for seed in cfg.get("seeds", [42]):
                out = root / f"{model['name']}__{dataset}__seed{seed}"
                state, reason, metrics = _existing_state(out, dataset, seed, expected_cache)
                expected = _expected_manifest(expected_cache, dataset, seed)
                audit_rows.append({
                    "model": model["name"],
                    "dataset": dataset,
                    "seed": seed,
                    "state": state,
                    "reason": reason,
                    "existing_n": metrics.get("n") if metrics else None,
                    "expected_test_n": expected["test_count"],
                    "dev_source": expected["dev_source"],
                    "test_source": expected["test_source"],
                })

    audit_df = pd.DataFrame(audit_rows)
    audit_path = Path(summary_root) / "audit.csv"
    audit_df.to_csv(audit_path, index=False)
    counts = audit_df["state"].value_counts().to_dict()
    print(
        "Sprint 4 baseline audit: "
        f"valid={counts.get('valid', 0)} | stale={counts.get('stale', 0)} | "
        f"missing={counts.get('missing', 0)} | total={len(audit_df)}"
    )
    for row in audit_rows:
        if row["state"] != "valid":
            name = f"{row['model']}__{row['dataset']}__seed{row['seed']}"
            print(f"[{row['state']}] {name}: {row['reason']}")
    print(f"Wrote {audit_path}")

    if args.audit_only:
        return

    rows = []
    failures = []

    for model in cfg["models"]:
        for dataset in cfg["datasets"]:
            for seed in cfg.get("seeds", [42]):
                out = root / f"{model['name']}__{dataset}__seed{seed}"
                metric_path = out / "test_metrics.json"
                state, reason, metrics = _existing_state(out, dataset, seed, expected_cache)

                if args.skip_existing and state == "valid":
                    print(f"[skip] {out.name}: {reason}")
                else:
                    # Never allow an old metric/checkpoint to make a failed rerun look
                    # successful. A requested rerun starts from a clean output directory.
                    if out.exists():
                        print(f"[clean] {out.name}: {state} ({reason})")
                        shutil.rmtree(out)

                    cmd = [
                        sys.executable, "scripts/train_baseline.py",
                        "--model", model["repo_id"], "--dataset", dataset,
                        "--output", str(out), "--seed", str(seed),
                        "--epochs", str(cfg.get("epochs", 3)),
                        "--batch-size", str(cfg.get("batch_size", 16)),
                        "--lr", str(cfg.get("learning_rate", 2e-5)),
                        "--max-length", str(cfg.get("max_length", 512)),
                    ]
                    print("[train]", " ".join(cmd), flush=True)
                    proc = subprocess.run(cmd)
                    if proc.returncode:
                        failures.append({
                            "name": out.name,
                            "returncode": proc.returncode,
                        })
                        print(f"[failed] {out.name} (exit={proc.returncode})")
                        if cfg.get("fail_fast", False):
                            raise SystemExit(proc.returncode)

                metrics = _load_json(metric_path) or {}
                final_state, final_reason, _ = _existing_state(out, dataset, seed, expected_cache)
                status = "complete" if final_state == "valid" else final_state
                rows.append({
                    "model": model["name"],
                    "dataset": dataset,
                    "seed": seed,
                    "status": status,
                    "split_status": final_reason,
                    **metrics,
                })

    write_json(Path(root) / "summary.json", rows)
    pd.DataFrame(rows).to_csv(Path(root) / "summary.csv", index=False)
    write_json(Path(summary_root) / "summary.json", rows)
    pd.DataFrame(rows).to_csv(Path(summary_root) / "summary.csv", index=False)

    failed_rows = [row for row in rows if row.get("status") != "complete"]
    failed_path = Path(summary_root) / "failed_runs.txt"
    failed_path.write_text(
        "\n".join(f"{r['model']}__{r['dataset']}__seed{r['seed']}" for r in failed_rows)
        + ("\n" if failed_rows else "")
    )

    print(f"Wrote {Path(summary_root) / 'summary.csv'}")
    print(f"Sprint 4 supervised runs: {len(rows) - len(failed_rows)} complete / {len(rows)} total")
    if failed_rows:
        print(f"WARNING: {len(failed_rows)} run(s) incomplete/stale. See {failed_path}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
