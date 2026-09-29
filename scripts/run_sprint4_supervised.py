#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from biojev.config import load_yaml
from biojev.utils.io import ensure_dir, write_json


def main():
    p = argparse.ArgumentParser(description="Sprint 4 target-supervised encoder baselines.")
    p.add_argument("--config", default="configs/sprint4/supervised_baselines.yaml")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()
    cfg = load_yaml(args.config)
    root = ensure_dir(cfg.get("checkpoint_root", "checkpoints/sprint4"))
    rows = []
    for model in cfg["models"]:
        for dataset in cfg["datasets"]:
            for seed in cfg.get("seeds", [42]):
                out = root / f"{model['name']}__{dataset}__seed{seed}"
                metric_path = out / "test_metrics.json"
                if args.skip_existing and metric_path.exists():
                    print(f"[skip] {out.name}")
                else:
                    cmd = [
                        sys.executable, "scripts/train_baseline.py",
                        "--model", model["repo_id"], "--dataset", dataset,
                        "--output", str(out), "--seed", str(seed),
                        "--epochs", str(cfg.get("epochs", 3)),
                        "--batch-size", str(cfg.get("batch_size", 16)),
                        "--lr", str(cfg.get("learning_rate", 2e-5)),
                        "--max-length", str(cfg.get("max_length", 512)),
                    ]
                    print("[train]", " ".join(cmd))
                    proc = subprocess.run(cmd)
                    if proc.returncode and cfg.get("fail_fast", False):
                        raise SystemExit(proc.returncode)
                metrics = json.loads(metric_path.read_text()) if metric_path.exists() else {}
                rows.append({"model":model["name"],"dataset":dataset,"seed":seed,**metrics})
    write_json(Path(root) / "summary.json", rows)
    pd.DataFrame(rows).to_csv(Path(root) / "summary.csv", index=False)
    summary_root = ensure_dir(cfg.get("summary_root", "results/sprint4/supervised_baselines"))
    write_json(Path(summary_root) / "summary.json", rows)
    pd.DataFrame(rows).to_csv(Path(summary_root) / "summary.csv", index=False)
    print(f"Wrote {Path(summary_root) / 'summary.csv'}")


if __name__ == "__main__":
    main()
