#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from biojev.config import load_yaml
from biojev.utils.io import ensure_dir, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    suite = load_yaml(args.config)
    results = []
    output_root = ensure_dir(suite.get("output_root", "results/suite"))
    for model in suite["models"]:
        for dataset in suite["datasets"]:
            for seed in suite.get("seeds", [42]):
                run_name = f"{model['name']}__{dataset['name']}__{dataset.get('split','test')}__seed{seed}"
                out = output_root / run_name
                cmd = [
                    sys.executable, "scripts/evaluate.py",
                    "--model-config", model["config"],
                    "--dataset", dataset["name"],
                    "--split", dataset.get("split", "test"),
                    "--seed", str(seed),
                    "--batch-size", str(model.get("batch_size", 8)),
                    "--output", str(out),
                ]
                if suite.get("max_examples"):
                    cmd.extend(["--max-examples", str(suite["max_examples"])])
                print("[run]", " ".join(cmd))
                proc = subprocess.run(cmd, text=True)
                status = "ok" if proc.returncode == 0 else "failed"
                metric_path = out / "metrics.json"
                metrics = json.loads(metric_path.read_text()) if metric_path.exists() else {}
                results.append({"run": run_name, "status": status, **metrics})
                if proc.returncode and suite.get("fail_fast", False):
                    raise SystemExit(proc.returncode)
    write_json(output_root / "summary.json", results)
    print(f"Wrote {output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
