#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from biojev.config import load_yaml
from biojev.utils.io import ensure_dir


def main():
    p = argparse.ArgumentParser(description="Train conventional biomedical encoder baselines per dataset.")
    p.add_argument("--config", default="configs/suites/encoder_baselines.yaml")
    args = p.parse_args()
    cfg = load_yaml(args.config)
    root = ensure_dir(cfg.get("checkpoint_root", "checkpoints"))
    for model in cfg["models"]:
        for dataset in cfg["datasets"]:
            for seed in cfg.get("seeds", [42]):
                out = root / f"{model['name']}__{dataset}__seed{seed}"
                cmd = [
                    sys.executable, "scripts/train_baseline.py",
                    "--model", model["repo_id"],
                    "--dataset", dataset,
                    "--output", str(out),
                    "--seed", str(seed),
                    "--epochs", str(cfg.get("epochs", 3)),
                    "--batch-size", str(cfg.get("batch_size", 16)),
                    "--lr", str(cfg.get("learning_rate", 2e-5)),
                    "--max-length", str(cfg.get("max_length", 512)),
                ]
                print("[train]", " ".join(cmd))
                subprocess.run(cmd, check=cfg.get("fail_fast", False))


if __name__ == "__main__":
    main()
