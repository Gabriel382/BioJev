#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _load_summary(path: Path, label: str) -> pd.DataFrame:
    rows = json.loads(path.read_text(encoding="utf-8"))
    df = pd.DataFrame(rows)
    df["model_group"] = label
    return df


def main():
    p = argparse.ArgumentParser(description="Build Qwen vs BioQwen Sprint-2 comparison tables.")
    p.add_argument("--summary", required=True, help="results/sprint2/summary.json")
    p.add_argument("--output", default="results/sprint2/comparison.csv")
    args = p.parse_args()

    rows = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No rows found")
    # Run names use model__dataset__split__seedN.
    parts = df["run"].str.split("__", expand=True)
    df["model"] = parts[0]
    df["dataset"] = parts[1]
    metric_cols = [x for x in ["accuracy", "macro_f1", "micro_f1", "ece", "brier", "nll"] if x in df]
    agg = df.groupby(["model", "dataset"], as_index=False)[metric_cols].agg(["mean", "std"])
    agg.columns = ["_".join([x for x in col if x]) for col in agg.columns.to_flat_index()]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out, index=False)
    print(agg.to_string(index=False))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
