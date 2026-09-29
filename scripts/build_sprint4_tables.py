#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _load_summaries(root: Path) -> pd.DataFrame:
    rows = []
    for path in root.rglob("summary.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, list):
            for row in data:
                if isinstance(row, dict) and "model" in row and "dataset" in row:
                    rows.append({"source_file": str(path), **row})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description="Build Sprint-4 paper-ready CSV tables.")
    p.add_argument("--root", default="results/sprint4")
    p.add_argument("--output", default="results/sprint4/tables")
    args = p.parse_args()
    root = Path(args.root)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    df = _load_summaries(root)
    if df.empty:
        raise SystemExit(f"No Sprint-4 summary.json files found under {root}")
    df.to_csv(out / "all_results.csv", index=False)

    metric = "f1_macro" if "f1_macro" in df.columns else "accuracy"
    frozen = df[df["source_file"].str.contains("frozen_")].copy()
    if not frozen.empty:
        pivot = frozen.pivot_table(index="model", columns="dataset", values=metric, aggfunc="first")
        pivot.to_csv(out / "table_frozen_transfer.csv")

    supervised = df[df["source_file"].str.contains("supervised_baselines")].copy()
    if not supervised.empty:
        sup_metric = "f1_macro" if "f1_macro" in supervised.columns else metric
        sup = supervised.groupby(["model", "dataset"], as_index=False)[sup_metric].agg(["mean", "std"]).reset_index()
        sup.to_csv(out / "table_supervised_baselines.csv", index=False)

    cross = df[df["source_file"].str.contains("cross_nli_")].copy()
    if not cross.empty:
        def source_from_model(name: str):
            if "bionli-only" in name: return "bionli"
            if "nli4ct-only" in name: return "nli4ct"
            return "unknown"
        cross["train_source"] = cross["model"].map(source_from_model)
        cross["is_cross_dataset"] = cross["train_source"] != cross["dataset"]
        cross.to_csv(out / "cross_nli_long.csv", index=False)
        pivot = cross.pivot_table(index=["model", "train_source"], columns="dataset", values=metric, aggfunc="first")
        pivot.to_csv(out / "table_cross_nli.csv")

    print(f"Wrote Sprint-4 tables to {out}")


if __name__ == "__main__":
    main()
