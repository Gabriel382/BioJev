#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["accuracy", "f1_macro", "ece", "adaptive_ece", "brier", "nll", "mean_confidence", "aurc"]


def fmt(mean, std, pct=False):
    if pd.isna(mean):
        return "—"
    scale = 100 if pct else 1
    if pd.isna(std) or abs(float(std)) < 1e-15:
        return f"{mean*scale:.2f}"
    return f"{mean*scale:.2f} ± {std*scale:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/sprint5")
    args = ap.parse_args()
    root = Path(args.results)
    out = root / "tables"
    out.mkdir(parents=True, exist_ok=True)

    s = pd.read_csv(root / "summary.csv")
    rows = []
    for (regime, model, dataset), g in s.groupby(["regime", "model", "dataset"]):
        row = {"regime": regime, "model": model, "dataset": dataset, "runs": len(g)}
        for m in METRICS:
            row[f"{m}_mean"] = g[m].mean()
            row[f"{m}_std"] = g[m].std(ddof=1) if len(g) > 1 else np.nan
        rows.append(row)
    agg = pd.DataFrame(rows)
    agg.to_csv(out / "reliability_summary.csv", index=False)

    pretty = agg[["regime", "model", "dataset", "runs"]].copy()
    for m in METRICS:
        pct = m in {"accuracy", "f1_macro", "ece", "adaptive_ece", "mean_confidence"}
        pretty[m] = [fmt(a, b, pct=pct) for a, b in zip(agg[f"{m}_mean"], agg[f"{m}_std"])]
    pretty.to_csv(out / "reliability_summary_pretty.csv", index=False)
    (out / "reliability_summary.md").write_text(pretty.to_markdown(index=False), encoding="utf-8")

    th_path = root / "thresholds_long.csv"
    if th_path.exists() and th_path.stat().st_size:
        th = pd.read_csv(th_path)
        t90 = th[np.isclose(th.threshold, 0.90)].copy()
        if len(t90):
            g = t90.groupby(["regime", "model", "dataset"])[["coverage", "accuracy", "risk", "f1_macro"]].agg(["mean", "std"]).reset_index()
            g.columns = ["_".join([str(x) for x in c if str(x)]) if isinstance(c, tuple) else c for c in g.columns]
            g.to_csv(out / "selective_at_090.csv", index=False)

    print(f"Wrote Sprint 5 tables to {out}")


if __name__ == "__main__":
    main()
