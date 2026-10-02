#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/sprint5")
    return ap.parse_args()


def weighted_reliability(files: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in files if p.exists()]
    if not frames:
        return pd.DataFrame()
    # Fixed-bin boundaries are identical; pool counts and weighted accuracies/confidences.
    x = pd.concat(frames, ignore_index=True)
    rows = []
    for b, g in x.groupby("bin"):
        n = g.n.sum()
        if n <= 0:
            continue
        rows.append({
            "bin": b,
            "n": n,
            "mean_confidence": np.average(g.mean_confidence.fillna(0), weights=g.n),
            "accuracy": np.average(g.accuracy.fillna(0), weights=g.n),
        })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    root = Path(args.results)
    s = pd.read_csv(root / "summary.csv")
    figdir = root / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    for dataset in sorted(s.dataset.unique()):
        sub = s[s.dataset == dataset]
        # Reliability diagram
        fig, ax = plt.subplots(figsize=(6.5, 5.2))
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="perfect calibration")
        for model in sorted(sub.model.unique()):
            files = []
            for _, r in sub[sub.model == model].iterrows():
                files.append(root / "runs" / r.model / r.dataset / r.run_id / "reliability_bins.csv")
            b = weighted_reliability(files)
            if len(b):
                ax.plot(b.mean_confidence, b.accuracy, marker="o", label=model)
        ax.set(xlabel="Mean confidence", ylabel="Empirical accuracy", title=f"Reliability — {dataset}", xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figdir / f"reliability__{dataset}.png", dpi=220)
        plt.close(fig)

        # Risk-coverage
        fig, ax = plt.subplots(figsize=(6.5, 5.2))
        grid = np.linspace(0.01, 1.0, 100)
        for model in sorted(sub.model.unique()):
            curves = []
            for _, r in sub[sub.model == model].iterrows():
                p = root / "runs" / r.model / r.dataset / r.run_id / "risk_coverage.csv"
                if p.exists():
                    c = pd.read_csv(p).sort_values("coverage")
                    curves.append(np.interp(grid, c.coverage, c.risk))
            if curves:
                arr = np.vstack(curves)
                ax.plot(grid, arr.mean(axis=0), label=model)
        ax.set(xlabel="Coverage", ylabel="Selective risk", title=f"Risk–coverage — {dataset}", xlim=(0, 1), ylim=(0, None))
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figdir / f"risk_coverage__{dataset}.png", dpi=220)
        plt.close(fig)

    print(f"Wrote Sprint 5 figures to {figdir}")


if __name__ == "__main__":
    main()
