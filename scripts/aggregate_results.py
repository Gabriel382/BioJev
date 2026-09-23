#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", nargs="?", default="results")
    p.add_argument("--output", default="results/aggregate.csv")
    args = p.parse_args()
    rows = []
    for path in Path(args.root).rglob("metrics.json"):
        try:
            metrics = json.loads(path.read_text())
        except Exception:
            continue
        run = path.parent.name
        parts = run.split("__")
        row = {"run": run, "path": str(path.parent), **metrics}
        if len(parts) >= 4:
            row.update(model=parts[0], dataset=parts[1], split=parts[2], seed=parts[3])
        rows.append(row)
    df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(df.to_string(index=False) if not df.empty else "No metrics.json files found.")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
