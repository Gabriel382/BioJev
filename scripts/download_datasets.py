#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from biojev.datasets.mednli import MedNLIAdapter
from biojev.datasets.registry import get_adapter
from biojev.utils.io import read_jsonl

DEFAULT = ["bionli", "nli4ct", "chemprot", "ddi2013", "biored"]
SPRINT3_GENERAL = ["snli", "mnli", "anli_r1"]


def _existing_counts(name: str, processed_root: Path = Path("data/processed")) -> dict[str, int]:
    """Return existing processed split counts, if any.

    This makes `--all` resumable after a partial failure. Use --force to rebuild.
    """
    dataset_dir = processed_root / name
    if not dataset_dir.exists():
        return {}
    counts: dict[str, int] = {}
    for path in sorted(dataset_dir.glob("*.jsonl")):
        try:
            counts[path.stem] = sum(1 for _ in read_jsonl(path))
        except Exception:
            return {}
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", help="Dataset names")
    parser.add_argument("--all", action="store_true", help="Prepare all public Sprint-1 datasets")
    parser.add_argument("--sprint3-general", action="store_true", help="Prepare SNLI, MNLI and ANLI-R1 for generic decision bootstrapping")
    parser.add_argument("--force", action="store_true", help="Re-download/rebuild even if processed files exist")
    parser.add_argument("--include-no-relation", action="store_true")
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--mednli-dir", type=str, default=None)
    args = parser.parse_args()

    names = DEFAULT if args.all else (SPRINT3_GENERAL if args.sprint3_general else args.datasets)
    if not names and not args.mednli_dir:
        parser.error("Specify dataset names, --all, --sprint3-general, or --mednli-dir")

    summary: dict[str, dict[str, int]] = {}

    for name in names:
        # Default preparation is safely resumable. Relation-negative generation changes
        # the benchmark definition, so it is always rebuilt unless explicitly cached later.
        if not args.force and not args.include_no_relation:
            counts = _existing_counts(name)
            if counts:
                print(f"[cached] {name}")
                print("  " + ", ".join(f"{split}={count}" for split, count in counts.items()))
                summary[name] = counts
                continue

        kwargs = {}
        if name in {"chemprot", "ddi2013", "biored"}:
            kwargs.update(include_no_relation=args.include_no_relation, negative_ratio=args.negative_ratio)
        print(f"[prepare] {name}")
        try:
            prepared = get_adapter(name, **kwargs).prepare(force=args.force)
        except ModuleNotFoundError as exc:
            if name == "biored" and exc.name == "bioc":
                raise RuntimeError(
                    "BioRED requires the 'bioc' package used by the BigBio loader. "
                    "Reinstall BioJev with: python -m pip install -e \".[dev,plots,notebook]\""
                ) from exc
            raise
        counts = {split: len(rows) for split, rows in prepared.items()}
        print("  " + ", ".join(f"{split}={count}" for split, count in counts.items()))
        summary[name] = counts

    if args.mednli_dir:
        print("[prepare] mednli (manual source)")
        prepared = MedNLIAdapter(args.mednli_dir).prepare(force=args.force)
        counts = {split: len(rows) for split, rows in prepared.items()}
        print("  " + ", ".join(f"{split}={count}" for split, count in counts.items()))
        summary["mednli"] = counts

    if summary:
        print("\nDataset summary")
        print(f"{'dataset':<12} {'splits':<46} {'total':>10}")
        print("-" * 72)
        grand_total = 0
        for dataset_name, counts in summary.items():
            total = sum(counts.values())
            grand_total += total
            split_text = ", ".join(f"{k}={v:,}" for k, v in counts.items())
            print(f"{dataset_name:<12} {split_text:<46} {total:>10,}")
        print("-" * 72)
        print(f"{'TOTAL':<59} {grand_total:>10,}")


if __name__ == "__main__":
    main()
