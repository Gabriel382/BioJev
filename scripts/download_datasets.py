#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from biojev.datasets.mednli import MedNLIAdapter
from biojev.datasets.registry import get_adapter
from biojev.utils.io import read_jsonl

DEFAULT = ["bionli", "nli4ct", "chemprot", "ddi2013", "biored"]


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
    parser.add_argument("--force", action="store_true", help="Re-download/rebuild even if processed files exist")
    parser.add_argument("--include-no-relation", action="store_true")
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--mednli-dir", type=str, default=None)
    args = parser.parse_args()

    names = DEFAULT if args.all else args.datasets
    if not names and not args.mednli_dir:
        parser.error("Specify dataset names, --all, or --mednli-dir")

    for name in names:
        # Default preparation is safely resumable. Relation-negative generation changes
        # the benchmark definition, so it is always rebuilt unless explicitly cached later.
        if not args.force and not args.include_no_relation:
            counts = _existing_counts(name)
            if counts:
                print(f"[cached] {name}")
                print("  " + ", ".join(f"{split}={count}" for split, count in counts.items()))
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
        print("  " + ", ".join(f"{split}={len(rows)}" for split, rows in prepared.items()))

    if args.mednli_dir:
        print("[prepare] mednli (manual source)")
        prepared = MedNLIAdapter(args.mednli_dir).prepare(force=args.force)
        print("  " + ", ".join(f"{split}={len(rows)}" for split, rows in prepared.items()))


if __name__ == "__main__":
    main()
