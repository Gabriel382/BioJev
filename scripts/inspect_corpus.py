#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter

from biojev.config import load_yaml
from biojev.corpus import BiomedicalCorpusStream, load_corpus_sources


def main():
    p = argparse.ArgumentParser(description="Inspect a streaming PubMed/PMC corpus without materializing it.")
    p.add_argument("--config", required=True)
    p.add_argument("--partition", default="train", choices=["train", "validation", "test", "all"])
    p.add_argument("--documents", type=int, default=10)
    p.add_argument("--show-text", action="store_true")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    sources = load_corpus_sources(cfg["corpus"])
    stream = BiomedicalCorpusStream(sources, partition=args.partition, seed=int(cfg.get("seed", 42)))
    counts = Counter()
    chars = 0
    for i, row in enumerate(stream):
        counts[row["source"]] += 1
        chars += len(row["text"])
        print(f"[{i+1:03d}] source={row['source']} id={row['doc_id']} chars={len(row['text']):,}")
        if args.show_text:
            print(row["text"][:700].replace("\n", " "))
            print()
        if i + 1 >= args.documents:
            break
    print("\nSample summary")
    for name, count in sorted(counts.items()):
        print(f"  {name:<12} {count:>6,} docs")
    print(f"  {'TOTAL':<12} {sum(counts.values()):>6,} docs, {chars:,} characters")


if __name__ == "__main__":
    main()
