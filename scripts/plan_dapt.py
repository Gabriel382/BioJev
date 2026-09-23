#!/usr/bin/env python
from __future__ import annotations

import argparse
import math

from biojev.config import load_yaml
from biojev.corpus import estimate_sequence_count, load_corpus_sources


def main():
    p = argparse.ArgumentParser(description="Print a BioQwen DAPT plan without downloading a model or corpus.")
    p.add_argument("--config", required=True)
    p.add_argument("--world-size", type=int, default=1)
    args = p.parse_args()

    cfg = load_yaml(args.config)
    corpus = cfg["corpus"]
    train = cfg["training"]
    budget = int(corpus["token_budget"])
    seq = int(corpus.get("sequence_length", 2048))
    sequences = estimate_sequence_count(budget, seq)
    per_device = int(train.get("per_device_batch_size", 1))
    accum = int(train.get("gradient_accumulation_steps", 1))
    global_sequences = per_device * accum * args.world_size
    steps = math.ceil(sequences / global_sequences)
    sources = load_corpus_sources(corpus)

    print("BioQwen DAPT plan")
    print(f"  base model:       {cfg['model'].get('base_model')}")
    print(f"  method:           {train.get('method', 'qlora')}")
    print(f"  token budget:     {budget:,}")
    print(f"  sequence length:  {seq:,}")
    print(f"  packed sequences: {sequences:,}")
    print(f"  world size:       {args.world_size}")
    print(f"  per-device batch: {per_device}")
    print(f"  grad accumulation:{accum:>8}")
    print(f"  effective batch:  {global_sequences:,} sequences / optimizer step")
    print(f"  optimizer steps:  {steps:,}")
    print("  sources:")
    total_weight = sum(x.weight for x in sources)
    for source in sources:
        print(f"    - {source.name:<10} {source.weight / total_weight:>6.1%}  {source.dataset}")


if __name__ == "__main__":
    main()
