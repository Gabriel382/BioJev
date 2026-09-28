#!/usr/bin/env python
from __future__ import annotations

import argparse

from biojev.config import load_yaml
from biojev.training.decision_data import build_decision_mixture


def main():
    p = argparse.ArgumentParser(description="Inspect Sprint-3 decision training without loading a model")
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_yaml(args.config)
    print("BioJev Sprint 3 decision plan")
    print(f"  run:            {cfg.get('run_name')}")
    print(f"  initialization: {cfg['model'].get('initialization')}")
    if cfg['model'].get('subfolder'):
        print(f"  OpenJev:        {cfg['model'].get('repo_id')} / {cfg['model'].get('subfolder')}")
    if cfg['model'].get('base_model'):
        print(f"  base model:     {cfg['model'].get('base_model')}")
    print(f"  DAPT adapter:   {cfg['model'].get('dapt_adapter') or 'none'}")
    for i, stage in enumerate(cfg.get("stages", []), start=1):
        mix = build_decision_mixture(
            stage["datasets"],
            seed=int(cfg.get("seed", 42)) + i - 1,
            epoch_examples=stage.get("epoch_examples"),
            dev_max_per_dataset=stage.get("dev_max_per_dataset", 2000),
        )
        print(f"\n  Stage {i}: {stage.get('name', i)}")
        print(f"    train: {len(mix.train):,}")
        print(f"    dev:   {len(mix.dev):,}")
        for source in mix.manifest['sources']:
            print(
                f"    - {source['name']:<12} weight={source['weight']:<5g} "
                f"sampled={source['sampled_train']:,}"
            )


if __name__ == "__main__":
    main()
