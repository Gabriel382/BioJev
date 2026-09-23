#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from biojev.config import load_yaml
from biojev.training.dapt import train_dapt
from biojev.utils.environment import snapshot_environment
from biojev.utils.io import write_json
from biojev.utils.seed import set_seed


def main():
    p = argparse.ArgumentParser(description="Continual biomedical pretraining for BioQwen.")
    p.add_argument("--config", required=True, help="YAML DAPT config")
    p.add_argument("--method", choices=["qlora", "lora", "full"], default=None)
    p.add_argument("--token-budget", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=None,
        help="Resume from latest checkpoint, or pass an explicit checkpoint path",
    )
    args = p.parse_args()

    config = load_yaml(args.config)
    if args.method:
        config.setdefault("training", {})["method"] = args.method
    if args.token_budget:
        config.setdefault("corpus", {})["token_budget"] = args.token_budget
    if args.output_dir:
        config["output_dir"] = args.output_dir
    seed = int(config.get("seed", 42))
    set_seed(seed)
    output_dir = Path(config.get("output_dir", "outputs/bioqwen/run"))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "environment.json", snapshot_environment())
    write_json(output_dir / "resolved_config.json", config)

    result = train_dapt(config, resume_from_checkpoint=args.resume)
    print("\nBioQwen DAPT complete")
    print(f"  checkpoint: {result.final_checkpoint}")
    print(f"  token budget: {result.token_budget:,}")
    print(f"  sequence length: {result.sequence_length:,}")
    print(f"  optimizer steps: {result.max_steps:,}")


if __name__ == "__main__":
    main()
