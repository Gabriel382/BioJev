#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from biojev.config import load_yaml
from biojev.training.decision import train_decision


def main():
    p = argparse.ArgumentParser(description="Sprint 3: train BioJev biomedical decision model")
    p.add_argument("--config", required=True)
    args = p.parse_args()
    result = train_decision(load_yaml(args.config))
    print(json.dumps(result.__dict__, indent=2, default=str))


if __name__ == "__main__":
    main()
