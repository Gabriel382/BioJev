#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from biojev.models.decision import BioJevDecisionModel


def main():
    p = argparse.ArgumentParser(description="Run a typed BioJev decision")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--context", required=True)
    p.add_argument("--hypothesis", action="append", required=True, dest="hypotheses")
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()
    model = BioJevDecisionModel(args.checkpoint)
    result = model.decide(args.context, args.hypotheses, batch_size=args.batch_size)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
