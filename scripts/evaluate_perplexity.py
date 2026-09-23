#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from biojev.config import load_yaml
from biojev.corpus import BiomedicalCorpusStream, CausalLMCollator, PackedTokenStream, load_corpus_sources
from biojev.models.causal import load_causal_model, model_input_device
from biojev.utils.environment import snapshot_environment
from biojev.utils.io import write_json
from biojev.utils.seed import set_seed


def main():
    p = argparse.ArgumentParser(description="Held-out biomedical perplexity evaluation.")
    p.add_argument("--model", required=True, help="HF model id, merged checkpoint, or local PEFT adapter")
    p.add_argument("--config", required=True, help="DAPT YAML config defining the corpus")
    p.add_argument("--partition", default="validation", choices=["validation", "test"])
    p.add_argument("--tokens", type=int, default=1_000_000)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--quantization", choices=["none", "4bit"], default="none")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    sequence_length = int(cfg["corpus"].get("sequence_length", 2048))
    model, tokenizer = load_causal_model(
        args.model,
        device_map="auto",
        quantization=None if args.quantization == "none" else args.quantization,
    )
    model.eval()

    docs = BiomedicalCorpusStream(
        load_corpus_sources(cfg["corpus"]),
        partition=args.partition,
        seed=seed,
        shuffle=False,
    )
    packed = PackedTokenStream(
        docs,
        tokenizer,
        token_budget=args.tokens,
        sequence_length=sequence_length,
    )
    loader = DataLoader(packed, batch_size=args.batch_size, collate_fn=CausalLMCollator(tokenizer))

    total_nll = 0.0
    total_predicted_tokens = 0
    device = model_input_device(model)
    with torch.inference_mode():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            # Hugging Face causal-LM loss predicts token t from tokens <t, so the first label
            # of each sequence is not part of the shifted loss.
            predicted = (batch["labels"][:, 1:] != -100).sum().item()
            total_nll += float(outputs.loss.detach().float().cpu()) * predicted
            total_predicted_tokens += predicted

    if total_predicted_tokens == 0:
        raise RuntimeError("No held-out tokens were evaluated")
    mean_nll = total_nll / total_predicted_tokens
    ppl = math.exp(mean_nll) if mean_nll < 50 else float("inf")
    metrics = {
        "model": args.model,
        "partition": args.partition,
        "requested_tokens": args.tokens,
        "predicted_tokens": total_predicted_tokens,
        "nll": mean_nll,
        "perplexity": ppl,
    }
    out = Path(args.output or f"results/perplexity/{Path(args.model).name}_{args.partition}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, metrics)
    write_json(out.with_name(out.stem + "_environment.json"), snapshot_environment())
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
