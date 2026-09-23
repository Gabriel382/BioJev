#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from biojev.models.causal import is_peft_checkpoint, load_causal_model, load_tokenizer


def _write_model_card(output: Path, source: str, merged: bool):
    text = f"""---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B-Base
pipeline_tag: text-generation
tags:
- biojev
- bioqwen
- biomedical
- continual-pretraining
- qwen3.5
---

# BioQwen

BioQwen is the biomedical domain-adapted Qwen3.5 backbone produced in **BioJev Sprint 2**.
It is trained with continued causal language modeling on a configurable streaming PubMed / PMC corpus.

Source checkpoint: `{source}`  
Export type: `{'merged model' if merged else 'training checkpoint / PEFT adapter'}`

BioQwen is an intermediate research artifact. It has **not** yet received BioJev decision/NLI supervision.
Use the BioJev repository for the exact corpus, configuration, evaluation, and reproducibility details.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="Export a BioQwen checkpoint in Hugging Face format.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--merge", action="store_true", help="Merge a LoRA/QLoRA adapter into the base model")
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--repo-id", default=None)
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    checkpoint = Path(args.checkpoint)
    output = Path(args.output)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    peft = is_peft_checkpoint(str(checkpoint))
    if peft and args.merge:
        model, tokenizer = load_causal_model(str(checkpoint), device_map="auto")
        if not hasattr(model, "merge_and_unload"):
            raise RuntimeError("Loaded PEFT checkpoint cannot be merged")
        model = model.merge_and_unload()
        model.save_pretrained(output, safe_serialization=True, max_shard_size="5GB")
        tokenizer.save_pretrained(output)
        merged = True
    else:
        for item in checkpoint.iterdir():
            dst = output / item.name
            if item.is_dir():
                shutil.copytree(item, dst)
            else:
                shutil.copy2(item, dst)
        # Ensure tokenizer is present even when Trainer saved only an adapter.
        try:
            tokenizer = load_tokenizer(str(checkpoint))
            tokenizer.save_pretrained(output)
        except Exception:
            pass
        merged = not peft

    _write_model_card(output, str(checkpoint), merged)

    if args.push_to_hub:
        if not args.repo_id:
            raise ValueError("--repo-id is required with --push-to-hub")
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(repo_id=args.repo_id, repo_type="model", folder_path=str(output))
        print(f"Uploaded to https://huggingface.co/{args.repo_id}")

    print(json.dumps({"output": str(output), "merged": merged, "peft_source": peft}, indent=2))


if __name__ == "__main__":
    main()
