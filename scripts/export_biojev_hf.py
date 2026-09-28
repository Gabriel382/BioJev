#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from biojev.models.decision import BioJevDecisionModel


def main():
    p = argparse.ArgumentParser(description="Merge/export a Sprint-3 BioJev decision checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--repo-id", default=None)
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    out = Path(args.output)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    wrapped = BioJevDecisionModel(args.checkpoint, device="cpu", torch_dtype="float32")
    model = wrapped.model
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
    model.config.nli_template = wrapped.template
    model.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")
    wrapped.tokenizer.save_pretrained(out)
    manifest = wrapped.manifest
    (out / "decision_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    card = f"""---
license: other
pipeline_tag: text-classification
tags:
- biojev
- biomedical
- nli
- cross-encoder
- qwen3.5
---

# BioJev

BioJev is a biomedical decision model derived from the Qwen3.5/OpenJev research line.
It scores premise/hypothesis pairs as contradiction, entailment, or neutral and supports
closed-set typed decisions by comparing entailment probabilities.

This export was produced from `{args.checkpoint}`. See the BioJev GitHub repository and
`decision_manifest.json` for the exact Sprint-2 DAPT and Sprint-3 decision-training lineage.
Model-weight redistribution should respect the licenses/terms of the upstream checkpoint and training datasets.
"""
    (out / "README.md").write_text(card, encoding="utf-8")

    if args.push_to_hub:
        if not args.repo_id:
            raise ValueError("--repo-id is required with --push-to-hub")
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(repo_id=args.repo_id, repo_type="model", folder_path=str(out))
        print(f"Uploaded to https://huggingface.co/{args.repo_id}")
    print(json.dumps({"output": str(out), "merged": True}, indent=2))


if __name__ == "__main__":
    main()
