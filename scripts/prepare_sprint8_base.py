#!/usr/bin/env python3
import argparse
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from biojev.systemone_native.config import load_yaml

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True)
    ap.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    ap.add_argument("--force",action="store_true")
    a=ap.parse_args(); cfg=load_yaml(a.config); m=cfg["model"]
    out=Path(m["prepared_base"])
    if out.exists() and any(out.iterdir()) and not a.force: raise SystemExit(f"{out} exists; use --force")
    out.mkdir(parents=True,exist_ok=True)
    dtype=torch.bfloat16 if a.device=="cuda" else torch.float32
    base=AutoModelForCausalLM.from_pretrained(m["base_model"],torch_dtype=dtype,device_map={"":a.device},low_cpu_mem_usage=True)
    merged=PeftModel.from_pretrained(base,m["dapt_adapter"],is_trainable=False).merge_and_unload(safe_merge=True)
    merged.save_pretrained(out,safe_serialization=True,max_shard_size="5GB")
    try: tok=AutoTokenizer.from_pretrained(m["dapt_adapter"],use_fast=True)
    except Exception: tok=AutoTokenizer.from_pretrained(m["base_model"],use_fast=True)
    tok.save_pretrained(out)
    print(f"[Sprint 8] prepared causal DAPT base: {out}")
if __name__=="__main__": main()
