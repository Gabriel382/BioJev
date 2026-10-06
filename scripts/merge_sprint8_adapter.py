#!/usr/bin/env python3
import argparse,shutil
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM,AutoTokenizer
from biojev.systemone_native.config import load_yaml
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--device",choices=["cpu","cuda"],default="cpu"); ap.add_argument("--force",action="store_true"); a=ap.parse_args()
    cfg=load_yaml(a.config); base=Path(cfg["model"]["prepared_base"]); adapter=Path(cfg["training"]["output_dir"])/"final"; out=Path(cfg["export"]["merged_dir"])
    if out.exists() and a.force: shutil.rmtree(out)
    if out.exists() and any(out.iterdir()): raise SystemExit(f"{out} exists; use --force")
    out.mkdir(parents=True,exist_ok=True); dtype=torch.bfloat16 if a.device=="cuda" else torch.float32
    model=AutoModelForCausalLM.from_pretrained(base,torch_dtype=dtype,device_map={"":a.device},low_cpu_mem_usage=True)
    model=PeftModel.from_pretrained(model,adapter,is_trainable=False).merge_and_unload(safe_merge=True)
    model.save_pretrained(out,safe_serialization=True,max_shard_size="5GB")
    AutoTokenizer.from_pretrained(adapter,use_fast=True).save_pretrained(out)
    print(f"[Sprint 8] merged causal model: {out}")
if __name__=="__main__": main()
