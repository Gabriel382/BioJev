#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from transformers import AutoTokenizer
from biojev.systemone_native.config import load_yaml
from biojev.systemone_native.format import validate_answer_tokens
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); a=ap.parse_args()
    cfg=load_yaml(a.config); m=cfg["model"]
    sources=[Path(m["prepared_base"]),m["dapt_adapter"],m["base_model"]]
    last=None
    for s in sources:
        try:
            tok=AutoTokenizer.from_pretrained(s,use_fast=True)
            ids=validate_answer_tokens(tok)
            print(json.dumps(ids,indent=2)); print("A-Z single-token candidates: PASS"); return
        except Exception as e: last=e
    raise SystemExit(last)
if __name__=="__main__": main()
