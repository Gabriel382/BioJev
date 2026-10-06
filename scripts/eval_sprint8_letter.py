#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
from biojev.systemone_native.config import load_yaml
from biojev.systemone_native.data import read_jsonl
from biojev.systemone_native.format import normalize_record,render_messages,validate_answer_tokens

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--limit",type=int); a=ap.parse_args()
    cfg=load_yaml(a.config); base=Path(cfg["model"]["prepared_base"]); adapter=Path(cfg["training"]["output_dir"])/"final"
    tok=AutoTokenizer.from_pretrained(adapter,use_fast=True); ids=validate_answer_tokens(tok)
    q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
    model=AutoModelForCausalLM.from_pretrained(base,quantization_config=q,device_map={"":0},torch_dtype=torch.bfloat16)
    model=PeftModel.from_pretrained(model,adapter,is_trainable=False); model.eval()
    rows=read_jsonl(Path(cfg["data"]["output_dir"])/"dev.jsonl")
    if a.limit: rows=rows[:a.limit]
    correct=0; by={}
    for i,row in enumerate(rows,1):
        r=normalize_record(row); msgs=render_messages(r)[:-1]
        try: enc=tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors="pt",return_dict=True)
        except TypeError: enc=tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,return_tensors="pt",return_dict=True)
        enc={k:v.to(model.device) for k,v in enc.items()}
        with torch.inference_mode(): logits=model(**enc).logits[0,-1].float()
        cand=[o["label"] for o in r["options"]]; scores={c:float(logits[ids[c]].cpu()) for c in cand}; pred=max(scores,key=scores.get)
        ok=pred==r["answer"]; correct+=ok; s=by.setdefault(r["source"],[0,0]); s[0]+=ok; s[1]+=1
        if i<=5: print(r["id"],"gold=",r["answer"],"pred=",pred,"scores=",scores)
    result={"n":len(rows),"accuracy":correct/len(rows) if rows else 0,"by_source":{k:{"n":n,"accuracy":c/n} for k,(c,n) in by.items()}}
    print(json.dumps(result,indent=2))
if __name__=="__main__": main()
