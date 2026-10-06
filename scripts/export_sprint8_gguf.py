#!/usr/bin/env python3
import argparse,json,subprocess
from pathlib import Path
from biojev.systemone_native.config import load_yaml

def run(cmd):
    print("+"," ".join(map(str,cmd)),flush=True)
    subprocess.run(list(map(str,cmd)),check=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--llama-cpp",required=True); ap.add_argument("--quantize")
    a=ap.parse_args(); cfg=load_yaml(a.config); e=cfg["export"]; merged=Path(e["merged_dir"])
    if not merged.exists(): raise SystemExit(f"{merged} missing; run merge_sprint8_adapter.py")
    out=Path(e["gguf_dir"]); out.mkdir(parents=True,exist_ok=True); name=cfg["name"]; f16=out/f"{name}-F16.gguf"
    metadata={
      "general.name":name,
      "general.author":"Gabriel Henrique Alencar Medeiros",
      "general.organization":"LITIS / Universite de Rouen Normandie",
      "general.description":"BioJev native biomedical System One decision model",
      "general.finetune":"biojev-systemone",
      "general.tags":["biojev","biomedical","decision","system-one","tev1"],
      "qwen35.decision.type":"tev1",
    }
    meta=out/"metadata-tev1.json"; meta.write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    conv=Path(a.llama_cpp)/"convert_hf_to_gguf.py"
    if not conv.exists(): raise SystemExit(f"converter missing: {conv}")
    run(["python3",conv,merged,"--outfile",f16,"--outtype","f16","--metadata",meta])
    chosen=f16; quant=a.quantize or e.get("quantize")
    if quant:
        cands=[Path(a.llama_cpp)/"build/bin/llama-quantize",Path(a.llama_cpp)/"build/bin/quantize",Path(a.llama_cpp)/"llama-quantize"]
        qb=next((p for p in cands if p.exists()),None)
        if qb is None: raise SystemExit("llama-quantize not found; build llama.cpp or omit quantization")
        qfile=out/f"{name}-{quant}.gguf"; run([qb,f16,qfile,quant]); chosen=qfile
    (out/"SELECTED_GGUF.txt").write_text(str(chosen.resolve())+"\n",encoding="utf-8")
    print(f"[Sprint 8] GGUF: {chosen}")
    print("[Sprint 8] Compatibility is accepted only after the real Ollama /v1/systemone test passes.")
if __name__=="__main__": main()
