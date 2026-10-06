#!/usr/bin/env python3
import argparse,subprocess
from pathlib import Path
from biojev.systemone_native.config import load_yaml
from biojev.systemone_native.format import SYSTEM_PROMPT

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--model-name"); a=ap.parse_args()
    cfg=load_yaml(a.config); e=cfg["export"]; gd=Path(e["gguf_dir"]); marker=gd/"SELECTED_GGUF.txt"
    if not marker.exists(): raise SystemExit("export GGUF first")
    gguf=Path(marker.read_text().strip())
    if not gguf.exists(): raise SystemExit(f"missing GGUF: {gguf}")
    name=a.model_name or e.get("ollama_model",cfg["name"]); mf=gd/"Modelfile"
    mf.write_text(
        f'FROM {gguf}\nCAPABILITY decision\nSYSTEM """{SYSTEM_PROMPT}"""\nPARAMETER num_ctx {int(cfg["training"].get("max_length",2048))}\n',
        encoding="utf-8")
    print(mf.read_text())
    subprocess.run(["ollama","create",name,"-f",str(mf)],check=True)
    print(f"[Sprint 8] Ollama model created: {name}")
if __name__=="__main__": main()
