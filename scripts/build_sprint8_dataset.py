#!/usr/bin/env python3
import argparse, hashlib, json, random
from collections import Counter
from pathlib import Path
from biojev.systemone_native.config import load_yaml
from biojev.systemone_native.data import (
    find_split_file,nli_to_canonical,permute_options,read_jsonl,read_table,
    synthetic_smoke_records,tev_to_canonical,write_jsonl,
)
from biojev.systemone_native.format import normalize_record

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda:f.read(1<<20),b""): h.update(c)
    return h.hexdigest()

def load_tev(path,source): return [tev_to_canonical(x,source) for x in read_jsonl(path)]

def load_nli(path,source):
    out=[]
    for i,x in enumerate(read_table(path)):
        try: out.append(nli_to_canonical(x,source,id_value=f"{source}:{i}"))
        except ValueError:
            label=str(x.get("label",x.get("gold_label",""))).lower()
            if label in {"","-","-1","none","nan"}: continue
            raise
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True)
    ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--overwrite",action="store_true")
    a=ap.parse_args()
    cfg=load_yaml(a.config); seed=int(cfg.get("seed",42)); data=cfg["data"]
    out=Path(data["output_dir"])
    if out.exists() and not a.overwrite: raise SystemExit(f"{out} exists; use --overwrite")
    out.mkdir(parents=True,exist_ok=True)

    if a.smoke:
        rows=synthetic_smoke_records(); train=[]
        for rep in range(24):
            for row in rows:
                train.append(permute_options(row,random.Random(seed+rep*100+len(train))))
        dev=[permute_options(x,random.Random(seed+9000+i)) for i,x in enumerate(rows)]
    else:
        train=[]; dev=[]
        if data.get("tev_train"): train += load_tev(Path(data["tev_train"]),"tev1-general")
        if data.get("tev_dev"): dev += load_tev(Path(data["tev_dev"]),"tev1-general")
        root=data.get("biojev_processed_root")
        for src in data.get("biomedical_nli",[]):
            name=src["name"]
            tp=src.get("train")
            dp=src.get("dev")
            if tp is None:
                if not root: raise ValueError(f"{name}: no train path/root")
                tp=find_split_file(root,name,src.get("train_split","train"))
            if dp is None and root:
                try: dp=find_split_file(root,name,src.get("dev_split","dev"))
                except FileNotFoundError: dp=None
            source_train = load_nli(tp,name)
            holdout = int(src.get("dev_holdout", 0) or 0)
            if dp:
                train += source_train
                dev += load_nli(dp,name)
            elif holdout > 0:
                rr = random.Random(seed + sum(ord(c) for c in name))
                rr.shuffle(source_train)
                dev += source_train[:holdout]
                train += source_train[holdout:]
            else:
                train += source_train
        for p in data.get("extra_canonical_train",[]): train += [normalize_record(x) for x in read_jsonl(p)]
        for p in data.get("extra_canonical_dev",[]): dev += [normalize_record(x) for x in read_jsonl(p)]
        aug=int(data.get("option_permutations",1))
        if aug>1:
            base=train; train=[]
            for rep in range(aug):
                rr=random.Random(seed+1000*rep)
                train += [permute_options(x,rr) for x in base]

    random.Random(seed).shuffle(train); random.Random(seed+1).shuffle(dev)
    if data.get("max_train"): train=train[:int(data["max_train"])]
    if data.get("max_dev"): dev=dev[:int(data["max_dev"])]

    tp=out/"train.jsonl"; dp=out/"dev.jsonl"
    write_jsonl(tp,train); write_jsonl(dp,dev)
    manifest={
      "version":"sprint8-v0.1","seed":seed,
      "train_examples":len(train),"dev_examples":len(dev),
      "train_sources":dict(Counter(x["source"] for x in train)),
      "dev_sources":dict(Counter(x["source"] for x in dev)),
      "files":{"train.jsonl":sha256(tp),"dev.jsonl":sha256(dp)},
      "note":"ChemProt/DDI2013/BioRED excluded by default to preserve transfer evaluation."
    }
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(manifest,indent=2))
if __name__=="__main__": main()
