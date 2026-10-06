
from __future__ import annotations
import csv, json, random
from pathlib import Path
from typing import Any, Iterable
from .format import normalize_record

# BioJev's fixed 3-way decision label convention.
LABEL_MAPS = {
    0:"contradiction",1:"entailment",2:"neutral",
    "0":"contradiction","1":"entailment","2":"neutral",
    "contradiction":"contradiction","entailment":"entailment","neutral":"neutral",
}

def read_jsonl(path):
    rows=[]
    with Path(path).open("r",encoding="utf-8") as f:
        for n,line in enumerate(f,1):
            line=line.strip()
            if line:
                try: rows.append(json.loads(line))
                except json.JSONDecodeError as e: raise ValueError(f"{path}:{n}: {e}") from e
    return rows

def write_jsonl(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+"\n")

def tev_to_canonical(row, source="tev1"):
    row=dict(row); row.setdefault("source",source)
    return normalize_record(row)

def _extract(row,names):
    for n in names:
        if n in row: return row[n]
    return None

def nli_to_canonical(row, source, id_value=""):
    premise=_extract(row,("premise","sentence1","text_a","context"))
    hypothesis=_extract(row,("hypothesis","sentence2","text_b","claim"))
    label=_extract(row,("label","gold_label","class","target"))
    if premise is None or hypothesis is None or label is None:
        raise ValueError(f"{source}: missing premise/hypothesis/label in {sorted(row)}")
    norm=LABEL_MAPS.get(label,LABEL_MAPS.get(str(label).lower()))
    if norm is None: raise ValueError(f"{source}: unknown label {label!r}")
    return normalize_record({
        "id":id_value or str(row.get("id","")),
        "source":source,
        "kind":"choice",
        "state":str(premise),
        "question":"Given the state, what is the relationship of this hypothesis to it? Hypothesis: "+str(hypothesis),
        "options":[
            {"key":"entailment","description":"The hypothesis is supported by the state."},
            {"key":"neutral","description":"The state is insufficient to determine the hypothesis."},
            {"key":"contradiction","description":"The hypothesis conflicts with the state."},
        ],
        "answer_key":norm,
        "provenance":{"original_label":label},
    })

def permute_options(row, rng):
    r=dict(row); opts=[dict(x) for x in r["options"]]; rng.shuffle(opts)
    r["options"]=[{"key":x["key"],"description":x["description"]} for x in opts]
    r.pop("answer",None)
    return normalize_record(r)

def read_table(path):
    path=Path(path); s=path.suffix.lower()
    if s==".jsonl": return read_jsonl(path)
    if s==".json":
        obj=json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj,list): return obj
        for k in ("data","records","examples"):
            if isinstance(obj,dict) and isinstance(obj.get(k),list): return obj[k]
        raise ValueError(f"unsupported JSON container: {path}")
    if s==".csv":
        with path.open("r",encoding="utf-8",newline="") as f: return list(csv.DictReader(f))
    if s==".parquet":
        import pandas as pd
        return pd.read_parquet(path).to_dict(orient="records")
    raise ValueError(f"unsupported data file: {path}")

def find_split_file(root,dataset,split):
    root=Path(root)
    patterns=[
        f"{dataset}/{split}.jsonl",f"{dataset}/{split}.json",f"{dataset}/{split}.csv",f"{dataset}/{split}.parquet",
        f"{dataset}_{split}.jsonl",f"{dataset}_{split}.json",f"{dataset}_{split}.csv",f"{dataset}_{split}.parquet",
    ]
    for rel in patterns:
        p=root/rel
        if p.exists(): return p
    candidates=[]
    for ext in ("jsonl","json","csv","parquet"):
        candidates += list(root.rglob(f"*{dataset}*{split}*.{ext}"))
        d=root/dataset
        if d.exists(): candidates += list(d.rglob(f"*{split}*.{ext}"))
    if not candidates: raise FileNotFoundError(f"could not locate {dataset}/{split} under {root}")
    return sorted(set(candidates))[0]

def synthetic_smoke_records():
    raw=[
      {"id":"smoke-1","source":"sprint8-smoke","state":"The patient has fever, productive cough, and a new lobar infiltrate.","question":"Which diagnosis is best supported?","options":[{"key":"pneumonia","description":"Community-acquired pneumonia"},{"key":"asthma","description":"Acute asthma exacerbation"},{"key":"migraine","description":"Migraine"}],"answer_key":"pneumonia"},
      {"id":"smoke-2","source":"sprint8-smoke","state":"The trial reports no statistically significant difference between treatment and placebo.","question":"Which conclusion is supported?","options":[{"key":"benefit","description":"Treatment clearly improves the primary outcome"},{"key":"no_evidence","description":"The trial does not establish a significant treatment benefit"}],"answer_key":"no_evidence"},
      {"id":"smoke-3","source":"sprint8-smoke","state":"A patient developed urticaria minutes after receiving penicillin.","question":"Which interpretation is best supported?","options":[{"key":"allergic","description":"An immediate hypersensitivity reaction is plausible"},{"key":"unrelated","description":"The symptoms are definitely unrelated to the drug"}],"answer_key":"allergic"},
    ]
    return [normalize_record(x) for x in raw]
