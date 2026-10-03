#!/usr/bin/env python
from __future__ import annotations
import argparse, csv, fnmatch, hashlib, json, platform, shutil, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
from biojev.config import load_yaml

HASH_LIMIT = 256 * 1024 * 1024

def now():
    return datetime.now(timezone.utc).isoformat()

def capture(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return {"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except Exception as e:
        return {"cmd": cmd, "error": repr(e)}

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8*1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def file_rec(path, repo, hash_large=False):
    size = path.stat().st_size
    r = {"path": path.relative_to(repo).as_posix(), "bytes": size}
    if hash_large or size <= HASH_LIMIT:
        r["sha256"] = sha256(path)
    else:
        r["sha256"] = None
        r["hash_skipped_reason"] = "large_file"
    return r

def dir_rec(path, repo, hash_large=False):
    members = [file_rec(p, repo, hash_large) for p in sorted(path.rglob("*")) if p.is_file()]
    return {"path": path.relative_to(repo).as_posix(), "files": len(members),
            "bytes": sum(x["bytes"] for x in members), "members": members}

def inspect(spec, repo, hash_large=False):
    p = repo/spec["path"]
    r = {**spec, "exists": p.exists()}
    if p.exists():
        r["record"] = file_rec(p, repo, hash_large) if p.is_file() else dir_rec(p, repo, hash_large)
    return r

def excluded(rel, pats):
    return any(fnmatch.fnmatch(rel, p) for p in pats)

def copy_release_files(repo, out, pats, excludes):
    seen, copied = set(), []
    for pat in pats:
        for src in repo.glob(pat):
            if not src.is_file():
                continue
            rel = src.relative_to(repo).as_posix()
            if rel in seen or excluded(rel, excludes):
                continue
            dst = out/rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            seen.add(rel); copied.append(rel)
    return sorted(copied)

def copy_adapter(src, out, name):
    allowed = {"adapter_config.json","adapter_model.safetensors","tokenizer.json",
               "tokenizer_config.json","chat_template.jinja","training_manifest.json",
               "decision_manifest.json","resolved_config.json","README.md"}
    copied = []
    if not src.exists(): return copied
    dst = out/"model_artifacts"/name
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.is_file() and p.name in allowed:
            shutil.copy2(p, dst/p.name)
            copied.append((Path("model_artifacts")/name/p.name).as_posix())
    return copied

def env_snapshot(repo):
    e = {"created_at_utc": now(), "python": sys.version, "platform": platform.platform(),
         "executable": sys.executable,
         "git_head": capture(["git","rev-parse","HEAD"]),
         "git_branch": capture(["git","branch","--show-current"]),
         "git_status": capture(["git","status","--porcelain=v1"]),
         "git_remote": capture(["git","remote","-v"]),
         "pip_freeze": capture([sys.executable,"-m","pip","freeze"]),
         "nvidia_smi": capture(["nvidia-smi"])}
    try:
        import torch
        e["torch"] = {"version": torch.__version__, "cuda_available": torch.cuda.is_available(),
                      "cuda_version": torch.version.cuda,
                      "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    except Exception as ex:
        e["torch_error"] = repr(ex)
    return e

def model_card(manifest):
    lines = [
        "# BioJev Model Card","",
        "Generated automatically by Sprint 7 from repository artifacts.","",
        "## Scope","",
        "BioJev is a biomedical decision-model research pipeline combining biomedical domain-adaptive pretraining, general NLI training and biomedical NLI training.","",
        "The release records the main 4B lineage and the 0.8B Nano ablation lineage.","",
        "## Intended use","",
        "Research on biomedical language understanding, transfer, calibration and ablation analysis. It is not validated as a clinical decision system.","",
        "## Evaluation interpretation","",
        "- BioNLI and NLI4CT can be seen during biomedical decision training, depending on the variant.",
        "- ChemProt, DDI2013 and BioRED are the cleanest common unseen target-task transfer datasets in Sprint 6.",
        "- Supervised biomedical BERT baselines are target-task-trained and are not the same regime as frozen task-general transfer.",
        "- Confidence is dataset dependent and must not be interpreted as clinical certainty.","",
        "## Required artifact status",""
    ]
    for a in manifest["required_artifacts"]:
        lines.append(f"- `{a['id']}`: {'available' if a['exists'] else 'MISSING'} — `{a['path']}`")
    lines += ["","## Reproducibility","","See `release_manifest.json`, `environment.json`, `REPRODUCE.md` and `checksums.sha256`.","",
              "## Limitations","","This release does not establish clinical efficacy or safety. Dataset construction, class imbalance and distribution shift can materially affect both performance and calibration.",""]
    return "\n".join(lines)

def reproduce():
    return """# BioJev reproducibility

Run from the repository root.

## Sprint 5
```bash
python scripts/run_sprint5_reliability.py --config configs/sprint5/reliability.yaml --regime all --skip-existing
python scripts/build_sprint5_figures.py
python scripts/build_sprint5_tables.py
```

## Sprint 6
```bash
python scripts/run_sprint6_train.py --skip-existing
python scripts/run_sprint6_eval.py --config configs/sprint6/eval_nano.yaml --skip-existing
python scripts/build_sprint6_tables.py
```

## Sprint 7
```bash
python scripts/run_sprint7_release.py --config configs/sprint7/release.yaml --audit-only
python scripts/run_sprint7_release.py --config configs/sprint7/release.yaml --clean
```

For archival packaging of portable adapters:
```bash
python scripts/run_sprint7_release.py --config configs/sprint7/release.yaml --include-adapters --hash-weights --clean
```
"""

def write_checksums(out):
    target = out/"checksums.sha256"
    lines = []
    for p in sorted(x for x in out.rglob("*") if x.is_file() and x != target):
        lines.append(f"{sha256(p)}  {p.relative_to(out).as_posix()}")
    target.write_text("\n".join(lines)+"\n", encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sprint7/release.yaml")
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--include-adapters", action="store_true")
    ap.add_argument("--hash-weights", action="store_true")
    ap.add_argument("--clean", action="store_true")
    a = ap.parse_args()

    repo = Path.cwd().resolve()
    cfg = load_yaml(a.config)
    req = [inspect(x, repo, a.hash_weights) for x in cfg["required_artifacts"]]
    models = [inspect(x, repo, a.hash_weights) for x in cfg.get("model_artifacts",[])]
    missing = [x for x in req if not x["exists"]]

    print(f"Sprint 7 audit: required={len(req)-len(missing)}/{len(req)} | missing={len(missing)}")
    for x in req:
        print(f"  [{'OK' if x['exists'] else 'MISSING':7}] {x['id']}: {x['path']}")
    print("Model artifacts:")
    for x in models:
        print(f"  [{'OK' if x['exists'] else 'MISSING':7}] {x['id']}: {x['path']}")

    audit = {"created_at_utc": now(), "release_config": cfg,
             "required_artifacts": req, "model_artifacts": models}
    audit_path = repo/"results/sprint7/release_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    if a.audit_only:
        if missing: raise SystemExit(2)
        return
    if missing and not a.allow_missing:
        raise SystemExit("Required paper artifacts are missing. Finish prior sprints or use --allow-missing.")

    out = repo/cfg.get("output_dir","releases/biojev-paper")
    if a.clean and out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    copied = copy_release_files(repo, out, cfg.get("copy_globs",[]), cfg.get("exclude_globs",[]))
    model_copies = []
    if a.include_adapters:
        for spec in cfg.get("model_artifacts",[]):
            model_copies += copy_adapter(repo/spec["path"], out, spec["id"])
        abl = cfg.get("ablation_models",{})
        base = repo/abl.get("root","")
        for variant in abl.get("variants",[]):
            finals = sorted((base/variant).glob("stage-*/final"))
            if finals:
                model_copies += copy_adapter(finals[-1], out, "sprint6_"+variant)

    env = env_snapshot(repo)
    (out/"environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    (out/"pip_freeze.txt").write_text(env.get("pip_freeze",{}).get("stdout","")+"\n", encoding="utf-8")

    manifest = {**audit,
                "git": {"head": env["git_head"], "branch": env["git_branch"], "status": env["git_status"]},
                "copied_files": copied, "copied_model_files": model_copies,
                "include_adapters": a.include_adapters, "hash_weights": a.hash_weights}
    (out/"release_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out/"MODEL_CARD_BIOJEV.md").write_text(model_card(manifest), encoding="utf-8")
    (out/"REPRODUCE.md").write_text(reproduce(), encoding="utf-8")
    write_checksums(out)

    archive_base = out.parent/f"{cfg.get('release_name','biojev-paper-release')}-{cfg.get('release_version','0.7.0')}"
    archive = shutil.make_archive(str(archive_base), "zip", root_dir=out)
    print("Sprint 7 release complete")
    print("  bundle:", out)
    print("  archive:", archive)

if __name__ == "__main__":
    main()
