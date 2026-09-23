# BioJev — PowerShell quick start

Run from the repository root.

## Setup

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,plots,notebook,train]"
```

## Sprint 1 — datasets and benchmark

```powershell
python scripts/download_datasets.py --all
python scripts/report_disk_usage.py

python scripts/evaluate.py `
  --model-config configs/models/openjev.yaml `
  --dataset bionli `
  --split test `
  --seed 42 `
  --max-examples 50
```

## Sprint 2 — inspect biomedical stream

```powershell
python scripts/inspect_corpus.py `
  --config configs/dapt/smoke.yaml `
  --documents 5 `
  --show-text
```

## Sprint 2 — smoke DAPT

```powershell
python scripts/train_dapt.py `
  --config configs/dapt/smoke.yaml
```

## Sprint 2 — 100M BioQwen

```powershell
python scripts/train_dapt.py `
  --config configs/dapt/pubmed_pmc_100m.yaml
```

Resume:

```powershell
python scripts/train_dapt.py `
  --config configs/dapt/pubmed_pmc_100m.yaml `
  --resume
```

## Sprint 2 — evaluate Qwen vs BioQwen

```powershell
.\scripts\powershell\sprint2_benchmark.ps1
```

See `docs/SPRINT2.md` for all training modes, multi-GPU launch, perplexity evaluation and Hugging Face export.
