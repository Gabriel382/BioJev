$ErrorActionPreference = "Stop"
& .\.venv\Scripts\Activate.ps1

python scripts/train_dapt.py `
  --config configs/dapt/pubmed_pmc_100m.yaml `
  --resume
