$ErrorActionPreference = "Stop"
& .\.venv\Scripts\Activate.ps1

python scripts/inspect_corpus.py `
  --config configs/dapt/smoke.yaml `
  --documents 5 `
  --show-text

python scripts/train_dapt.py `
  --config configs/dapt/smoke.yaml
