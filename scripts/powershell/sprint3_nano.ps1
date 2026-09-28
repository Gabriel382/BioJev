$ErrorActionPreference = "Stop"
python scripts\download_datasets.py --all
python scripts\download_datasets.py --sprint3-general
python scripts\plan_decision.py `
  --config configs\decision\nano_full.yaml
python scripts\train_decision.py `
  --config configs\decision\nano_full.yaml
