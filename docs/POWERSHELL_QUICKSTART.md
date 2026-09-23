# BioJev Sprint 1 — PowerShell quick start

Run from the repository root.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,plots,notebook]"

python scripts/download_datasets.py --all
python scripts/report_disk_usage.py

python scripts/evaluate.py `
  --model-config configs/models/openjev.yaml `
  --dataset bionli `
  --split test `
  --seed 42 `
  --max-examples 50

python scripts/run_suite.py `
  --config configs/suites/smoke.yaml
```

After the smoke tests pass, the heavy commands are:

```powershell
python scripts/train_baseline_matrix.py `
  --config configs/suites/encoder_baselines.yaml

python scripts/run_suite.py `
  --config configs/suites/sprint1.yaml

python scripts/aggregate_results.py results `
  --output results/aggregate.csv
```
