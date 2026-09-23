$ErrorActionPreference = "Stop"
.\.venv\Scripts\Activate.ps1
python scripts/train_baseline_matrix.py --config configs/suites/encoder_baselines.yaml
