$ErrorActionPreference = "Stop"
.\.venv\Scripts\Activate.ps1
python scripts/run_suite.py --config configs/suites/smoke.yaml
