$ErrorActionPreference = "Stop"
& .\.venv\Scripts\Activate.ps1

python scripts/run_sprint2_suite.py `
  --config configs/suites/sprint2_smoke.yaml
