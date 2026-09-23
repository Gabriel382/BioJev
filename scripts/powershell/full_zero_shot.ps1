$ErrorActionPreference = "Stop"
.\.venv\Scripts\Activate.ps1
python scripts/run_suite.py --config configs/suites/sprint1.yaml
python scripts/aggregate_results.py results --output results/aggregate.csv
