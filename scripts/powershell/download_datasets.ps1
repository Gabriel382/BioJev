$ErrorActionPreference = "Stop"
.\.venv\Scripts\Activate.ps1
python scripts/download_datasets.py --all
python scripts/report_disk_usage.py
