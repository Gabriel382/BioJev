$ErrorActionPreference = "Stop"

python scripts\run_sprint4_frozen.py `
  --config configs\sprint4\frozen_4b.yaml `
  --skip-existing

python scripts\build_sprint4_tables.py
