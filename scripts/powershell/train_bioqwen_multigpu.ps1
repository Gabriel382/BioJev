param(
    [int]$NumProcesses = 2,
    [string]$Config = "configs/dapt/pubmed_pmc_100m.yaml"
)

$ErrorActionPreference = "Stop"
& .\.venv\Scripts\Activate.ps1

accelerate launch `
  --multi_gpu `
  --num_processes $NumProcesses `
  scripts/train_dapt.py `
  --config $Config
