$ErrorActionPreference = "Stop"
& .\.venv\Scripts\Activate.ps1

# Intrinsic biomedical LM comparison on the same deterministic held-out documents.
python scripts/evaluate_perplexity.py `
  --model Qwen/Qwen3.5-4B-Base `
  --config configs/dapt/pubmed_pmc_100m.yaml `
  --partition validation `
  --tokens 1000000 `
  --output results/perplexity/qwen35_4b_base.json

python scripts/evaluate_perplexity.py `
  --model outputs/bioqwen/pubmed_pmc_100m/final `
  --config configs/dapt/pubmed_pmc_100m.yaml `
  --partition validation `
  --tokens 1000000 `
  --output results/perplexity/bioqwen_100m.json

# Sprint-1 multi-dataset benchmark, now with fair base-vs-domain-adapted likelihood scoring.
python scripts/run_sprint2_suite.py `
  --config configs/suites/sprint2_ablation.yaml

python scripts/compare_sprint2.py `
  --summary results/sprint2/summary.json `
  --output results/sprint2/comparison.csv
