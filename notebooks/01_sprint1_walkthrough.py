# %% [markdown]
# # BioJev Sprint 1 — executable walkthrough
# This file uses `# %%` cells, so VS Code/Spyder can execute it cell by cell.
# Run from the repository root.

# %%
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path.cwd()
print("Python:", sys.version)
print("Repository:", ROOT)

# %% [markdown]
# ## 1. Install/update BioJev and all Sprint-1 dependencies
# BioRED needs `bioc`; this is now installed by the BioJev package.

# %%
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", ".[dev,plots,notebook]"],
    check=True,
)

# %% [markdown]
# ## 2. Download and normalize all public Sprint-1 datasets
# The downloader is resumable: already processed datasets are skipped unless `--force` is used.

# %%
subprocess.run(
    [sys.executable, "scripts/download_datasets.py", "--all"],
    check=True,
)

# %% [markdown]
# ## 3. Inspect disk usage after dataset preparation

# %%
subprocess.run([sys.executable, "scripts/report_disk_usage.py"], check=True)

# %% [markdown]
# ## 4. OpenJev smoke evaluation on BioNLI
# This downloads the OpenJev checkpoint on first use. Increase/remove `--max-examples` after validation.

# %%
subprocess.run(
    [
        sys.executable,
        "scripts/evaluate.py",
        "--model-config", "configs/models/openjev.yaml",
        "--dataset", "bionli",
        "--split", "test",
        "--seed", "42",
        "--max-examples", "50",
    ],
    check=True,
)

# %% [markdown]
# ## 5. Qwen smoke evaluation on BioNLI

# %%
subprocess.run(
    [
        sys.executable,
        "scripts/evaluate.py",
        "--model-config", "configs/models/qwen35_4b.yaml",
        "--dataset", "bionli",
        "--split", "test",
        "--seed", "42",
        "--max-examples", "25",
    ],
    check=True,
)

# %% [markdown]
# ## 6. Run the configured smoke suite

# %%
subprocess.run(
    [sys.executable, "scripts/run_suite.py", "--config", "configs/suites/smoke.yaml"],
    check=True,
)

# %% [markdown]
# ## 7. Train conventional biomedical encoder baselines — HEAVY
# BioBERT, BiomedBERT/PubMedBERT and BioLinkBERT are supervised task baselines.

# %%
# Uncomment when ready for the full baseline training matrix.
# subprocess.run(
#     [sys.executable, "scripts/train_baseline_matrix.py", "--config", "configs/suites/encoder_baselines.yaml"],
#     check=True,
# )

# %% [markdown]
# ## 8. Run the full zero-shot Sprint-1 suite — HEAVY

# %%
# Uncomment when the smoke runs are successful.
# subprocess.run(
#     [sys.executable, "scripts/run_suite.py", "--config", "configs/suites/sprint1.yaml"],
#     check=True,
# )

# %% [markdown]
# ## 9. Aggregate result folders into a CSV

# %%
subprocess.run(
    [sys.executable, "scripts/aggregate_results.py", "results", "--output", "results/aggregate.csv"],
    check=True,
)
