# %% [markdown]
# BioJev Sprint 2 — BioQwen domain adaptation
#
# Execute cells in order in VS Code/Spyder. Expensive training/evaluation cells are explicit.

# %%
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path.cwd()
if not (ROOT / "pyproject.toml").exists() and (ROOT.parent / "pyproject.toml").exists():
    ROOT = ROOT.parent
os.chdir(ROOT)
print("Repository:", ROOT)
print("Python:", sys.executable)

# %% [markdown]
# 1) Verify the local environment and GPU.

# %%
subprocess.run([sys.executable, "-c", "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('gpus=', torch.cuda.device_count())"], check=True)

# %% [markdown]
# 2) Prepare/verify Sprint-1 datasets. The command now prints a complete dataset summary.

# %%
subprocess.run([sys.executable, "scripts/download_datasets.py", "--all"], check=True)

# %% [markdown]
# 3) Inspect a few documents from the streaming PubMed/PMC mixture without materializing the corpus.

# %%
subprocess.run([
    sys.executable,
    "scripts/inspect_corpus.py",
    "--config", "configs/dapt/smoke.yaml",
    "--documents", "5",
    "--show-text",
], check=True)

# %% [markdown]
# 4) Optional real-model smoke test: 50K biomedical tokens using QLoRA.
# This downloads/loads Qwen3.5-4B-Base, so it is much heavier than a unit test.

# %%
RUN_SMOKE_TRAINING = False
if RUN_SMOKE_TRAINING:
    subprocess.run([
        sys.executable,
        "scripts/train_dapt.py",
        "--config", "configs/dapt/smoke.yaml",
    ], check=True)

# %% [markdown]
# 5) Main 100M-token BioQwen run. Change the flag only when ready to train.

# %%
RUN_100M_TRAINING = False
if RUN_100M_TRAINING:
    subprocess.run([
        sys.executable,
        "scripts/train_dapt.py",
        "--config", "configs/dapt/pubmed_pmc_100m.yaml",
    ], check=True)

# %% [markdown]
# 6) Resume the 100M run from the most recent checkpoint if interrupted.

# %%
RESUME_100M = False
if RESUME_100M:
    subprocess.run([
        sys.executable,
        "scripts/train_dapt.py",
        "--config", "configs/dapt/pubmed_pmc_100m.yaml",
        "--resume",
    ], check=True)

# %% [markdown]
# 7) Evaluate held-out biomedical perplexity for the untouched Qwen3.5-4B-Base.

# %%
RUN_PERPLEXITY = False
if RUN_PERPLEXITY:
    subprocess.run([
        sys.executable,
        "scripts/evaluate_perplexity.py",
        "--model", "Qwen/Qwen3.5-4B-Base",
        "--config", "configs/dapt/pubmed_pmc_100m.yaml",
        "--partition", "validation",
        "--tokens", "1000000",
        "--output", "results/perplexity/qwen35_4b_base.json",
    ], check=True)

# %% [markdown]
# 8) Evaluate the same held-out stream for BioQwen.

# %%
if RUN_PERPLEXITY:
    subprocess.run([
        sys.executable,
        "scripts/evaluate_perplexity.py",
        "--model", "outputs/bioqwen/pubmed_pmc_100m/final",
        "--config", "configs/dapt/pubmed_pmc_100m.yaml",
        "--partition", "validation",
        "--tokens", "1000000",
        "--output", "results/perplexity/bioqwen_100m.json",
    ], check=True)

# %% [markdown]
# 9) Run the Sprint-1 downstream benchmark for Qwen-Base vs BioQwen.
# Sprint 2 uses mean conditional likelihood so neither model needs decision supervision.

# %%
RUN_DOWNSTREAM = False
if RUN_DOWNSTREAM:
    subprocess.run([
        sys.executable,
        "scripts/run_sprint2_suite.py",
        "--config", "configs/suites/sprint2_ablation.yaml",
    ], check=True)

# %% [markdown]
# 10) Aggregate the Qwen-vs-BioQwen downstream results.

# %%
if RUN_DOWNSTREAM:
    subprocess.run([
        sys.executable,
        "scripts/compare_sprint2.py",
        "--summary", "results/sprint2/summary.json",
        "--output", "results/sprint2/comparison.csv",
    ], check=True)

# %% [markdown]
# 11) Export the final PEFT checkpoint to a clean Hugging Face directory.

# %%
EXPORT_CHECKPOINT = False
if EXPORT_CHECKPOINT:
    subprocess.run([
        sys.executable,
        "scripts/export_hf.py",
        "--checkpoint", "outputs/bioqwen/pubmed_pmc_100m/final",
        "--output", "exports/BioQwen-4B-100M",
    ], check=True)

# %% [markdown]
# 12) Optional merged export. This requires enough memory to load and merge the base model.

# %%
MERGE_EXPORT = False
if MERGE_EXPORT:
    subprocess.run([
        sys.executable,
        "scripts/export_hf.py",
        "--checkpoint", "outputs/bioqwen/pubmed_pmc_100m/final",
        "--output", "exports/BioQwen-4B-100M-merged",
        "--merge",
    ], check=True)
