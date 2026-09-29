# %% [markdown]
# BioJev Sprint 4 walkthrough
# Run cells in order. Nano cells can be skipped until Sprint 3 Nano finishes.

# %%
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path.cwd()
print(ROOT)

# %% [markdown]
# 1. Check that the finished 4B Sprint-3 checkpoint exists.

# %%
ckpt_4b = ROOT / "outputs/biojev/4b_full/stage-02-biomedical/final"
print("4B checkpoint exists:", ckpt_4b.exists(), ckpt_4b)

# %% [markdown]
# 2. Run a 100-example 4B smoke benchmark first.

# %%
subprocess.run([
    sys.executable, "scripts/run_sprint4_frozen.py",
    "--config", "configs/sprint4/frozen_4b.yaml",
    "--model", "biojev-4b",
    "--max-examples", "100",
], check=True)

# %% [markdown]
# 3. Full frozen 4B BioJev benchmark.

# %%
subprocess.run([
    sys.executable, "scripts/run_sprint4_frozen.py",
    "--config", "configs/sprint4/frozen_4b.yaml",
    "--model", "biojev-4b",
    "--skip-existing",
], check=True)

# %% [markdown]
# 4. Run OpenJev and Qwen baselines when compute is available.

# %%
# Uncomment one at a time for controlled runs.
# subprocess.run([sys.executable, "scripts/run_sprint4_frozen.py", "--config", "configs/sprint4/frozen_4b.yaml", "--model", "openjev-4b", "--skip-existing"], check=True)
# subprocess.run([sys.executable, "scripts/run_sprint4_frozen.py", "--config", "configs/sprint4/frozen_4b.yaml", "--model", "qwen35-4b-base", "--skip-existing"], check=True)

# %% [markdown]
# 5. Inspect the main frozen summary.

# %%
summary = ROOT / "results/sprint4/frozen_4b/summary.json"
if summary.exists():
    rows = json.loads(summary.read_text())
    for row in rows:
        print(row.get("model"), row.get("dataset"), row.get("f1_macro"), row.get("accuracy"))

# %% [markdown]
# 6. Once Nano Sprint 3 finishes, run its size-matched benchmark.

# %%
nano_ckpt = ROOT / "outputs/biojev/nano_full/stage-02-biomedical/final"
print("Nano checkpoint exists:", nano_ckpt.exists())
# if nano_ckpt.exists():
#     subprocess.run([sys.executable, "scripts/run_sprint4_frozen.py", "--config", "configs/sprint4/frozen_nano.yaml", "--skip-existing"], check=True)

# %% [markdown]
# 7. Cross-dataset NLI: train source-isolated models before evaluating the matrix.

# %%
# Nano first (cheaper):
# subprocess.run([sys.executable, "scripts/train_decision.py", "--config", "configs/decision/nano_bionli_only.yaml"], check=True)
# subprocess.run([sys.executable, "scripts/train_decision.py", "--config", "configs/decision/nano_nli4ct_only.yaml"], check=True)
# subprocess.run([sys.executable, "scripts/run_sprint4_frozen.py", "--config", "configs/sprint4/cross_nli_nano.yaml"], check=True)

# %% [markdown]
# 8. Conventional target-supervised baselines.

# %%
# subprocess.run([sys.executable, "scripts/run_sprint4_supervised.py", "--config", "configs/sprint4/supervised_baselines.yaml", "--skip-existing"], check=True)

# %% [markdown]
# 9. Build CSV tables from every completed Sprint-4 suite.

# %%
subprocess.run([sys.executable, "scripts/build_sprint4_tables.py"], check=False)
