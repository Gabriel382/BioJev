# %% [markdown]
# # BioJev Sprint 3 — Nano first
# Run these cells in order. The same scripts/config layout is reused for 4B later.

# %%
import subprocess

def run(*args):
    subprocess.run([str(x) for x in args], check=True)

# %%
run("python", "scripts/download_datasets.py", "--all")

# %%
run("python", "scripts/download_datasets.py", "--sprint3-general")

# %%
run("python", "scripts/plan_decision.py", "--config", "configs/decision/nano_full.yaml")

# %% [markdown]
# The next cell expects the existing Sprint-2 DAPT adapter at `outputs/biojev-nano/final`.

# %%
run("python", "scripts/train_decision.py", "--config", "configs/decision/nano_full.yaml")

# %%
run("python", "scripts/evaluate.py", "--model-config", "configs/models/biojev_nano_local.yaml", "--dataset", "bionli", "--split", "test")

# %%
run("python", "scripts/evaluate.py", "--model-config", "configs/models/biojev_nano_local.yaml", "--dataset", "nli4ct", "--split", "dev")

# %%
run("python", "scripts/evaluate.py", "--model-config", "configs/models/biojev_nano_local.yaml", "--dataset", "chemprot", "--split", "test", "--max-examples", "100")

# %%
run("python", "scripts/evaluate.py", "--model-config", "configs/models/biojev_nano_local.yaml", "--dataset", "ddi2013", "--split", "test", "--max-examples", "100")

# %%
run("python", "scripts/evaluate.py", "--model-config", "configs/models/biojev_nano_local.yaml", "--dataset", "biored", "--split", "test", "--max-examples", "100")

# %%
run("python", "scripts/decide.py", "--checkpoint", "outputs/biojev/nano_full/stage-02-biomedical/final", "--context", "Gefitinib reduced EGFR phosphorylation in treated cells.", "--hypothesis", "Gefitinib inhibits EGFR signaling.", "--hypothesis", "Gefitinib activates EGFR signaling.")

# %%
run("python", "scripts/export_biojev_hf.py", "--checkpoint", "outputs/biojev/nano_full/stage-02-biomedical/final", "--output", "exports/BioJev-Nano")
