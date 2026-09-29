# Sprint 4 — Multi-dataset and cross-dataset generalization

## Goal

Test whether a BioJev checkpoint trained as a biomedical NLI decision model transfers to heterogeneous biomedical tasks **without target-task gradient updates**.

Sprint 4 keeps two regimes separate:

1. **Frozen task-general transfer** — Qwen, OpenJev and BioJev are frozen and evaluated on BioNLI, NLI4CT, ChemProt, DDI2013 and BioRED.
2. **Target-supervised baselines** — BioBERT, PubMedBERT/BiomedBERT and BioLinkBERT receive target-dataset training and provide conventional task-specific reference points.

The relation datasets are evaluated as **relation-type classification over gold entity pairs** in Sprint 4. This is not reported as end-to-end relation extraction.

## Main 4B frozen benchmark

The released Sprint-3 4B checkpoint path is already configured in `configs/models/biojev_4b_local.yaml`.

PowerShell:

```powershell
python scripts\run_sprint4_frozen.py `
  --config configs\sprint4\frozen_4b.yaml `
  --skip-existing
```

A quick subset can be run with:

```powershell
python scripts\run_sprint4_frozen.py `
  --config configs\sprint4\frozen_4b.yaml `
  --max-examples 100
```

For one model or one dataset:

```powershell
python scripts\run_sprint4_frozen.py `
  --config configs\sprint4\frozen_4b.yaml `
  --model biojev-4b `
  --dataset chemprot
```

The runner loads each large model only once and evaluates the complete requested dataset list before unloading it.

## Nano benchmark

Once Sprint 3 Nano finishes:

```powershell
python scripts\run_sprint4_frozen.py `
  --config configs\sprint4\frozen_nano.yaml `
  --skip-existing
```

The Nano suite compares size-matched Qwen3.5-0.8B-Base, OpenJev-0.8B and BioJev-Nano.

## Cross-dataset NLI experiment

The full BioJev checkpoint has already seen both BioNLI and NLI4CT during Stage 2, so it cannot establish held-out-source NLI transfer by itself.

Sprint 4 therefore provides source-isolated variants:

```text
General NLI -> BioNLI only
General NLI -> NLI4CT only
```

For Nano:

```powershell
python scripts\train_decision.py `
  --config configs\decision\nano_bionli_only.yaml

python scripts\train_decision.py `
  --config configs\decision\nano_nli4ct_only.yaml

python scripts\run_sprint4_frozen.py `
  --config configs\sprint4\cross_nli_nano.yaml
```

For 4B:

```powershell
python scripts\train_decision.py `
  --config configs\decision\4b_bionli_only.yaml

python scripts\train_decision.py `
  --config configs\decision\4b_nli4ct_only.yaml

python scripts\run_sprint4_frozen.py `
  --config configs\sprint4\cross_nli_4b.yaml
```

This allows direct source-to-target transfer measurements. The source-only configs intentionally preserve the same general-NLI stage as the full BioJev training recipe.

## Conventional supervised baselines

```powershell
python scripts\run_sprint4_supervised.py `
  --config configs\sprint4\supervised_baselines.yaml `
  --skip-existing
```

These baselines are intentionally **not** presented as zero-shot comparisons. They receive target-task training.

## Tables

After one or more Sprint-4 suites have finished:

```powershell
python scripts\build_sprint4_tables.py
```

Outputs are written under:

```text
results/sprint4/tables/
```

including a frozen-transfer table and, when source-isolated checkpoints exist, a cross-NLI matrix.

## Evaluation outputs

Each frozen run stores:

```text
predictions.jsonl
metrics.json
model_config.json
environment.json
protocol.json
```

`metrics.json` contains overall classification/calibration metrics plus per-class precision/recall/F1 and a confusion matrix. Relation predictions also retain raw entailment scores for every candidate relation label.

## Methodological notes

- BioNLI currently comes from a one-split mirror. Sprint 4 uses the same deterministic seed-42 stratified partitioning as Sprint 3 so its test partition remains held out from training.
- NLI4CT `dev` is a validation set for the full model and must be described as such; source-isolated BioNLI-only -> NLI4CT evaluation is the clean cross-dataset transfer setting.
- ChemProt, DDI2013 and BioRED Sprint-4 frozen tests are zero-shot with respect to relation-type supervision, but operate on annotated/gold entity pairs.
- Do not compare the fully supervised encoder numbers and frozen BioJev numbers as if they used identical supervision. The point is to show both conventional task-specific performance and task-general transfer.
