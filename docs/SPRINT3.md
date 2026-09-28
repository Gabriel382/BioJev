# Sprint 3 — BioJev decision model

Sprint 3 converts the Sprint-2 biomedical backbone work into a biomedical **decision model**.
The default development artifact is **BioJev-Nano**; the same training/evaluation code is then
re-run for larger configurations.

## Scientific variants

1. `nano_openjev_bionli.yaml`: OpenJev-Nano -> BioNLI.
2. `nano_openjev_biomedical.yaml`: OpenJev-Nano -> BioNLI + NLI4CT.
3. `nano_full.yaml` (**main BioJev-Nano**): Qwen3.5-0.8B-Base + the existing Sprint-2 DAPT adapter -> general NLI -> BioNLI + NLI4CT.

The third variant is the main Sprint-3 Nano model. It keeps the Sprint-2 DAPT adapter on the exact
Base checkpoint it was trained against, then learns the Jev-style 3-way decision primitive explicitly.
The OpenJev variants are baselines/ablations rather than the parent of the main BioJev model.

OpenJev currently provides v5 0.8B, 2B and 4B decision checkpoints. For another Qwen3.5 scale,
`from_qwen_general_then_bio_template.yaml` initializes a 3-way Qwen sequence classifier and runs
an explicit general-NLI stage before biomedical NLI, so the code is not tied to OpenJev having a
checkpoint at every size.

## Prepare data

```powershell
python scripts\download_datasets.py --all
```

General NLI is required for the main BioJev route:

```powershell
python scripts\download_datasets.py --sprint3-general
```

## Inspect Nano without loading a model

```powershell
python scripts\plan_decision.py `
  --config configs\decision\nano_full.yaml
```

## Train Nano

The config expects the already-trained Sprint-2 adapter at `outputs\biojev-nano\final`.

```powershell
python scripts\train_decision.py `
  --config configs\decision\nano_full.yaml
```

Principal checkpoint:

`outputs\biojev\nano_full\stage-02-biomedical\final`

## Typed decision smoke test

```powershell
python scripts\decide.py `
  --checkpoint outputs\biojev\nano_full\stage-02-biomedical\final `
  --context "Gefitinib reduced EGFR phosphorylation in treated cells." `
  --hypothesis "Gefitinib inhibits EGFR signaling." `
  --hypothesis "Gefitinib activates EGFR signaling."
```

## Benchmark Nano

```powershell
python scripts\evaluate.py `
  --model-config configs\models\biojev_nano_local.yaml `
  --dataset bionli `
  --split test
```

The same local model config works with NLI4CT, ChemProt, DDI2013 and BioRED. Relation benchmarks
reuse the Sprint-1 natural-language hypothesis transformation instead of creating a task-specific head.

## 4B rerun

Once the Sprint-2 4B DAPT adapter exists:

```powershell
python scripts\plan_decision.py `
  --config configs\decision\4b_full.yaml
python scripts\train_decision.py `
  --config configs\decision\4b_full.yaml
```

## Export to Hugging Face format

```powershell
python scripts\export_biojev_hf.py `
  --checkpoint outputs\biojev\nano_full\stage-02-biomedical\final `
  --output exports\BioJev-Nano
```

Optionally append `--push-to-hub --repo-id YOUR_USER/BioJev-Nano`.

## DAPT adapter transplantation

Sprint-2 DAPT adapters were trained as causal-LM PEFT adapters on Qwen3.5 Base checkpoints. Sprint 3
creates an equivalent sequence-classification LoRA configuration on that same Base checkpoint,
transplants the learned LoRA matrices, and adds a trainable/saved NLI `score` head. General NLI then
teaches the Jev-style 3-way decision primitive before biomedical NLI specialization. OpenJev is kept
as a comparison model; its released v5 checkpoints come from the post-trained Qwen3.5 line, so the
Base-trained DAPT adapter is not silently overlaid onto a different parent checkpoint.
