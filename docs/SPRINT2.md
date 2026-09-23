# Sprint 2 — Biomedical Domain Model

## Scientific question

> Does biomedical continual pretraining improve the underlying Qwen3.5 model before any BioJev decision supervision is introduced?

Sprint 2 creates **BioQwen**, an intermediate biomedical language model. BioQwen is *not yet BioJev*: no NLI/decision labels are used in this sprint.

## Base model

The DAPT pipeline starts from `Qwen/Qwen3.5-4B-Base`, the pretrained-only Qwen3.5 4B checkpoint. This is intentionally different from Sprint 1's instruction/post-trained Qwen baseline. Using the Base checkpoint makes the ablation causal: the main difference between Qwen-Base and BioQwen is biomedical continued pretraining.

## Corpus

Default mixture:

- 80% PubMed abstracts: `slinusc/PubMedAbstractsSubset` (streaming, ~2.39M records)
- 20% PMC Open Access full text: `aochongoliverli/pmc_openaccess_split`
- PMC defaults to commercial-use licenses only and excludes retracted records when metadata is available.

The corpus is **streamed**, not fully downloaded or materialized. Training and validation use deterministic document-ID hashing: 98% train / 1% validation / 1% test.

## Token budgets

Ready-made configurations:

- `configs/dapt/pubmed_pmc_100m.yaml`
- `configs/dapt/pubmed_pmc_500m.yaml`
- `configs/dapt/pubmed_pmc_1b.yaml`
- `configs/dapt/pubmed_pmc_3b.yaml`

All budgets count packed model tokens, not documents. Documents are tokenized without task labels and separated by EOS tokens.

## Plan a run without downloading anything

```powershell
python scripts/plan_dapt.py `
  --config configs/dapt/pubmed_pmc_100m.yaml `
  --world-size 1
```

This reports the exact packed-sequence and optimizer-step plan implied by the token budget.

## Training methods

The same pipeline supports:

- `qlora` — recommended for a single consumer GPU;
- `lora` — full-precision/bfloat16 base weights with trainable LoRA adapters;
- `full` — all text-model parameters trainable.

Override a YAML without editing it:

```powershell
python scripts/train_dapt.py `
  --config configs/dapt/pubmed_pmc_100m.yaml `
  --method qlora
```

For a tiny pipeline check:

```powershell
python scripts/train_dapt.py `
  --config configs/dapt/smoke.yaml
```

## Resume

Resume from the most recent Trainer checkpoint:

```powershell
python scripts/train_dapt.py `
  --config configs/dapt/pubmed_pmc_100m.yaml `
  --resume
```

Or from one exact checkpoint:

```powershell
python scripts/train_dapt.py `
  --config configs/dapt/pubmed_pmc_100m.yaml `
  --resume outputs/bioqwen/pubmed_pmc_100m/checkpoint-500
```

## Multi-GPU

The trainer uses Hugging Face Trainer/Accelerate. After `accelerate config`, launch for two GPUs with PowerShell:

```powershell
accelerate launch `
  --multi_gpu `
  --num_processes 2 `
  scripts/train_dapt.py `
  --config configs/dapt/pubmed_pmc_100m.yaml
```

For distributed QLoRA, each process quantizes onto its `LOCAL_RANK`. For LoRA/full fine-tuning, Trainer/Accelerate owns device placement.

## Logging

Every run writes:

```text
outputs/bioqwen/<run>/
├── resolved_config.json
├── environment.json
├── trainer_log.jsonl
├── training_manifest.json
├── checkpoint-*/
└── final/
```

TensorBoard is enabled by default:

```powershell
tensorboard --logdir outputs/bioqwen
```

Set `report_to: [tensorboard, wandb]` in a config to additionally use Weights & Biases.

## Evaluation 1 — held-out biomedical perplexity

Base model:

```powershell
python scripts/evaluate_perplexity.py `
  --model Qwen/Qwen3.5-4B-Base `
  --config configs/dapt/pubmed_pmc_100m.yaml `
  --partition validation `
  --tokens 1000000 `
  --output results/perplexity/qwen35_4b_base.json
```

BioQwen:

```powershell
python scripts/evaluate_perplexity.py `
  --model outputs/bioqwen/pubmed_pmc_100m/final `
  --config configs/dapt/pubmed_pmc_100m.yaml `
  --partition validation `
  --tokens 1000000 `
  --output results/perplexity/bioqwen_100m.json
```

The held-out partition is deterministic and excluded from DAPT training.

## Evaluation 2 — Sprint-1 downstream datasets

Sprint 2 uses **conditional likelihood**, not instruction generation. This lets the pretrained-only Qwen-Base checkpoint and BioQwen select among NLI/relation hypotheses without either model having decision supervision.

```powershell
python scripts/run_sprint2_suite.py `
  --config configs/suites/sprint2_ablation.yaml
```

Then aggregate Qwen-vs-BioQwen results:

```powershell
python scripts/compare_sprint2.py `
  --summary results/sprint2/summary.json `
  --output results/sprint2/comparison.csv
```

## Hugging Face export

PEFT adapter export:

```powershell
python scripts/export_hf.py `
  --checkpoint outputs/bioqwen/pubmed_pmc_100m/final `
  --output exports/BioQwen-4B-100M
```

Merged checkpoint (requires enough RAM/VRAM to load and merge the base model):

```powershell
python scripts/export_hf.py `
  --checkpoint outputs/bioqwen/pubmed_pmc_100m/final `
  --output exports/BioQwen-4B-100M-merged `
  --merge
```

Push the exported directory:

```powershell
python scripts/export_hf.py `
  --checkpoint outputs/bioqwen/pubmed_pmc_100m/final `
  --output exports/BioQwen-4B-100M `
  --push-to-hub `
  --repo-id Gabriel382/BioQwen-4B-100M
```

## Expected deliverable

Sprint 2 ends with:

1. one or more BioQwen checkpoints at controlled token budgets;
2. held-out biomedical perplexity comparisons against Qwen3.5-4B-Base;
3. multi-dataset Sprint-1 comparisons using identical likelihood-based evaluation;
4. a reproducible training manifest describing the exact corpus mixture and token budget.
