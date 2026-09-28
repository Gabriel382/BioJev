# BioJev

BioJev is a biomedical decision-model research project built from the OpenJev/Qwen3.5 line. The repository is cumulative and reproducible: Sprint 1 builds the benchmark, Sprint 2 performs PubMed/PMC domain-adaptive pretraining (BioQwen), and **Sprint 3 builds the actual biomedical decision model**.

The development strategy is **Nano first**: complete every sprint with Qwen3.5-0.8B, then rerun the same code/config structure at 4B and larger scales.

Sprint 3 adds weighted decision/NLI training, current OpenJev v5 baselines, and the main **Base-aligned BioJev lineage**: Sprint-2 biomedical DAPT -> general NLI/OpenJev-style decision training -> biomedical NLI. It also adds a typed `decide(context, hypotheses)` interface. OpenJev is kept as a baseline/adaptation comparison rather than mixed with a DAPT adapter trained against a different parent checkpoint.

See `docs/SPRINT3.md` for the current workflow.

## Repository layout

```text
BioJev/
├── configs/
│   ├── dapt/                    # Sprint-2 biomedical DAPT budgets
│   ├── decision/                # Sprint-3 decision-training recipes
│   ├── models/                  # OpenJev, Qwen, BioQwen and local BioJev models
│   └── suites/                  # reproducible evaluation matrices
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   ├── DATASETS.md
│   ├── CORPUS.md
│   ├── EXPERIMENT_PROTOCOL.md
│   ├── SPRINT2.md
│   └── SPRINT3.md
├── notebooks/
│   ├── 01_sprint1_walkthrough.py/.ipynb
│   ├── 02_sprint2_walkthrough.py/.ipynb
│   └── 03_sprint3_walkthrough.py/.ipynb
├── outputs/
│   ├── bioqwen/                 # ignored DAPT outputs
│   └── biojev/                  # ignored decision-model outputs
├── scripts/
│   ├── powershell/
│   ├── download_datasets.py
│   ├── train_dapt.py
│   ├── plan_decision.py
│   ├── train_decision.py
│   ├── decide.py
│   ├── export_biojev_hf.py
│   ├── evaluate.py
│   └── run_suite.py
├── src/biojev/
│   ├── corpus/
│   ├── datasets/
│   ├── evaluation/
│   ├── models/
│   ├── training/
│   └── transformations/
└── tests/
```

## 1. Setup — Windows PowerShell

Python 3.11+ is recommended.

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,plots,notebook,train]"
```

Or:

```powershell
.\scripts\powershell\sprint2_setup.ps1
```

QLoRA uses `bitsandbytes`. Current releases support NVIDIA CUDA on Windows, but the installed PyTorch CUDA build and bitsandbytes wheel still need to be compatible.

## 2. Sprint-1 datasets

```powershell
python scripts/download_datasets.py --all
```

The command now ends with a readable summary table showing every dataset, split and grand total.

Included public datasets:

```text
BioNLI
NLI4CT
ChemProt
DDI2013
BioRED
```

MedNLI remains optional/manual because of PhysioNet access restrictions.

## 3. Inspect the biomedical streaming corpus

No full PubMed/PMC clone is required.

```powershell
python scripts/inspect_corpus.py `
  --config configs/dapt/pubmed_pmc_100m.yaml `
  --documents 10 `
  --show-text
```

The default mixture is 80% PubMed abstracts and 20% PMC Open Access full text. PMC defaults to commercial-use licenses only.

## 4. Inspect a training plan without downloading the model

```powershell
python scripts/plan_dapt.py `
  --config configs/dapt/pubmed_pmc_100m.yaml `
  --world-size 1
```

This prints the token budget, packed-sequence count, effective batch size, optimizer-step count and corpus mixture.

## 5. Smoke-test Sprint 2

The smoke configuration uses only 50K tokens but still loads the real Qwen3.5-4B-Base model.

```powershell
python scripts/train_dapt.py `
  --config configs/dapt/smoke.yaml
```

Or:

```powershell
.\scripts\powershell\sprint2_smoke.ps1
```

## 6. Train BioQwen

Recommended first scientific run:

```powershell
python scripts/train_dapt.py `
  --config configs/dapt/pubmed_pmc_100m.yaml
```

Other ready-made budgets:

```text
configs/dapt/pubmed_pmc_500m.yaml
configs/dapt/pubmed_pmc_1b.yaml
configs/dapt/pubmed_pmc_3b.yaml
```

Override the fine-tuning method from PowerShell:

```powershell
python scripts/train_dapt.py `
  --config configs/dapt/pubmed_pmc_100m.yaml `
  --method lora `
  --output-dir outputs/bioqwen/pubmed_pmc_100m_lora
```

Or full fine-tuning:

```powershell
python scripts/train_dapt.py `
  --config configs/dapt/pubmed_pmc_100m.yaml `
  --method full `
  --output-dir outputs/bioqwen/pubmed_pmc_100m_full
```

Full 4B fine-tuning requires substantially more memory than QLoRA and is mainly intended for a suitable multi-GPU machine.

## 7. Resume a run

Latest checkpoint:

```powershell
python scripts/train_dapt.py `
  --config configs/dapt/pubmed_pmc_100m.yaml `
  --resume
```

Exact checkpoint:

```powershell
python scripts/train_dapt.py `
  --config configs/dapt/pubmed_pmc_100m.yaml `
  --resume outputs/bioqwen/pubmed_pmc_100m/checkpoint-500
```

## 8. Multi-GPU

After configuring Accelerate:

```powershell
accelerate config
```

Launch, for example, on two GPUs:

```powershell
accelerate launch `
  --multi_gpu `
  --num_processes 2 `
  scripts/train_dapt.py `
  --config configs/dapt/pubmed_pmc_100m.yaml
```

Or:

```powershell
.\scripts\powershell\train_bioqwen_multigpu.ps1 -NumProcesses 2
```

## 9. Qwen vs BioQwen: intrinsic biomedical language modeling

Qwen3.5-4B-Base:

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

The validation documents are isolated by a deterministic document-ID split and are not used during DAPT.

## 10. Qwen vs BioQwen: Sprint-1 benchmark

Because Sprint 2 intentionally contains no decision supervision, this comparison uses **mean conditional log-likelihood** over candidate labels/hypotheses rather than instruction-following generation.

```powershell
python scripts/run_sprint2_suite.py `
  --config configs/suites/sprint2_ablation.yaml
```

Aggregate:

```powershell
python scripts/compare_sprint2.py `
  --summary results/sprint2/summary.json `
  --output results/sprint2/comparison.csv
```

Or run the complete evaluation block:

```powershell
.\scripts\powershell\sprint2_benchmark.ps1
```

## 11. Export BioQwen to Hugging Face format

Adapter export:

```powershell
python scripts/export_hf.py `
  --checkpoint outputs/bioqwen/pubmed_pmc_100m/final `
  --output exports/BioQwen-4B-100M
```

Merged model:

```powershell
python scripts/export_hf.py `
  --checkpoint outputs/bioqwen/pubmed_pmc_100m/final `
  --output exports/BioQwen-4B-100M-merged `
  --merge
```

Push when ready:

```powershell
python scripts/export_hf.py `
  --checkpoint outputs/bioqwen/pubmed_pmc_100m/final `
  --output exports/BioQwen-4B-100M `
  --push-to-hub `
  --repo-id Gabriel382/BioQwen-4B-100M
```

## 12. Notebook and cell-based Python workflows

For VS Code/Spyder cells:

```text
notebooks/02_sprint2_walkthrough.py
```

For Jupyter:

```powershell
python -m jupyter lab
```

then open:

```text
notebooks/02_sprint2_walkthrough.ipynb
```

See `docs/SPRINT2.md` for the Sprint-2 domain-adaptation protocol.

## 13. Sprint 3 — BioJev decision model

The default development path uses the already-trained Nano DAPT adapter:

```text
Qwen3.5-0.8B-Base
  -> biomedical DAPT (Sprint 2)
  -> general NLI decision training
  -> biomedical NLI decision training
  -> BioJev-Nano
```

PowerShell:

```powershell
python scripts\download_datasets.py --sprint3-general
python scripts\plan_decision.py --config configs\decision\nano_full.yaml
python scripts\train_decision.py --config configs\decision\nano_full.yaml
```

The final Nano decision checkpoint is written to:

```text
outputs/biojev/nano_full/stage-02-biomedical/final
```

The same training code is used by `configs/decision/4b_full.yaml` after the 4B DAPT checkpoint exists. See `docs/SPRINT3.md` and `notebooks/03_sprint3_walkthrough.*` for the complete workflow.

## Research scope

Sprint 3 establishes the reusable BioJev decision-training stack and its key lineage ablations. Multi-dataset transfer, reliability/calibration and the exhaustive paper ablations remain the focus of later sprints.
