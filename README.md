# BioJev

<p align="center">
  <strong>Biomedical decision models built on Qwen3.5</strong>
</p>

<p align="center">
  Biomedical domain adaptation · General NLI · Biomedical NLI · Reliability · Ablations · System One API
</p>

---

## Overview

**BioJev** is a biomedical decision-model research framework built on the Qwen3.5 family.

The project studies how three different forms of adaptation contribute to biomedical decision making:

1. **Biomedical domain-adaptive pretraining (DAPT)** on PubMed and PMC.
2. **General decision training** using natural-language inference datasets.
3. **Biomedical decision specialization** using biomedical and clinical NLI datasets.

The goal is not only to improve predictive performance, but also to study **cross-task transfer, calibration, confidence, selective prediction, and component-level ablations**.

BioJev is developed as a reproducible experimental pipeline rather than as a single checkpoint.

---

## People

**Creator:** Gabriel Henrique Alencar Medeiros  
**Supervisor:** Lina F. Soualmia

BioJev is developed at **LITIS / Université de Rouen Normandie**.

---

## Model checkpoints

Public BioJev checkpoints are available on Hugging Face:

| Model | Hugging Face |
|---|---|
| **BioJev-Nano** | https://huggingface.co/Gabriel382/BioJev-Nano |
| **BioJev-4B** | https://huggingface.co/Gabriel382/BioJev |

BioJev-Nano is primarily used for controlled ablation experiments and lightweight inference.

BioJev-4B is the main model used for the primary evaluation pipeline.

---

## Architecture

The full BioJev training pipeline is:

```text
Qwen3.5 Base
      │
      ▼
Biomedical Domain-Adaptive Pretraining
PubMed + PMC
      │
      ▼
BioQwen
      │
      ▼
General NLI
SNLI + MNLI + ANLI
      │
      ▼
Biomedical NLI
BioNLI + NLI4CT
      │
      ▼
BioJev
```

The current BioJev checkpoints use a **three-class sequence-classification head**:

```text
contradiction
entailment
neutral
```

---

## Training stages

### 1. Biomedical DAPT

Biomedical domain-adaptive pretraining creates the BioQwen initialization.

The main 4B configuration uses:

```text
100M biomedical tokens
80% PubMed
20% PMC Open Access
sequence length 2048
QLoRA
```

Example:

```bash
python scripts/train_dapt.py \
  --config configs/dapt/pubmed_pmc_100m.yaml
```

---

### 2. General NLI

The general decision stage uses:

```text
SNLI      35,000
MNLI      45,000
ANLI R1   20,000
----------------
Total    100,000
```

This stage is designed to provide general inference and transfer capability before biomedical specialization.

---

### 3. Biomedical NLI

The biomedical stage uses:

```text
BioNLI    30,000
NLI4CT    10,000
----------------
Total     40,000
```

Example full BioJev training:

```bash
python scripts/train_decision.py \
  --config configs/decision/4b_full.yaml
```

---

## Datasets

BioJev is currently evaluated on five biomedical datasets:

| Dataset | Task |
|---|---|
| **BioNLI** | Biomedical natural-language inference |
| **NLI4CT** | Clinical-trial natural-language inference |
| **ChemProt** | Chemical–protein relation classification |
| **DDI2013** | Drug–drug interaction classification |
| **BioRED** | Biomedical relation classification |

Relation datasets are evaluated as **gold entity-pair relation typing** tasks.

---

## Experimental pipeline

The repository is organized as cumulative experimental sprints.

### Sprint 1 — Data preparation

Download, normalize, validate, and freeze the benchmark datasets.

### Sprint 2 — Biomedical DAPT

Train BioQwen through domain-adaptive pretraining on PubMed and PMC.

### Sprint 3 — Decision training

Train the general-NLI and biomedical-NLI decision stages.

### Sprint 4 — Predictive evaluation

Evaluate:

- raw Qwen
- OpenJev-style baselines
- BioJev
- supervised biomedical encoder baselines

### Sprint 5 — Reliability

Evaluate:

```text
Accuracy
Macro-F1
Micro-F1
ECE
Adaptive ECE
Brier score
Negative log-likelihood
Mean confidence
AURC
Selective prediction
```

### Sprint 6 — Component ablations

BioJev-Nano is used to isolate the contribution of each stage.

The eight ablation variants are:

```text
general_only
biomedical_only
dapt_general
dapt_biomedical
general_biomedical
full
bionli_only
nli4ct_only
```

### Sprint 7 — Reproducibility and release

Sprint 7 freezes the complete experimental state:

```text
Git revision
training configurations
Python environment
PyTorch / CUDA versions
GPU information
result tables
model provenance
SHA-256 checksums
release artifacts
```

---

## Main ablation finding

The Nano ablation experiments indicate that the three BioJev stages play different roles.

- **General NLI training** provides the strongest contribution to transfer toward unseen biomedical tasks.
- **Biomedical NLI training** strongly specializes the model toward biomedical inference tasks, especially BioNLI.
- That specialization can reduce performance on some unseen relation-classification tasks.
- At the Nano scale, **biomedical DAPT produces smaller and more task-dependent gains** than general decision training.

These results suggest that domain adaptation, general decision learning, and biomedical specialization should be treated as distinct components rather than assuming that more biomedical specialization always improves cross-task transfer.

---

# System One compatibility API

BioJev includes a compatibility layer that exposes the existing BioJev NLI checkpoints through a **Jev/System-One-shaped API**.

The endpoint is:

```text
POST /v1/systemone
```

This allows applications to send:

```text
state
questions
choice / noul / score
```

without retraining the current BioJev checkpoints.

> **Important:** this is an engineering compatibility bridge over BioJev's NLI classifier. It is not a native Ollama/System-One checkpoint and should be evaluated separately before making accuracy claims about arbitrary structured decisions.

---

## How the adapter works

For each candidate answer, the adapter builds an NLI pair:

```text
Premise:
<state>

Hypothesis:
Decision question: <instructions>
Candidate answer: <candidate>
```

BioJev then produces:

```text
contradiction
entailment
neutral
```

Candidate support is computed as:

```text
support(candidate) =
entailment_logit - contradiction_logit
```

The candidate supports are normalized with a softmax:

```text
probabilities = softmax(candidate_supports)
```

Candidates are evaluated sequentially with `batch_size=1` for robust compatibility with Qwen3.5 sequence-classification checkpoints.

---

## Install the API dependencies

Install BioJev first, then add the lightweight HTTP dependencies:

```bash
pip install fastapi uvicorn
```

---

## Serve BioJev-4B

```bash
python scripts/serve_systemone.py \
  --checkpoint outputs/biojev/4b_full/stage-02-biomedical/final \
  --model-name biojev-4b \
  --load-in-4bit \
  --host 0.0.0.0 \
  --port 8000
```

Check that the service is running:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "backend": "biojev-nli-bridge",
  "model": "biojev-4b",
  "checkpoint": "outputs/biojev/4b_full/stage-02-biomedical/final"
}
```

---

## Serve BioJev-Nano

```bash
python scripts/serve_systemone.py \
  --checkpoint outputs/biojev/sprint6_nano/full/stage-02-biomedical/final \
  --model-name biojev-nano \
  --load-in-4bit \
  --host 0.0.0.0 \
  --port 8000
```

You can also point the server to another compatible BioJev checkpoint.

---

## Choice example

```bash
curl http://127.0.0.1:8000/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "model": "biojev-4b",
    "state": "The patient has fever, productive cough, and a new lobar infiltrate.",
    "questions": {
      "diagnosis": {
        "type": "choice",
        "instructions": "Which diagnosis is best supported?",
        "criteria": {
          "pneumonia": "Community-acquired pneumonia",
          "asthma": "Acute asthma exacerbation",
          "migraine": "Migraine"
        }
      }
    }
  }'
```

Example response shape:

```json
{
  "model": "biojev-4b",
  "answers": {
    "diagnosis": {
      "type": "choice",
      "choice": "pneumonia",
      "probabilities": {
        "pneumonia": 0.81,
        "asthma": 0.15,
        "migraine": 0.04
      },
      "confidence": 0.53
    }
  },
  "usage": {
    "input_tokens": 123,
    "output_tokens": 0
  }
}
```

The values above are illustrative; actual probabilities are produced by BioJev.

---

## `noul` example

`noul` represents a binary decision.

```bash
curl http://127.0.0.1:8000/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "model": "biojev-4b",
    "state": "The patient has fever, productive cough, and a new lobar infiltrate.",
    "questions": {
      "infection": {
        "type": "noul",
        "instructions": "Is an infectious pulmonary process supported?"
      }
    }
  }'
```

The returned `noul` value is the probability assigned to the `true` candidate.

---

## Score example

```bash
curl http://127.0.0.1:8000/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "model": "biojev-4b",
    "state": "The patient has fever, productive cough, and a new lobar infiltrate.",
    "questions": {
      "severity": {
        "type": "score",
        "instructions": "How severe is the presentation?",
        "criteria": [
          "Low",
          "Moderate",
          "High"
        ]
      }
    }
  }'
```

The adapter returns:

- a weighted score,
- the score legend,
- probabilities for each level,
- a confidence value based on probability concentration.

---

## Local JSON CLI

Requests can also be executed without starting the HTTP server.

Save a request to `request.json`, then run:

```bash
python scripts/systemone_cli.py \
  --checkpoint outputs/biojev/4b_full/stage-02-biomedical/final \
  --request request.json \
  --model-name biojev-4b \
  --load-in-4bit
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Gabriel382/BioJev.git
cd BioJev
```

Create an environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the package:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

For development/training environments, install the appropriate optional dependencies defined by the repository.

---

## Data preparation

Download and prepare the supported datasets:

```bash
python scripts/download_datasets.py --all
```

Processed datasets are stored under:

```text
data/processed/
```

Large datasets, checkpoints, and model weights are intentionally excluded from normal Git history.

---

## Evaluation

Run the evaluation scripts associated with each sprint.

For example, the Sprint 6 Nano ablations are evaluated with:

```bash
python scripts/run_sprint6_eval.py \
  --config configs/sprint6/eval_nano.yaml \
  --skip-existing
```

Then generate the ablation tables:

```bash
python scripts/build_sprint6_tables.py
```

---

## Reproducibility

Audit the complete experimental state:

```bash
python scripts/run_sprint7_release.py \
  --config configs/sprint7/release.yaml \
  --audit-only
```

Build a reproducibility release:

```bash
python scripts/run_sprint7_release.py \
  --config configs/sprint7/release.yaml \
  --clean
```

Build an archival release including portable adapters:

```bash
python scripts/run_sprint7_release.py \
  --config configs/sprint7/release.yaml \
  --include-adapters \
  --hash-weights \
  --clean
```

Verify the generated release:

```bash
python scripts/verify_sprint7_release.py \
  releases/biojev-paper
```

---

## Project structure

```text
BioJev/
├── configs/
│   ├── dapt/
│   ├── decision/
│   ├── sprint4/
│   ├── sprint5/
│   ├── sprint6/
│   └── sprint7/
│
├── data/
├── docs/
│   └── SYSTEMONE.md
├── notebooks/
├── results/
├── scripts/
│   ├── serve_systemone.py
│   └── systemone_cli.py
├── src/
│   └── biojev/
│       └── systemone/
└── tests/
```

---

## Research status

BioJev is a research project.

The models, confidence estimates, and System One compatibility interface are **not validated clinical decision systems** and must not be used as the sole basis for diagnosis, treatment, or other high-stakes medical decisions.

---

## Citation

A formal publication citation will be added when the BioJev paper is available.

For now, if you use BioJev in research, please cite the repository and identify the model checkpoint used.

---

## Acknowledgements

BioJev was created by **Gabriel Henrique Alencar Medeiros** under the supervision of **Lina F. Soualmia** at **LITIS / Université de Rouen Normandie**.

The project builds on the Qwen3.5 model family and publicly available biomedical and NLI datasets.

---

## License

Repository code is distributed under the project license.

Datasets, pretrained models, and third-party artifacts remain subject to their respective licenses.
