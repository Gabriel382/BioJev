# Sprint 8 — Native Ollama System One BioJev

## Goal

Train a **causal** BioJev decision model that works directly with Ollama's
`POST /v1/systemone` endpoint in the same way as Tev1.

This is distinct from the existing BioJev NLI sequence classifiers and from the
NLI-to-System-One compatibility bridge.

## Architecture

```text
Qwen3.5 Base
    ↓
existing BioQwen biomedical DAPT adapter
    ↓ merge
Biomedical causal LM
    ↓
Tev-style decision SFT
(completion-only loss on A..Z)
    ↓
BioJev-SystemOne
    ↓ merge LoRA
HF causal checkpoint
    ↓
GGUF + qwen35.decision.type=tev1
    ↓
Ollama CAPABILITY decision
    ↓
POST /v1/systemone
```

Presets are supplied for **0.8B, 4B and 9B**. The BioJev Qwen3.5 scale is 9B,
so the package uses 9B rather than inventing an 8B checkpoint.

## Data

General decision behavior comes from Together's public Tev1 recipe. Biomedical
specialization adds BioNLI and NLI4CT from the existing BioJev processed data.

ChemProt, DDI2013 and BioRED are excluded by default so they remain transfer
benchmarks.

Build Tev data separately:

```bash
git clone https://github.com/togethercomputer/tev1 external/tev1
cd external/tev1
uv sync --locked
uv run python fetch_sources.py
uv run python build_all.py
cd ../..
```

Review third-party dataset licenses before redistributing data.

## Install

```bash
pip install -r requirements-sprint8.txt
pip install -e .
```

## First run: 0.8B smoke pipeline

Build a tiny synthetic dataset:

```bash
python scripts/build_sprint8_dataset.py \
  --config configs/sprint8/systemone_nano.yaml \
  --smoke --overwrite
```

Merge the existing Nano biomedical DAPT adapter into the causal Qwen base:

```bash
python scripts/prepare_sprint8_base.py \
  --config configs/sprint8/systemone_nano.yaml \
  --device cpu
```

Validate that A-Z are one-token candidates:

```bash
python scripts/validate_sprint8_tokenizer.py \
  --config configs/sprint8/systemone_nano.yaml
```

Smoke train:

```bash
python scripts/train_sprint8_systemone.py \
  --config configs/sprint8/systemone_nano.yaml \
  --smoke
```

Evaluate candidate-letter preference before conversion:

```bash
python scripts/eval_sprint8_letter.py \
  --config configs/sprint8/systemone_nano.yaml
```

## Full 0.8B run

After building external Tev data:

```bash
python scripts/build_sprint8_dataset.py \
  --config configs/sprint8/systemone_nano.yaml \
  --overwrite

python scripts/train_sprint8_systemone.py \
  --config configs/sprint8/systemone_nano.yaml
```

## Export to native Ollama

Merge Sprint-8 LoRA:

```bash
python scripts/merge_sprint8_adapter.py \
  --config configs/sprint8/systemone_nano.yaml \
  --device cpu
```

Prepare recent llama.cpp with Qwen3.5 support:

```bash
git clone https://github.com/ggml-org/llama.cpp external/llama.cpp
cd external/llama.cpp
cmake -B build
cmake --build build -j
cd ../..
```

Convert and optionally quantize:

```bash
python scripts/export_sprint8_gguf.py \
  --config configs/sprint8/systemone_nano.yaml \
  --llama-cpp external/llama.cpp \
  --quantize Q8_0
```

The export metadata declares:

```text
qwen35.decision.type = tev1
```

Create the Ollama model (Ollama >= 0.35.1):

```bash
python scripts/make_sprint8_ollama.py \
  --config configs/sprint8/systemone_nano.yaml
```

The generated Modelfile includes:

```text
CAPABILITY decision
```

Run the acceptance test:

```bash
python scripts/test_sprint8_ollama.py \
  --model biojev-systemone:0.8b
```

This sends `choice`, `noul`, and `score` together to the **real Ollama**
`/v1/systemone` endpoint. No BioJev Python decision bridge is used.

## Scale to 4B

Use the same commands with:

```text
configs/sprint8/systemone_4b.yaml
```

The DAPT source is:

```text
outputs/bioqwen/pubmed_pmc_100m/final
```

## Scale to 9B

Use:

```text
configs/sprint8/systemone_9b.yaml
```

The DAPT source is:

```text
outputs/bioqwen-9b/pubmed_pmc_100m/final
```

## Acceptance criteria

Do not label a release "native Ollama System One" until:

1. A-Z tokenizer validation passes.
2. Candidate-letter dev accuracy is recorded.
3. The SFT adapter merges into a Qwen3.5 causal LM.
4. The merged model converts to GGUF.
5. Ollama creates it as a decision model.
6. `/v1/systemone` returns HTTP 200 for `choice`.
7. `/v1/systemone` returns HTTP 200 for `noul`.
8. `/v1/systemone` returns HTTP 200 for `score`.
9. A mixed multi-question request succeeds.
10. At API level, switching `model` from `tev1:0.8b` to
   `biojev-systemone:0.8b` is sufficient.

## Scientific comparison

Keep both:

```text
BioJev NLI + compatibility bridge
vs.
BioJev-SystemOne native causal decision model
```

That allows a direct test of whether native typed-decision training is superior
to NLI adaptation for biomedical decisions.
