# BioJev Sprint 8 — Adaptive Native Ollama Benchmark (v1.1)

**Drop-in overlay** for the existing BioJev repository. Uses Python standard library only.

## What changes

- Replaces **only** `scripts/eval_sprint8_ollama_models.py` with an adaptive version.
- Adds `scripts/create_sprint8_ollama_context_variants.py` to create **optional, local** 4096-token Ollama tags.
- Adds `tests/test_sprint8_adaptive_ollama.py` for offline regression tests.
- **Does not modify** `configs/sprint8/*.yaml`, existing GGUF files, primary Ollama tags, adapters, or training scripts.

## Install

From the BioJev repository root, where the ZIP was downloaded/copied:

```bash
unzip -o BioJev_Sprint8_Adaptive_Ollama_Eval_v1.1.zip -d .
source .venv/bin/activate
python -m unittest discover -s tests -p 'test_sprint8_adaptive_ollama.py' -v
```

## Create optional 4K Ollama profiles **once** (no retraining)

Inspect the changes without applying:

```bash
python scripts/create_sprint8_ollama_context_variants.py --dry-run
```

Create all three local variants:

```bash
python scripts/create_sprint8_ollama_context_variants.py
```

The script **refuses to proceed** if any original model does not show `PARAMETER num_ctx 2048` in `ollama show --modelfile`. It does not change these originals.

| Original (unchanged) | Additional local variant |
|---|---|
| `biojev-systemone:0.8b` | `biojev-systemone:0.8b-4k` |
| `biojev-systemone:4b` | `biojev-systemone:4b-4k` |
| `biojev-systemone:9b` | `biojev-systemone:9b-4k` |

Variants inherit their original GGUF weights; `CAPABILITY decision` is explicitly declared. They are separate Ollama **tags**, not new training checkpoints. They persist as optional local profiles until explicitly removed, but **do not change** the default 2048-token settings of the original tags.

## Run the benchmark

```bash
python scripts/eval_sprint8_ollama_models.py \
  --per-source 5 \
  --output results/sprint8/native_ollama_eval_adaptive_5
```

Increase to 50 examples per source:

```bash
python scripts/eval_sprint8_ollama_models.py \
  --per-source 50 \
  --output results/sprint8/native_ollama_eval_adaptive_50
```

The benchmark calls **only the original 2048-token model first**. If Ollama returns HTTP 400 with the specific `prompt 0 has N tokens; expected 1–2048` diagnostic, it retries that one example with the corresponding `-4k` tag, **only if N <= 4096**. There is no truncation. HTTP 400 errors of other kinds are not retried. Inputs longer than 4096 are explicitly marked as errors; the script does not silently truncate or increase further.

### New audit columns

`predictions.csv` retains all previous columns and adds:

- `model_used` — actual base or 4K Ollama model tag
- `context_limit_tokens` — 2048 or 4096
- `attempts` — 1 or 2 HTTP requests
- `overflow_tokens` — prompt size reported on the initial 2048-token rejection

`summary.csv` retains its original metrics and adds `n_extended` (successfully evaluated after retry). Latency includes **both** HTTP requests when fallback occurs. Stopping Ollama between model families also stops both the primary and extended variants.

Run with the original strict 2048-token benchmark behavior:

```bash
python scripts/eval_sprint8_ollama_models.py \
  --per-source 5 --no-adaptive-context \
  --output results/sprint8/native_ollama_eval_strict_2k
```

## Scientific limitation

All models were fine-tuned using `max_length: 2048`; the optional 4K mode changes **inference context capacity, not training**. Report 2048 and longer-context cases separately. Evaluating on Sprint 8 dev data is validation, **not** an independent held-out test. This overlay has local automated mock tests but has not been executed against the user's live remote Ollama server.
