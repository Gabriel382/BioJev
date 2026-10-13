# BioJev Sprint 8: targeted NLI4CT context repair

This additive overlay addresses **only 11 unique NLI4CT examples** that failed because their prompts require more than 4096 tokens. Each of the three trained BioJev-SystemOne models has the same failed IDs (33 predictions total).

## What this script does

1. Reads **existing** `results/sprint8/native_ollama_eval_full/predictions.csv` and `run_config.json`, and retrieves those exact examples from the original dev JSONL.
2. Refuses to touch non-context errors, cases in other datasets, or unequal ID sets across models.
3. Inspects and verifies that all three original Ollama model tags still use `PARAMETER num_ctx 2048`.
4. Creates **temporary local Ollama variants** of the original weights, with `num_ctx 5120` (5 Ki tokens) and `num_ctx 6144` (6 Ki tokens). Only the 11 previously failing examples per model are sent to these variants. Successful records are not re-evaluated.
5. Stops and **removes** every temporary tag after use, including if requests fail; the original models, GGUFs, adapters and existing `-4k` profiles remain unchanged.
6. Writes a **new result set** in `results/sprint8/native_ollama_eval_full_repaired/` (never modifies the original run) with recalculated per-source and overall metrics, plus a `repairs.csv` audit trail.

## The precise 11 cases

| ID | Reported prompt tokens | Temporary context |
|---|---:|---:|
| nli4ct:72 | 4181 | 5120 |
| nli4ct:44 | 4573 | 5120 |
| nli4ct:144 | 4100 | 5120 |
| nli4ct:151 | 4572 | 5120 |
| nli4ct:80 | 4177 | 5120 |
| nli4ct:197 | 5044 | 5120 |
| nli4ct:117 | 4456 | 5120 |
| nli4ct:175 | 5225 | 6144 |
| nli4ct:29 | 5045 | 5120 |
| nli4ct:69 | 5216 | 6144 |
| nli4ct:160 | 4445 | 5120 |

**9** unique cases need 5120, **2** need 6144, all repeated across **3** models.

## Install

From BioJev repo root, *with v1.1 adaptive evaluator already installed*:

```bash
unzip -o BioJev_Sprint8_NLI4CT_Selective_Repair_v1.0.zip -d .
source .venv/bin/activate
python -m unittest discover -s tests -p 'test_sprint8_selective_repair.py' -v
```

The patch adds one script and one test; it **does not replace** `scripts/eval_sprint8_ollama_models.py`.

## Preview: read-only, no Ollama requests

```bash
python scripts/repair_sprint8_nli4ct_outliers.py --dry-run
```

This prints the precise failed IDs and required token budget. It refuses to start if the stored benchmark differs from the expected 11 NLI4CT over-context cases per model.

## Apply just these 33 missing predictions

```bash
python scripts/repair_sprint8_nli4ct_outliers.py
```

Expected output files (in a NEW directory):

```text
results/sprint8/native_ollama_eval_full_repaired/
  predictions.csv     all original rows, with only 33 failed rows updated
  summary.csv         recomputed overall and per-dataset accuracy/latencies
  repairs.csv         audit only for the 33 targeted requests
  run_config.json     original configuration plus selective_repair metadata
```

**No full rerun** and **no permanent context changes**. Each example is sent once to the temporary model with the smallest 1024-token tier that accommodates it. No 2K/4K calls are repeated for those examples.

If the output directory already exists and contains files, the script refuses to overwrite it. To rerun, choose another destination:

```bash
python scripts/repair_sprint8_nli4ct_outliers.py \
  --output results/sprint8/native_ollama_eval_full_repaired_second_run
```

## Important benchmark notes

- The models were **fine-tuned with max_length=2048**. Long-context evaluation checks technical usability beyond that limit; it is not evidence of training with longer sequences.
- The data here are Sprint 8 **dev**, not a separate held-out test. Treat accuracy as validation metrics.
- The original 6757 valid predictions per model retain their old elapsed times. The 11 repaired rows, if successful, record **only the new 5K/6K call latency**, not the previous failed call. Consequently the combined timing in `summary.csv` is a mixed-run descriptive metric; use a clean, consistently configured benchmark for publication-grade latency.
- `repairs.csv` preserves the original HTTP 400 error, size requirement, temporary model name, new prediction, and result.
- A temporary variant loads the same frozen weights with more context; it **does not create new trained weights**, change the original model, or duplicate the GGUF checkpoint.
- The script uses standard library Python and the existing `scripts/eval_sprint8_ollama_models.py` functions. It was mock-tested locally and cannot be validated against your remote Ollama server here.
