#!/usr/bin/env python3
"""Repair ONLY failed NLI4CT over-context rows in an existing Sprint 8 Ollama run.

- Reads the existing predictions CSV; never reruns successful rows.
- Resends each failed example once to a temporary native Ollama model, at
  the smallest 1024-token context tier that accepts Ollama's token count.
- Creates/removes temporary derived tags. Original Ollama tags and GGUFs are
  never modified. Supports 5120 and 6144 tokens for the reported failures.
- Writes NEW predictions/summary/repair audit to a separate output directory.

Run from BioJev repository root with v1.1 benchmark evaluator installed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import tempfile
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path

from eval_sprint8_ollama_models import (post_choice, read_jsonl, save_csv, summary)

PROMPT_RE = re.compile(
    r"\bprompt\s+\d+\s+has\s+(\d+)\s+tokens;\s*expected\s+1\s*(?:[\u2013\u2014-]|\\u2013|\\u2014)\s*(\d+)",
    flags=re.IGNORECASE,
)
MODEL_CONTEXT_RE = re.compile(r"(?im)^\s*PARAMETER\s+num_ctx\s+(\d+)\s*$")
PREDICTION_FIELDS = [
    "model", "source", "id", "gold_key", "prediction", "confidence",
    "gold_probability", "elapsed_seconds", "error", "model_used",
    "context_limit_tokens", "attempts", "overflow_tokens",
]
SUMMARY_FIELDS = [
    "model", "source", "n", "n_success", "n_errors", "n_extended",
    "accuracy", "mean_latency_s", "median_latency_s", "p95_latency_s",
]
AUDIT_FIELDS = [
    "model", "source", "id", "gold_key", "prompt_tokens", "context_tokens",
    "temporary_model", "previous_error", "new_prediction", "new_error",
    "correct", "repair_latency_s",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def prompt_requirement(error: str) -> tuple[int, int] | None:
    """Extract exact prompt size / rejected limit from an Ollama overflow error."""
    if "HTTP 400" not in error:
        return None
    match = PROMPT_RE.search(error)
    if not match:
        return None
    tokens, limit = map(int, match.groups())
    return (tokens, limit) if tokens > limit else None


def required_context(prompt_tokens: int, step: int = 1024) -> int:
    return math.ceil(prompt_tokens / step) * step


def index_examples(data: list[dict]) -> dict[tuple[str, str], dict]:
    indexed = {}
    for ex in data:
        key = (str(ex.get("source", "unknown")), str(ex.get("id", "")))
        if key in indexed:
            raise ValueError(f"Duplicate source/id in dev.jsonl: {key}")
        indexed[key] = ex
    return indexed


def discover_repairs(rows, examples, source="nli4ct", max_context=6144,
                     expected_per_model=11):
    repairs = defaultdict(list)
    all_models = sorted(set(r["model"] for r in rows))
    seen = set()
    for position, row in enumerate(rows):
        key = (row["model"], row["source"], row["id"])
        if key in seen:
            raise ValueError(f"Duplicate prediction: {key}")
        seen.add(key)
        if not row["error"].strip():
            continue
        if row["source"] != source:
            raise ValueError(f"Unexpected failure outside {source}: {key}: {row['error']}")
        token_info = prompt_requirement(row["error"])
        if token_info is None:
            raise ValueError(f"Non-overflow failure cannot be selectively retried: {key}: {row['error']}")
        tokens, previous_limit = token_info
        # These are the failed cases after an existing adaptive 2K->4K benchmark.
        if previous_limit != 2048 or tokens <= 4096:
            raise ValueError(f"Unexpected overflow for {key}: {tokens}/{previous_limit}; only >4096 targeted")
        needed = required_context(tokens)
        if needed > max_context:
            raise ValueError(f"{key} requires {tokens} tokens, exceeding --max-context {max_context}")
        ex_key = (row["source"], row["id"])
        if ex_key not in examples:
            raise ValueError(f"Example missing from dev.jsonl: {ex_key}")
        ex = examples[ex_key]
        if str(ex.get("answer_key")) != row["gold_key"]:
            raise ValueError(f"Gold label mismatch for {key}")
        if not (isinstance(ex.get("options"), list) and 2 <= len(ex["options"]) <= 26):
            raise ValueError(f"Invalid choice options: {ex_key}")
        repairs[row["model"]].append({
            "row_index": position, "example": ex, "tokens": tokens,
            "context": needed, "prior_error": row["error"],
        })
    if not repairs:
        raise ValueError("No failed over-context rows found; nothing to repair")
    for model in all_models:
        count = len(repairs[model])
        if count != expected_per_model:
            raise ValueError(f"Expected {expected_per_model} errors for {model}; found {count}")
    signatures = [set((c["example"]["source"], c["example"]["id"], c["tokens"])
                      for c in repairs[model]) for model in all_models]
    if signatures and any(s != signatures[0] for s in signatures[1:]):
        raise ValueError("Different failed examples/token counts between models; refusing uneven benchmark")
    return {model: repairs[model] for model in all_models}


def ollama_modelfile(model):
    return subprocess.run(["ollama", "show", "--modelfile", model],
                          capture_output=True, text=True, check=True).stdout


def check_model_context(model, expected):
    matches = MODEL_CONTEXT_RE.findall(ollama_modelfile(model))
    actual = int(matches[-1]) if matches else None
    if actual != expected:
        raise RuntimeError(f"{model}: num_ctx={actual}; expected={expected}. Refusing model mutation.")


def create_variant(model: str, ctx: int, run_id: str):
    """Only create a random new tag; return tag for guaranteed finally cleanup."""
    if ":" not in model or ctx <= 4096:
        raise ValueError(f"Unsafe model or context: {model}/{ctx}")
    tag = f"{model}-s8repair{ctx // 1024}k-{run_id}"
    modelfile = f"FROM {model}\nCAPABILITY decision\nPARAMETER num_ctx {ctx}\n"
    with tempfile.TemporaryDirectory(prefix="biojev-s8-repair-") as tmp:
        path = Path(tmp) / "Modelfile"
        path.write_text(modelfile, encoding="utf-8")
        try:
            subprocess.run(["ollama", "create", tag, "-f", str(path)], check=True)
        except Exception:
            # An interrupted create could leave a partial tag; remove only our
            # fresh, uniquely named temporary tag.
            remove_variant(tag)
            raise
    try:
        check_model_context(tag, ctx)
    except Exception:
        remove_variant(tag)
        raise
    return tag


def remove_variant(tag):
    # Never remove an original: only the unique temporary repair tag.
    if "-s8repair" not in tag:
        raise ValueError(f"Refusing to delete non-repair Ollama tag {tag}")
    subprocess.run(["ollama", "stop", tag], check=False, capture_output=True)
    result = subprocess.run(["ollama", "rm", tag], check=False,
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(f"WARNING: could not remove temporary tag {tag}: {result.stderr.strip()}", flush=True)


def score_one(base_url, model, example, timeout, keep_alive):
    criteria = {str(opt["key"]): str(opt.get("description") or opt["key"])
                for opt in example["options"]}
    payload = {
        "model": model,
        "state": example["state"],
        "questions": {"decision": {
            "type": "choice", "instructions": example["question"],
            "criteria": criteria,
        }},
        "keep_alive": keep_alive,
    }
    start = time.perf_counter()
    try:
        result = post_choice(base_url, payload, timeout)
        answer = result["answers"]["decision"]
        if answer["type"] != "choice":
            raise ValueError(f"Expected 'choice', received {answer.get('type')}")
        prediction = str(answer["choice"])
        if prediction not in criteria:
            raise ValueError(f"Unknown choice {prediction!r}")
        return {
            "prediction": prediction,
            "confidence": answer.get("confidence", ""),
            "gold_probability": answer.get("probabilities", {}).get(example["answer_key"], ""),
            "elapsed_seconds": time.perf_counter() - start,
            "error": "",
        }
    except Exception as exc:
        if hasattr(exc, "read") and hasattr(exc, "code"):
            detail = exc.read().decode("utf-8", "replace")[:1200]
            message = f"HTTP {exc.code}: {detail}"
        else:
            message = f"{type(exc).__name__}: {exc}"
        return {
            "prediction": "", "confidence": "", "gold_probability": "",
            "elapsed_seconds": time.perf_counter() - start, "error": message,
        }


def recompute_summary(rows):
    by_model = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)
    aggregate = []
    for model, model_rows in by_model.items():
        aggregate.append(summary(model, "ALL", model_rows))
        for source in sorted(set(r["source"] for r in model_rows)):
            aggregate.append(summary(model, source,
                                     [r for r in model_rows if r["source"] == source]))
    return aggregate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=Path("results/sprint8/native_ollama_eval_full"))
    ap.add_argument("--output", type=Path, default=Path("results/sprint8/native_ollama_eval_full_repaired"))
    ap.add_argument("--data", type=Path, default=None,
                    help="Defaults to 'data' from the benchmark's run_config.json")
    ap.add_argument("--source", default="nli4ct")
    ap.add_argument("--expected-per-model", type=int, default=11)
    ap.add_argument("--max-context", type=int, default=6144)
    ap.add_argument("--base-url", default="http://127.0.0.1:11434")
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--keep-alive", default="5m")
    ap.add_argument("--dry-run", action="store_true", help="List the target IDs without contacting Ollama")
    args = ap.parse_args()
    if args.input.resolve() == args.output.resolve():
        ap.error("--output MUST be different from --input; original results are immutable")
    if args.output.exists() and any(args.output.iterdir()):
        ap.error(f"Output already contains files: {args.output}; choose a new --output")
    run_config_path = args.input / "run_config.json"
    run_config = json.loads(run_config_path.read_text(encoding="utf-8")) if run_config_path.is_file() else {}
    data_path = args.data or Path(run_config.get("data", "data/sprint8/systemone_nano/dev.jsonl"))
    data_index = index_examples(read_jsonl(data_path))
    original = read_csv(args.input / "predictions.csv")
    if not original:
        ap.error("No predictions found")
    repairs = discover_repairs(original, data_index, args.source,
                               args.max_context, args.expected_per_model)
    tier_counts = Counter((c["context"] for tasks in repairs.values() for c in tasks))
    unique_cases = sorted({(c["example"]["id"], c["tokens"], c["context"])
                           for tasks in repairs.values() for c in tasks}, key=lambda x: x[0])
    print(f"Original predictions: {len(original)}; failed unique IDs: {len(unique_cases)}", flush=True)
    print("Requested contexts:", dict(sorted(tier_counts.items())), "(across all models)", flush=True)
    for ex_id, tokens, ctx in unique_cases:
        print(f"  {ex_id:<13} tokens={tokens:>5} -> temporary context={ctx}", flush=True)
    if args.dry_run:
        print("DRY RUN: no calls or changes to Ollama or saved benchmark", flush=True)
        return

    # Safety check on ALL original models BEFORE creating any temporary profiles.
    for model in repairs:
        check_model_context(model, 2048)

    run_id = uuid.uuid4().hex[:8]
    repaired = [r.copy() for r in original]
    audit = []
    total = sum(map(len, repairs.values()))
    completed = 0
    for model, tasks in repairs.items():
        variants = {}
        try:
            for ctx in sorted({c["context"] for c in tasks}):
                variants[ctx] = create_variant(model, ctx, run_id)
                print(f"Temporary variant created: {variants[ctx]}", flush=True)
            for case in tasks:
                index = case["row_index"]
                old = repaired[index]
                ctx = case["context"]
                temp_model = variants[ctx]
                result = score_one(args.base_url, temp_model, case["example"],
                                   args.timeout, args.keep_alive)
                completed += 1
                row = {
                    **old, **result,
                    "model_used": temp_model,
                    "context_limit_tokens": ctx,
                    "attempts": int(old.get("attempts", "1") or 1) + 1,
                    "overflow_tokens": case["tokens"],
                }
                repaired[index] = row
                audit.append({
                    "model": model, "source": old["source"], "id": old["id"],
                    "gold_key": old["gold_key"], "prompt_tokens": case["tokens"],
                    "context_tokens": ctx, "temporary_model": temp_model,
                    "previous_error": case["prior_error"],
                    "new_prediction": result["prediction"], "new_error": result["error"],
                    "correct": (str(result["prediction"] == old["gold_key"])
                                if not result["error"] else ""),
                    "repair_latency_s": result["elapsed_seconds"],
                })
                status = "OK" if not result["error"] else "ERROR"
                print(f"[{completed}/{total}] {model} {old['id']} {ctx} tokens: {status}", flush=True)
        finally:
            # Unconditionally clean up only the tags created in this script.
            for tag in reversed(list(variants.values())):
                remove_variant(tag)
                print(f"Temporary tag removed: {tag}", flush=True)

    # Write only after attempting all failed rows. Original dir remains immutable.
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        save_csv(args.output / "predictions.csv", repaired, PREDICTION_FIELDS)
        combined_summary = recompute_summary(repaired)
        save_csv(args.output / "summary.csv", combined_summary, SUMMARY_FIELDS)
        save_csv(args.output / "repairs.csv", audit, AUDIT_FIELDS)
        updated_config = {**run_config, "selective_repair": {
            "original_results": str(args.input),
            "data": str(data_path), "source": args.source,
            "max_context": args.max_context,
            "temporary_variants_removed": True,
            "n_retried": len(audit),
            "n_repaired": sum(not r["new_error"] for r in audit),
            "latency_note": "Repaired row latency = second-run temporary model request only, not prior failed request",
            "contexts": {str(k): v for k, v in sorted(tier_counts.items())},
        }}
        (args.output / "run_config.json").write_text(json.dumps(updated_config, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        print("Failed to write output; original results were not changed", flush=True)
        raise
    print("\nREPAIRED BENCHMARK (successful requests only):")
    for row in combined_summary:
        if row["source"] == "ALL" or row["source"] == "nli4ct":
            print(f"{row['model']} | {row['source']}: accuracy={row['accuracy']} "
                  f"n={row['n_success']}/{row['n']} errors={row['n_errors']}")
    print(f"Saved: {args.output}/{{predictions.csv,summary.csv,repairs.csv,run_config.json}}")
    print(f"Original unchanged: {args.input}")


if __name__ == "__main__":
    main()
