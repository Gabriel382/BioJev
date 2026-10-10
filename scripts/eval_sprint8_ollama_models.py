#!/usr/bin/env python3
"""Compare native Ollama System One accuracy/latency on a fixed stratified dev sample.

Run from the BioJev repository root. Requires only the Python standard library.
"""
import argparse
import csv
import json
import math
import random
import re
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

DEFAULT_MODELS = ["biojev-systemone:0.8b", "biojev-systemone:4b", "biojev-systemone:9b"]


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def select_examples(rows, per_source, seed):
    by_source = defaultdict(list)
    for row in rows:
        if not isinstance(row.get("options"), list) or not 2 <= len(row["options"]) <= 26:
            continue
        if not row.get("answer_key") or not row.get("question"):
            continue
        by_source[row.get("source", "unknown")].append(row)
    rng = random.Random(seed)
    chosen = []
    for source in sorted(by_source):
        group = by_source[source][:]
        rng.shuffle(group)
        chosen.extend(group[:per_source] if per_source > 0 else group)
    return chosen


# Ollama System One returns this exact family of messages for oversized inputs.
# The dash between 1 and the limit is a Unicode en dash in current Ollama builds.
OVERFLOW_RE = re.compile(
    r"\bprompt\s+\d+\s+has\s+(\d+)\s+tokens;\s*expected\s+1\s*[\u2013\u2014-]\s*(\d+)",
    flags=re.IGNORECASE,
)


def detect_context_overflow(status_code, detail):
    """Return (prompt_tokens, allowed_tokens) only for Ollama's known HTTP 400."""
    if status_code != 400:
        return None
    # Ollama sends JSON; serialized JSON may escape the Unicode en dash as \u2013.
    try:
        parsed = json.loads(detail)
        message = parsed.get("error", detail) if isinstance(parsed, dict) else detail
    except (ValueError, TypeError):
        message = detail
    match = OVERFLOW_RE.search(str(message))
    return (int(match.group(1)), int(match.group(2))) if match else None


def post_choice(base_url, payload, timeout):
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/systemone",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def query(base_url, model, row, timeout, keep_alive, *,
          adaptive_context=True, base_context=2048, extended_context=4096,
          extended_suffix="-4k"):
    criteria = {str(opt["key"]): str(opt.get("description") or opt["key"])
                for opt in row["options"]}
    payload = {
        "model": model,
        "state": row["state"],
        "questions": {
            "decision": {
                "type": "choice",
                "instructions": row["question"],
                "criteria": criteria,
            }
        },
        "keep_alive": keep_alive,
    }
    start = time.perf_counter()
    selected_model = model
    context_limit = base_context
    attempts = 0
    overflow_tokens = ""

    # At most two calls: default model first, extended variant only on overflow.
    while True:
        attempts += 1
        try:
            result = post_choice(base_url, payload, timeout)
            decision = result["answers"]["decision"]
            if decision["type"] != "choice":
                raise ValueError(f"Expected choice, received {decision.get('type')}")
            prediction = str(decision["choice"])
            probs = decision.get("probabilities", {})
            return {
                "prediction": prediction,
                "confidence": decision.get("confidence"),
                "gold_probability": probs.get(row["answer_key"]),
                "elapsed_seconds": time.perf_counter() - start,
                "error": "",
                "model_used": selected_model,
                "context_limit_tokens": context_limit,
                "attempts": attempts,
                "overflow_tokens": overflow_tokens,
            }
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", "replace")[:2000]
            except Exception:
                detail = str(exc)
            overflow = detect_context_overflow(exc.code, detail)
            if (adaptive_context and attempts == 1 and overflow
                    and overflow[0] <= extended_context
                    and overflow[0] > overflow[1]):
                overflow_tokens = overflow[0]
                selected_model = model + extended_suffix
                context_limit = extended_context
                payload["model"] = selected_model
                continue
            error = f"HTTP {exc.code}: {detail}"
            if overflow and overflow[0] > extended_context and attempts == 1:
                error += f" [requires > {extended_context} context tokens; not retried]"
            if attempts == 2 and exc.code == 404:
                error += (f" [missing {selected_model}? run "
                          "scripts/create_sprint8_ollama_context_variants.py]")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        return {
            "prediction": "", "confidence": "", "gold_probability": "",
            "elapsed_seconds": time.perf_counter() - start,
            "error": error,
            "model_used": selected_model,
            "context_limit_tokens": context_limit,
            "attempts": attempts,
            "overflow_tokens": overflow_tokens,
        }


def percentile(values, q):
    if not values:
        return float("nan")
    values = sorted(values)
    pos = (len(values) - 1) * q
    i = math.floor(pos)
    j = math.ceil(pos)
    return values[i] + (values[j] - values[i]) * (pos - i)


def summary(model, source, results):
    successes = [r for r in results if not r["error"]]
    durations = [float(r["elapsed_seconds"]) for r in successes]
    correct = sum(r["prediction"] == r["gold_key"] for r in successes)
    return {
        "model": model,
        "source": source,
        "n": len(results),
        "n_success": len(successes),
        "n_errors": len(results) - len(successes),
        "n_extended": sum(r["attempts"] == 2 and not r["error"] for r in results),
        "accuracy": round(correct / len(successes), 6) if successes else "",
        "mean_latency_s": round(statistics.mean(durations), 4) if durations else "",
        "median_latency_s": round(statistics.median(durations), 4) if durations else "",
        "p95_latency_s": round(percentile(durations, 0.95), 4) if durations else "",
    }


def save_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/sprint8/systemone_nano/dev.jsonl"))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--per-source", type=int, default=5,
                        help="Examples per source (default 5; use 0 for complete dev split)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--keep-alive", default="15m")
    parser.add_argument("--output", type=Path, default=Path("results/sprint8/native_ollama_eval"))
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--no-stop", action="store_true", help="Do not call ollama stop between models")
    parser.add_argument("--no-adaptive-context", action="store_true",
                        help="Disable retry on a dedicated extended-context model")
    parser.add_argument("--base-context", type=int, default=2048)
    parser.add_argument("--extended-context", type=int, default=4096)
    parser.add_argument("--extended-suffix", default="-4k")
    args = parser.parse_args()
    if args.base_context >= args.extended_context:
        parser.error("--extended-context must be greater than --base-context")

    def ask(model, example):
        return query(
            args.base_url, model, example, args.timeout, args.keep_alive,
            adaptive_context=not args.no_adaptive_context,
            base_context=args.base_context,
            extended_context=args.extended_context,
            extended_suffix=args.extended_suffix,
        )

    if not args.data.is_file():
        parser.error(f"dev data not found: {args.data}")
    data = select_examples(read_jsonl(args.data), args.per_source, args.seed)
    if not data:
        parser.error("No examples with choice options and answer_key found")
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Benchmark: {len(data)} identical examples/model from {len(set(x['source'] for x in data))} sources")
    print(f"Models: {', '.join(args.models)}", flush=True)

    predictions, aggregates = [], []
    for model in args.models:
        print(f"\n=== {model} ===", flush=True)
        if not args.skip_warmup:
            warm = ask(model, data[0])
            if warm["error"]:
                print(f"Warm-up error: {warm['error']}", flush=True)
            else:
                print("Warm-up OK (excluded from benchmark)", flush=True)
        model_rows = []
        for i, example in enumerate(data, 1):
            response = ask(model, example)
            row = {
                "model": model,
                "source": example.get("source", "unknown"),
                "id": example.get("id", ""),
                "gold_key": example["answer_key"],
                **response,
            }
            predictions.append(row)
            model_rows.append(row)
            if i % 10 == 0 or i == len(data):
                good = [r for r in model_rows if not r["error"]]
                accuracy = sum(r["prediction"] == r["gold_key"] for r in good) / len(good) if good else 0
                extended_n = sum(r["attempts"] == 2 and not r["error"] for r in model_rows)
                print(f"  {i}/{len(data)} | accuracy={accuracy:.3f} | "
                      f"errors={len(model_rows)-len(good)} | extended={extended_n}", flush=True)
        aggregates.append(summary(model, "ALL", model_rows))
        for source in sorted(set(x["source"] for x in model_rows)):
            aggregates.append(summary(model, source, [x for x in model_rows if x["source"] == source]))
        if not args.no_stop:
            # Release both model profiles to avoid memory accumulation between sizes.
            for loaded_model in (model, model + args.extended_suffix):
                subprocess.run(["ollama", "stop", loaded_model],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               check=False)

    save_csv(args.output / "predictions.csv", predictions,
             ["model", "source", "id", "gold_key", "prediction", "confidence",
              "gold_probability", "elapsed_seconds", "error", "model_used",
              "context_limit_tokens", "attempts", "overflow_tokens"])
    save_csv(args.output / "summary.csv", aggregates,
             ["model", "source", "n", "n_success", "n_errors", "n_extended",
              "accuracy", "mean_latency_s", "median_latency_s", "p95_latency_s"])
    (args.output / "run_config.json").write_text(json.dumps({
        "data": str(args.data), "models": args.models, "per_source": args.per_source,
        "seed": args.seed, "base_url": args.base_url,
        "adaptive_context": not args.no_adaptive_context,
        "base_context": args.base_context,
        "extended_context": args.extended_context,
        "extended_suffix": args.extended_suffix,
        "selected_ids": [x.get("id", "") for x in data],
    }, indent=2), encoding="utf-8")
    print("\nOVERALL RESULTS")
    for r in aggregates:
        if r["source"] == "ALL":
            print(f"{r['model']:<30} acc={r['accuracy']} n={r['n_success']}/{r['n']} "
                  f"extended={r['n_extended']} "
                  f"median_latency={r['median_latency_s']}s p95={r['p95_latency_s']}s")
    print(f"Saved: {args.output / 'summary.csv'}")
    print(f"Saved: {args.output / 'predictions.csv'}")


if __name__ == "__main__":
    main()
