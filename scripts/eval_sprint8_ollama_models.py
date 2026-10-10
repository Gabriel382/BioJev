#!/usr/bin/env python3
"""Compare native Ollama System One accuracy/latency on a fixed stratified dev sample.

Run from the BioJev repository root. Requires only the Python standard library.
"""
import argparse
import csv
import json
import math
import random
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


def query(base_url, model, row, timeout, keep_alive):
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
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/systemone",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
        elapsed = time.perf_counter() - start
        decision = result["answers"]["decision"]
        if decision["type"] != "choice":
            raise ValueError(f"Expected choice, received {decision.get('type')}")
        prediction = str(decision["choice"])
        probs = decision.get("probabilities", {})
        return {"prediction": prediction,
                "confidence": decision.get("confidence"),
                "gold_probability": probs.get(row["answer_key"]),
                "elapsed_seconds": elapsed,
                "error": ""}
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", "replace")[:500]
        except Exception:
            detail = str(exc)
        return {"prediction": "", "confidence": "", "gold_probability": "",
                "elapsed_seconds": round(time.perf_counter() - start, 4),
                "error": f"HTTP {exc.code}: {detail}"}
    except Exception as exc:
        return {"prediction": "", "confidence": "", "gold_probability": "",
                "elapsed_seconds": round(time.perf_counter() - start, 4),
                "error": f"{type(exc).__name__}: {exc}"}


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
    args = parser.parse_args()

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
            warm = query(args.base_url, model, data[0], args.timeout, args.keep_alive)
            if warm["error"]:
                print(f"Warm-up error: {warm['error']}", flush=True)
            else:
                print("Warm-up OK (excluded from benchmark)", flush=True)
        model_rows = []
        for i, example in enumerate(data, 1):
            response = query(args.base_url, model, example, args.timeout, args.keep_alive)
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
                print(f"  {i}/{len(data)} | accuracy={accuracy:.3f} | errors={len(model_rows)-len(good)}", flush=True)
        aggregates.append(summary(model, "ALL", model_rows))
        for source in sorted(set(x["source"] for x in model_rows)):
            aggregates.append(summary(model, source, [x for x in model_rows if x["source"] == source]))
        if not args.no_stop:
            subprocess.run(["ollama", "stop", model], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=False)

    save_csv(args.output / "predictions.csv", predictions,
             ["model", "source", "id", "gold_key", "prediction", "confidence",
              "gold_probability", "elapsed_seconds", "error"])
    save_csv(args.output / "summary.csv", aggregates,
             ["model", "source", "n", "n_success", "n_errors", "accuracy",
              "mean_latency_s", "median_latency_s", "p95_latency_s"])
    (args.output / "run_config.json").write_text(json.dumps({
        "data": str(args.data), "models": args.models, "per_source": args.per_source,
        "seed": args.seed, "base_url": args.base_url, "selected_ids": [x.get("id", "") for x in data],
    }, indent=2), encoding="utf-8")
    print("\nOVERALL RESULTS")
    for r in aggregates:
        if r["source"] == "ALL":
            print(f"{r['model']:<30} acc={r['accuracy']} n={r['n_success']}/{r['n']} "
                  f"median_latency={r['median_latency_s']}s p95={r['p95_latency_s']}s")
    print(f"Saved: {args.output / 'summary.csv'}")
    print(f"Saved: {args.output / 'predictions.csv'}")


if __name__ == "__main__":
    main()
