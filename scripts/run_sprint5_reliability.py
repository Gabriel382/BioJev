#!/usr/bin/env python3
"""Sprint 5: calibration, reliability and selective prediction analysis.

No model training occurs here. The script consumes per-example prediction
artifacts produced by Sprint 4, validates them, computes reliability metrics,
and writes source-fingerprinted outputs so --skip-existing is safe.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score

KNOWN_DATASETS = ("bionli", "nli4ct", "chemprot", "ddi2013", "biored")

GOLD_KEYS = ("gold", "gold_label", "true_label", "y_true", "target", "label", "labels")
PRED_KEYS = ("pred", "pred_label", "prediction", "predicted_label", "y_pred")
CONF_KEYS = ("confidence", "conf", "max_prob", "max_probability", "probability")
PROB_KEYS = ("probabilities", "probs", "softmax", "class_probabilities")
LOGIT_KEYS = ("logits",)
ID_KEYS = ("id", "example_id", "uid", "guid", "pair_id", "sample_id")


@dataclass(frozen=True)
class RunSpec:
    regime: str
    model: str
    dataset: str
    seed: int | None
    source: Path

    @property
    def run_id(self) -> str:
        if self.seed is not None:
            return f"seed{self.seed}"
        # frozen runs are normally one run/model/dataset. Preserve a stable suffix
        # only if the source parent adds useful identity.
        return "frozen"

    @property
    def key(self) -> tuple[str, str, str, int | None]:
        return (self.regime, self.model, self.dataset, self.seed)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _first_existing(df: pd.DataFrame, keys: Iterable[str]) -> str | None:
    low = {str(c).lower(): c for c in df.columns}
    for k in keys:
        if k in low:
            return low[k]
    return None


def _parse_obj(x: Any) -> Any:
    if isinstance(x, (list, dict, tuple, np.ndarray)):
        return x
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return x
        if s[0] in "[{(" and s[-1] in "]})":
            try:
                return json.loads(s)
            except Exception:
                try:
                    return ast.literal_eval(s)
                except Exception:
                    return x
    return x


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


def load_prediction_file(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception as e:
                    raise ValueError(f"invalid JSONL at line {i}: {e}") from e
        return pd.DataFrame(rows)
    if suf == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for k in ("predictions", "rows", "examples", "data"):
                if isinstance(data.get(k), list):
                    data = data[k]
                    break
        if not isinstance(data, list):
            raise ValueError("JSON prediction file must contain a list or a predictions/rows/examples/data list")
        return pd.DataFrame(data)
    if suf == ".csv":
        return pd.read_csv(path)
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf == ".npz":
        z = np.load(path, allow_pickle=True)
        return pd.DataFrame({k: list(z[k]) if z[k].ndim > 1 else z[k] for k in z.files})
    raise ValueError(f"unsupported prediction format: {path}")


def normalize_predictions(raw: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray | None, list[Any] | None, list[str]]:
    warnings: list[str] = []
    if len(raw) == 0:
        raise ValueError("prediction file is empty")

    gold_col = _first_existing(raw, GOLD_KEYS)
    pred_col = _first_existing(raw, PRED_KEYS)
    conf_col = _first_existing(raw, CONF_KEYS)
    prob_col = _first_existing(raw, PROB_KEYS)
    logit_col = _first_existing(raw, LOGIT_KEYS)
    id_col = _first_existing(raw, ID_KEYS)

    if gold_col is None:
        raise ValueError(f"no gold-label column found; expected one of {GOLD_KEYS}")

    probs: np.ndarray | None = None
    class_names: list[Any] | None = None

    if prob_col is not None:
        vals = [_parse_obj(x) for x in raw[prob_col].tolist()]
        if vals and all(isinstance(v, dict) for v in vals):
            # Use first-row key order, then append any later keys deterministically.
            keys = list(vals[0].keys())
            for v in vals[1:]:
                for k in v:
                    if k not in keys:
                        keys.append(k)
            class_names = keys
            probs = np.asarray([[float(v.get(k, 0.0)) for k in keys] for v in vals], dtype=float)
        else:
            try:
                probs = np.asarray([np.asarray(v, dtype=float) for v in vals], dtype=float)
            except Exception as e:
                raise ValueError(f"could not parse probability column {prob_col}: {e}") from e
    elif logit_col is not None:
        vals = [_parse_obj(x) for x in raw[logit_col].tolist()]
        try:
            logits = np.asarray([np.asarray(v, dtype=float) for v in vals], dtype=float)
        except Exception as e:
            raise ValueError(f"could not parse logits column {logit_col}: {e}") from e
        probs = _softmax(logits)

    if probs is not None:
        if probs.ndim != 2 or probs.shape[0] != len(raw):
            raise ValueError(f"probability matrix has invalid shape {probs.shape}; expected ({len(raw)}, K)")
        row_sum = probs.sum(axis=1)
        if np.any(~np.isfinite(probs)):
            raise ValueError("probabilities contain NaN/Inf")
        # If scores are nonnegative but not exactly normalized, normalize safely.
        if np.all(probs >= 0) and np.all(row_sum > 0) and not np.allclose(row_sum, 1.0, atol=1e-4):
            probs = probs / row_sum[:, None]
            warnings.append("probabilities were renormalized to sum to 1")

    gold = raw[gold_col].map(_parse_obj).to_numpy()

    if pred_col is not None:
        pred = raw[pred_col].map(_parse_obj).to_numpy()
    elif probs is not None:
        idx = np.argmax(probs, axis=1)
        pred = np.asarray([class_names[i] for i in idx], dtype=object) if class_names is not None else idx
    else:
        raise ValueError(f"no predicted-label column found and no probabilities/logits available; expected one of {PRED_KEYS}")

    if conf_col is not None:
        conf = pd.to_numeric(raw[conf_col], errors="coerce").to_numpy(dtype=float)
    elif probs is not None:
        conf = np.max(probs, axis=1)
    else:
        raise ValueError(f"no confidence column and no probabilities/logits available; expected one of {CONF_KEYS}")

    if np.any(~np.isfinite(conf)):
        raise ValueError("confidence contains NaN/Inf")
    if np.any((conf < -1e-8) | (conf > 1 + 1e-8)):
        raise ValueError("confidence must be in [0,1]")
    conf = np.clip(conf, 0.0, 1.0)

    norm = pd.DataFrame({
        "example_id": raw[id_col].astype(str) if id_col is not None else [str(i) for i in range(len(raw))],
        "gold": gold,
        "pred": pred,
        "confidence": conf,
    })
    norm["correct"] = norm["gold"].astype(str).to_numpy() == norm["pred"].astype(str).to_numpy()
    return norm, probs, class_names, warnings


def fixed_reliability(conf: np.ndarray, correct: np.ndarray, n_bins: int) -> tuple[pd.DataFrame, float]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    n = len(conf)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf >= lo) & (conf <= hi if i == n_bins - 1 else conf < hi)
        k = int(mask.sum())
        if k:
            mc = float(conf[mask].mean())
            acc = float(correct[mask].mean())
            gap = abs(acc - mc)
            ece += (k / n) * gap
        else:
            mc = acc = gap = float("nan")
        rows.append({"bin": i, "lower": lo, "upper": hi, "n": k, "coverage": k / n, "mean_confidence": mc, "accuracy": acc, "gap": gap})
    return pd.DataFrame(rows), float(ece)


def adaptive_reliability(conf: np.ndarray, correct: np.ndarray, n_bins: int) -> tuple[pd.DataFrame, float]:
    order = np.argsort(conf)
    chunks = np.array_split(order, min(n_bins, len(order)))
    rows = []
    n = len(conf)
    ece = 0.0
    for i, idx in enumerate(chunks):
        if len(idx) == 0:
            continue
        mc = float(conf[idx].mean())
        acc = float(correct[idx].mean())
        gap = abs(acc - mc)
        ece += len(idx) / n * gap
        rows.append({"bin": i, "lower": float(conf[idx].min()), "upper": float(conf[idx].max()), "n": len(idx), "coverage": len(idx) / n, "mean_confidence": mc, "accuracy": acc, "gap": gap})
    return pd.DataFrame(rows), float(ece)


def _gold_indices(gold: np.ndarray, probs: np.ndarray, class_names: list[Any] | None) -> np.ndarray | None:
    k = probs.shape[1]
    if class_names is not None:
        mapping = {str(v): i for i, v in enumerate(class_names)}
        idx = np.asarray([mapping.get(str(v), -1) for v in gold], dtype=int)
        return idx if np.all(idx >= 0) else None
    out = []
    for v in gold:
        try:
            i = int(v)
        except Exception:
            return None
        if i < 0 or i >= k:
            return None
        out.append(i)
    return np.asarray(out, dtype=int)


def probabilistic_scores(gold: np.ndarray, probs: np.ndarray | None, class_names: list[Any] | None) -> tuple[float, float, str | None]:
    if probs is None:
        return float("nan"), float("nan"), "Brier/NLL unavailable: no full probability vector"
    idx = _gold_indices(gold, probs, class_names)
    if idx is None:
        return float("nan"), float("nan"), "Brier/NLL unavailable: could not map gold labels to probability columns"
    p = np.clip(probs, 1e-12, 1.0)
    nll = float(-np.log(p[np.arange(len(idx)), idx]).mean())
    y = np.zeros_like(probs, dtype=float)
    y[np.arange(len(idx)), idx] = 1.0
    brier = float(np.mean(np.sum((probs - y) ** 2, axis=1)))
    return brier, nll, None


def risk_coverage(conf: np.ndarray, correct: np.ndarray, points: int) -> tuple[pd.DataFrame, float]:
    order = np.argsort(-conf, kind="stable")
    errors = (~correct[order]).astype(float)
    cum_risk = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    exact_aurc = float(cum_risk.mean())
    grid = np.linspace(1 / len(errors), 1.0, min(points, len(errors)))
    idx = np.maximum(0, np.ceil(grid * len(errors)).astype(int) - 1)
    rows = pd.DataFrame({
        "coverage": (idx + 1) / len(errors),
        "risk": cum_risk[idx],
        "accuracy": 1.0 - cum_risk[idx],
        "confidence_cutoff": conf[order][idx],
        "accepted_n": idx + 1,
    }).drop_duplicates(subset=["accepted_n"]).reset_index(drop=True)
    return rows, exact_aurc


def threshold_table(norm: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        s = norm[norm.confidence >= t]
        if len(s):
            acc = float(s.correct.mean())
            macro = float(f1_score(s.gold.astype(str), s.pred.astype(str), average="macro", zero_division=0))
            mean_conf = float(s.confidence.mean())
        else:
            acc = macro = mean_conf = float("nan")
        rows.append({
            "threshold": float(t), "accepted_n": len(s), "coverage": len(s) / len(norm),
            "accuracy": acc, "risk": (1.0 - acc) if np.isfinite(acc) else float("nan"),
            "f1_macro": macro, "mean_confidence": mean_conf,
        })
    return pd.DataFrame(rows)


def slice_table(norm: pd.DataFrame, min_n: int) -> pd.DataFrame:
    rows = []
    for label, s in norm.groupby(norm.gold.astype(str), sort=True):
        if len(s) < min_n:
            continue
        _, ece = fixed_reliability(s.confidence.to_numpy(), s.correct.to_numpy(bool), min(10, max(2, int(np.sqrt(len(s))))))
        rows.append({
            "slice_type": "gold_label", "slice_value": label, "n": len(s),
            "accuracy": float(s.correct.mean()), "mean_confidence": float(s.confidence.mean()), "ece": ece,
        })
    return pd.DataFrame(rows)


def infer_model(path: Path, aliases: dict[str, list[str]], allowed: list[str]) -> str | None:
    s = str(path).lower()
    # Prefer longer aliases first so biojev is not swallowed by qwen paths in metadata-like names.
    for model in allowed:
        for a in sorted(aliases.get(model, [model]), key=len, reverse=True):
            if a.lower() in s:
                return model
    return None


def infer_dataset(path: Path, allowed: list[str]) -> str | None:
    s = str(path).lower()
    for d in allowed:
        if re.search(rf"(^|[^a-z0-9]){re.escape(d)}([^a-z0-9]|$)", s):
            return d
    return None


def infer_seed(path: Path) -> int | None:
    m = re.search(r"seed[_-]?(\d+)", str(path).lower())
    return int(m.group(1)) if m else None


def discover_for_regime(name: str, cfg: dict[str, Any], root: Path) -> tuple[list[RunSpec], list[dict[str, Any]]]:
    rcfg = cfg["regimes"][name]
    filenames = set(cfg["prediction_filenames"])
    aliases = cfg.get("model_aliases", {})
    candidates: list[Path] = []
    for r in rcfg["roots"]:
        p = root / r
        if not p.exists():
            continue
        for f in p.rglob("*"):
            if f.is_file() and f.name in filenames:
                candidates.append(f)

    grouped: dict[tuple[str, str, str, int | None], list[Path]] = {}
    stray: list[dict[str, Any]] = []
    for p in sorted(set(candidates)):
        model = infer_model(p, aliases, rcfg["models"])
        dataset = infer_dataset(p, rcfg["datasets"])
        seed = infer_seed(p)
        if rcfg.get("seeds") and seed not in set(int(x) for x in rcfg["seeds"]):
            seed = None
        if model is None or dataset is None:
            stray.append({"regime": name, "status": "unclassified", "source": str(p), "model": model, "dataset": dataset, "seed": seed, "message": "could not infer model/dataset from path"})
            continue
        spec = RunSpec(name, model, dataset, seed, p)
        grouped.setdefault(spec.key, []).append(p)

    priority = {fn: i for i, fn in enumerate(cfg["prediction_filenames"])}
    runs = []
    for key, paths in grouped.items():
        paths = sorted(paths, key=lambda p: (priority.get(p.name, 999), len(str(p))))
        p = paths[0]
        regime, model, dataset, seed = key
        runs.append(RunSpec(regime, model, dataset, seed, p))
        if len(paths) > 1:
            stray.append({"regime": name, "status": "duplicate", "source": " | ".join(map(str, paths)), "model": model, "dataset": dataset, "seed": seed, "message": f"using {p}"})
    return sorted(runs, key=lambda x: (x.model, x.dataset, -1 if x.seed is None else x.seed)), stray


def expected_keys(name: str, cfg: dict[str, Any]) -> list[tuple[str, str, str, int | None]]:
    r = cfg["regimes"][name]
    seeds = r.get("seeds") or [None]
    return [(name, m, d, int(s) if s is not None else None) for m in r["models"] for d in r["datasets"] for s in seeds]


def audit_runs(regimes: list[str], cfg: dict[str, Any], root: Path) -> tuple[list[RunSpec], pd.DataFrame]:
    all_runs: list[RunSpec] = []
    rows: list[dict[str, Any]] = []
    bykey: dict[tuple[str, str, str, int | None], RunSpec] = {}
    extras: list[dict[str, Any]] = []
    for reg in regimes:
        runs, ex = discover_for_regime(reg, cfg, root)
        extras.extend(ex)
        for r in runs:
            bykey[r.key] = r
            all_runs.append(r)

    for reg in regimes:
        for key in expected_keys(reg, cfg):
            run = bykey.get(key)
            if run is None:
                rows.append({"regime": reg, "model": key[1], "dataset": key[2], "seed": key[3], "status": "missing", "source": "", "n": "", "message": "no prediction artifact found"})
                continue
            try:
                raw = load_prediction_file(run.source)
                norm, probs, _, warns = normalize_predictions(raw)
                msg = "; ".join(warns)
                if probs is None:
                    msg = (msg + "; " if msg else "") + "no full probability vector: ECE/risk-coverage work, Brier/NLL will be unavailable"
                rows.append({"regime": reg, "model": run.model, "dataset": run.dataset, "seed": run.seed, "status": "ready", "source": str(run.source), "n": len(norm), "message": msg})
            except Exception as e:
                rows.append({"regime": reg, "model": run.model, "dataset": run.dataset, "seed": run.seed, "status": "invalid", "source": str(run.source), "n": "", "message": str(e)})
    rows.extend(extras)
    return all_runs, pd.DataFrame(rows)


def analyze_run(run: RunSpec, cfg: dict[str, Any], root: Path, out_root: Path, skip_existing: bool) -> dict[str, Any]:
    acfg = cfg["analysis"]
    source_sha = sha256_file(run.source)
    analysis_sig = stable_hash(acfg)
    run_dir = out_root / "runs" / run.model / run.dataset / run.run_id
    summary_path = run_dir / "summary.json"

    if skip_existing and summary_path.exists():
        old = json.loads(summary_path.read_text(encoding="utf-8"))
        if old.get("source_sha256") == source_sha and old.get("analysis_signature") == analysis_sig:
            old["_skipped"] = True
            return old

    raw = load_prediction_file(run.source)
    norm, probs, class_names, warnings = normalize_predictions(raw)
    correct = norm.correct.to_numpy(bool)
    conf = norm.confidence.to_numpy(float)

    fixed_bins, ece = fixed_reliability(conf, correct, int(acfg["fixed_bins"]))
    adaptive_bins, adaptive_ece = adaptive_reliability(conf, correct, int(acfg["adaptive_bins"]))
    rc, aurc = risk_coverage(conf, correct, int(acfg["risk_coverage_points"]))
    th = threshold_table(norm, [float(x) for x in acfg["thresholds"]])
    slices = slice_table(norm, int(acfg["min_slice_size"]))
    brier, nll, prob_warning = probabilistic_scores(norm.gold.to_numpy(), probs, class_names)
    if prob_warning:
        warnings.append(prob_warning)

    hc_t = float(acfg["high_confidence_threshold"])
    hc = norm[(norm.confidence >= hc_t) & (~norm.correct)].copy().sort_values("confidence", ascending=False)

    summary = {
        "regime": run.regime,
        "model": run.model,
        "dataset": run.dataset,
        "seed": run.seed,
        "run_id": run.run_id,
        "source_file": str(run.source),
        "source_sha256": source_sha,
        "analysis_signature": analysis_sig,
        "n": int(len(norm)),
        "accuracy": float(accuracy_score(norm.gold.astype(str), norm.pred.astype(str))),
        "f1_macro": float(f1_score(norm.gold.astype(str), norm.pred.astype(str), average="macro", zero_division=0)),
        "f1_micro": float(f1_score(norm.gold.astype(str), norm.pred.astype(str), average="micro", zero_division=0)),
        "ece": ece,
        "adaptive_ece": adaptive_ece,
        "brier": brier,
        "nll": nll,
        "mean_confidence": float(conf.mean()),
        "aurc": aurc,
        "high_confidence_threshold": hc_t,
        "high_confidence_errors": int(len(hc)),
        "high_confidence_error_rate": float(len(hc) / len(norm)),
        "warnings": warnings,
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    fixed_bins.to_csv(run_dir / "reliability_bins.csv", index=False)
    adaptive_bins.to_csv(run_dir / "adaptive_bins.csv", index=False)
    rc.to_csv(run_dir / "risk_coverage.csv", index=False)
    th.to_csv(run_dir / "thresholds.csv", index=False)
    slices.to_csv(run_dir / "slices.csv", index=False)
    hc.to_csv(run_dir / "high_confidence_errors.csv", index=False)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    return summary


def aggregate_outputs(summaries: list[dict[str, Any]], out_root: Path) -> None:
    clean = [{k: v for k, v in s.items() if k != "_skipped"} for s in summaries]
    sdf = pd.DataFrame(clean)
    out_root.mkdir(parents=True, exist_ok=True)
    sdf.to_csv(out_root / "summary.csv", index=False)

    metric_cols = ["accuracy", "f1_macro", "f1_micro", "ece", "adaptive_ece", "brier", "nll", "mean_confidence", "aurc", "high_confidence_error_rate"]
    if len(sdf):
        g = sdf.groupby(["regime", "model", "dataset"], dropna=False)[metric_cols].agg(["mean", "std", "count"])
        g.columns = [f"{a}_{b}" for a, b in g.columns]
        g.reset_index().to_csv(out_root / "summary_by_model_dataset.csv", index=False)

    threshold_frames = []
    slice_frames = []
    for s in clean:
        d = out_root / "runs" / s["model"] / s["dataset"] / s["run_id"]
        for fname, bucket in (("thresholds.csv", threshold_frames), ("slices.csv", slice_frames)):
            p = d / fname
            if p.exists():
                x = pd.read_csv(p)
                x.insert(0, "regime", s["regime"])
                x.insert(1, "model", s["model"])
                x.insert(2, "dataset", s["dataset"])
                x.insert(3, "seed", s.get("seed"))
                bucket.append(x)
    pd.concat(threshold_frames, ignore_index=True).to_csv(out_root / "thresholds_long.csv", index=False) if threshold_frames else pd.DataFrame().to_csv(out_root / "thresholds_long.csv", index=False)
    pd.concat(slice_frames, ignore_index=True).to_csv(out_root / "slices_long.csv", index=False) if slice_frames else pd.DataFrame().to_csv(out_root / "slices_long.csv", index=False)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sprint5/reliability.yaml")
    ap.add_argument("--regime", choices=["all", "frozen_task_general", "supervised_baselines"], default="all")
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    root = Path.cwd()
    cfg = yaml.safe_load((root / args.config).read_text(encoding="utf-8"))
    regimes = list(cfg["regimes"]) if args.regime == "all" else [args.regime]
    out_root = root / cfg.get("output_dir", "results/sprint5")

    runs, audit = audit_runs(regimes, cfg, root)
    out_root.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out_root / "input_audit.csv", index=False)
    counts = audit[audit.status.isin(["ready", "missing", "invalid"])].status.value_counts().to_dict() if len(audit) else {}
    print("Sprint 5 input audit:", " | ".join(f"{k}={counts.get(k, 0)}" for k in ("ready", "missing", "invalid")))
    bad = audit[audit.status.isin(["missing", "invalid"])]
    if len(bad):
        print(bad[["regime", "model", "dataset", "seed", "status", "source", "message"]].to_string(index=False))
    print(f"Wrote {out_root / 'input_audit.csv'}")

    if args.audit_only:
        return 0
    if cfg.get("strict_inputs", True) and len(bad):
        print("\nERROR: Sprint 5 strict input audit failed. No reliability analysis was run.", file=sys.stderr)
        print("Use --audit-only to inspect the missing/invalid prediction artifacts. This does not require retraining.", file=sys.stderr)
        return 2

    ready_keys = set(tuple(x) for x in audit[audit.status.eq("ready")][["regime", "model", "dataset", "seed"]].itertuples(index=False, name=None))
    selected = [r for r in runs if (r.regime, r.model, r.dataset, r.seed) in ready_keys]
    summaries = []
    for i, run in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {run.regime} | {run.model} | {run.dataset} | seed={run.seed} | {run.source}")
        s = analyze_run(run, cfg, root, out_root, args.skip_existing)
        print("  skipped (source unchanged)" if s.pop("_skipped", False) else f"  n={s['n']} acc={s['accuracy']:.4f} ece={s['ece']:.4f} aurc={s['aurc']:.4f}")
        summaries.append(s)

    aggregate_outputs(summaries, out_root)
    print(f"Sprint 5 reliability: {len(summaries)} runs analyzed")
    print(f"Wrote {out_root / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
