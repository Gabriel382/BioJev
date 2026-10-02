#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from biojev.config import load_yaml


def _final_checkpoint(output_dir: Path) -> Path | None:
    manifest = output_dir / "decision_manifest.json"
    if not manifest.exists():
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        p = Path(data["final_checkpoint"])
    except Exception:
        return None
    return p if p.exists() else None


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _stream(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        return proc.wait()


def audit(plan: dict, selected: set[str]) -> list[dict]:
    rows = []
    dapt = Path(plan["dapt_adapter"])
    for spec in plan["variants"]:
        name = spec["name"]
        if selected and name not in selected:
            continue
        cfg_path = Path(spec["config"])
        status = "ready"
        message = ""
        if not cfg_path.exists():
            status = "invalid"
            message = f"missing config: {cfg_path}"
        else:
            cfg = load_yaml(cfg_path)
            if spec.get("dapt") and not dapt.exists():
                status = "blocked"
                message = f"DAPT adapter missing: {dapt}"
            output_dir = Path(cfg["output_dir"])
            final = _final_checkpoint(output_dir)
            if final is not None:
                status = "complete"
                message = str(final)
        rows.append({
            "variant": name,
            "status": status,
            "config": str(cfg_path),
            "dapt": bool(spec.get("dapt")),
            "stages": ",".join(spec.get("stages", [])),
            "message": message,
        })
    return rows


def main():
    p = argparse.ArgumentParser(description="Sprint 6: run BioJev-Nano ablations.")
    p.add_argument("--plan", default="configs/sprint6/nano_ablation_plan.yaml")
    p.add_argument("--variant", action="append", default=[], help="Run only named variant(s).")
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    args = p.parse_args()

    plan = load_yaml(args.plan)
    selected = set(args.variant)
    known = {x["name"] for x in plan["variants"]}
    unknown = selected - known
    if unknown:
        raise SystemExit(f"Unknown variant(s): {sorted(unknown)}")

    rows = audit(plan, selected)
    print("\nSprint 6 Nano training audit")
    for row in rows:
        print(f"  {row['variant']:<20} {row['status']:<9} {row['message']}")
    _write_csv(Path(plan.get("results_root", "results/sprint6")) / "train_audit.csv", rows)

    blocked = [r for r in rows if r["status"] in {"invalid", "blocked"}]
    if args.audit_only:
        if blocked:
            raise SystemExit(2)
        return
    if blocked:
        raise SystemExit("Audit failed; fix blocked/invalid variants before training.")

    statuses = []
    for index, spec in enumerate(plan["variants"], start=1):
        name = spec["name"]
        if selected and name not in selected:
            continue
        cfg = load_yaml(spec["config"])
        output_dir = Path(cfg["output_dir"])
        final = _final_checkpoint(output_dir)
        if args.skip_existing and final is not None:
            print(f"\n[skip] {name}: {final}")
            statuses.append({"variant": name, "status": "skipped", "final_checkpoint": str(final)})
            continue

        cmd = [sys.executable, plan.get("train_script", "scripts/train_decision.py"),
               "--config", spec["config"]]
        print(f"\n[{index}/{len(plan['variants'])}] {name}")
        print("  " + " ".join(cmd))
        if args.dry_run:
            statuses.append({"variant": name, "status": "dry_run", "final_checkpoint": ""})
            continue

        code = _stream(cmd, Path(plan.get("results_root", "results/sprint6")) / "logs" / f"{name}.log")
        final = _final_checkpoint(output_dir)
        status = "complete" if code == 0 and final is not None else "failed"
        statuses.append({
            "variant": name,
            "status": status,
            "returncode": code,
            "final_checkpoint": str(final or ""),
        })
        _write_csv(Path(plan.get("results_root", "results/sprint6")) / "train_status.csv", statuses)
        if status == "failed" and not args.continue_on_error:
            raise SystemExit(f"{name} failed; see results/sprint6/logs/{name}.log")

    _write_csv(Path(plan.get("results_root", "results/sprint6")) / "train_status.csv", statuses)
    done = sum(r["status"] in {"complete", "skipped"} for r in statuses)
    print(f"\nSprint 6 Nano training: {done}/{len(statuses)} complete or reused.")


if __name__ == "__main__":
    main()
