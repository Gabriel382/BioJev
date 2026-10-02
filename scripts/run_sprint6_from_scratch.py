#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from biojev.config import load_yaml


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def dapt_complete(path: Path) -> bool:
    manifest = path / "training_manifest.json"
    if not manifest.exists():
        return False
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return False
    return data.get("final_checkpoint") in {None, str(path)} or path.exists()


def latest_checkpoint(output_dir: Path) -> Path | None:
    cps = []
    for p in output_dir.glob("checkpoint-*"):
        try:
            step = int(p.name.split("-")[-1])
        except ValueError:
            continue
        cps.append((step, p))
    return max(cps, default=(None, None))[1]


def selected_need_dapt(plan: dict, selected: set[str]) -> bool:
    for spec in plan["variants"]:
        if selected and spec["name"] not in selected:
            continue
        if spec.get("dapt"):
            return True
    return False


def main():
    p = argparse.ArgumentParser(
        description="Bootstrap BioJev Sprint 6 from base Qwen through Nano DAPT and ablation training."
    )
    p.add_argument("--plan", default="configs/sprint6/nano_ablation_plan.yaml")
    p.add_argument("--dapt-config", default="configs/sprint6/dapt_nano_50k.yaml")
    p.add_argument("--variant", action="append", default=[], help="Train only selected ablation variant(s).")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument(
        "--through",
        choices=["dapt", "train", "eval", "tables"],
        default="train",
        help="How far to run the pipeline. Default stops after ablation training.",
    )
    p.add_argument("--continue-on-error", action="store_true")
    args = p.parse_args()

    plan = load_yaml(args.plan)
    selected = set(args.variant)
    known = {x["name"] for x in plan["variants"]}
    unknown = selected - known
    if unknown:
        raise SystemExit(f"Unknown variant(s): {sorted(unknown)}")

    dapt_cfg = load_yaml(args.dapt_config)
    dapt_output = Path(dapt_cfg.get("output_dir", "outputs/biojev-nano"))
    dapt_final = dapt_output / "final"
    need_dapt = selected_need_dapt(plan, selected)

    print("Sprint 6 from-scratch pipeline")
    print(f"  selected variants: {', '.join(sorted(selected)) if selected else 'all'}")
    print(f"  DAPT required:     {need_dapt}")
    print(f"  DAPT final:        {dapt_final}")
    print(f"  through:           {args.through}")

    if need_dapt:
        if dapt_complete(dapt_final):
            print(f"\n[DAPT] reuse existing {dapt_final}")
        else:
            cmd = [
                sys.executable,
                "scripts/train_dapt.py",
                "--config",
                args.dapt_config,
            ]
            cp = latest_checkpoint(dapt_output)
            if cp is not None:
                print(f"\n[DAPT] incomplete prior run detected; resuming from {cp}")
                cmd += ["--resume", str(cp)]
            else:
                print("\n[DAPT] no completed adapter found; training Nano DAPT from base Qwen.")
            run(cmd)
            if not dapt_complete(dapt_final):
                raise SystemExit(f"DAPT command returned but final adapter is not complete: {dapt_final}")
    else:
        print("\n[DAPT] selected variants do not require DAPT; bootstrap skipped.")

    if args.through == "dapt":
        return

    train_cmd = [sys.executable, "scripts/run_sprint6_train.py", "--plan", args.plan]
    for name in args.variant:
        train_cmd += ["--variant", name]
    if args.skip_existing:
        train_cmd.append("--skip-existing")
    if args.continue_on_error:
        train_cmd.append("--continue-on-error")
    run(train_cmd)

    if args.through == "train":
        return

    eval_cmd = [
        sys.executable,
        "scripts/run_sprint6_eval.py",
        "--config",
        "configs/sprint6/eval_nano.yaml",
    ]
    if args.skip_existing:
        eval_cmd.append("--skip-existing")
    run(eval_cmd)

    if args.through == "eval":
        return

    run([sys.executable, "scripts/build_sprint6_tables.py"])


if __name__ == "__main__":
    main()
