#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PAIR_EFFECTS = [
    ("dapt_effect_full_path", "full", "general_biomedical"),
    ("dapt_effect_after_general", "dapt_general", "general_only"),
    ("dapt_effect_after_biomedical", "dapt_biomedical", "biomedical_only"),
    ("biomedical_stage_effect_after_dapt_general", "full", "dapt_general"),
    ("general_stage_effect_after_dapt", "full", "dapt_biomedical"),
    ("mixed_biomed_vs_bionli_only", "full", "bionli_only"),
    ("mixed_biomed_vs_nli4ct_only", "full", "nli4ct_only"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default="results/sprint6/nano_ablation_eval/summary.csv")
    p.add_argument("--out", default="results/sprint6/tables")
    args = p.parse_args()

    df = pd.read_csv(args.summary)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    metric = "f1_macro"
    pivot = df.pivot(index="variant", columns="dataset", values=metric)
    dataset_order = [x for x in ["bionli", "nli4ct", "chemprot", "ddi2013", "biored"] if x in pivot.columns]
    pivot = pivot.reindex(columns=dataset_order)
    pivot["relation_transfer_mean"] = pivot[[x for x in ["chemprot","ddi2013","biored"] if x in pivot.columns]].mean(axis=1)
    pivot["all5_mean"] = pivot[dataset_order].mean(axis=1)

    pct = (100 * pivot).round(2)
    pct.to_csv(out / "ablation_macro_f1.csv")
    (out / "ablation_macro_f1.md").write_text(pct.to_markdown(), encoding="utf-8")

    effects = []
    for effect_name, lhs, rhs in PAIR_EFFECTS:
        if lhs not in pivot.index or rhs not in pivot.index:
            continue
        for dataset in dataset_order + ["relation_transfer_mean", "all5_mean"]:
            effects.append({
                "effect": effect_name,
                "comparison": f"{lhs} - {rhs}",
                "dataset": dataset,
                "delta_f1_macro_pp": round(100 * (pivot.loc[lhs, dataset] - pivot.loc[rhs, dataset]), 3),
            })
    effects_df = pd.DataFrame(effects)
    effects_df.to_csv(out / "ablation_effects.csv", index=False)
    (out / "ablation_effects.md").write_text(effects_df.to_markdown(index=False), encoding="utf-8")

    seen = df[["variant","dataset","seen_status","dapt","general_nli","biomedical_nli","biomedical_source"]].drop_duplicates()
    seen.to_csv(out / "training_provenance.csv", index=False)

    print(f"Wrote Sprint 6 tables to {out}")


if __name__ == "__main__":
    main()
