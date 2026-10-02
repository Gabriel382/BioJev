from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

VARIANTS = {
    "general_only": (False, ["general-nli"]),
    "biomedical_only": (False, ["biomedical"]),
    "dapt_general": (True, ["general-nli"]),
    "dapt_biomedical": (True, ["biomedical"]),
    "general_biomedical": (False, ["general-nli", "biomedical"]),
    "full": (True, ["general-nli", "biomedical"]),
    "bionli_only": (True, ["general-nli", "biomedical"]),
    "nli4ct_only": (True, ["general-nli", "biomedical"]),
}


def _cfg(name):
    return yaml.safe_load((ROOT / f"configs/sprint6/train/{name}.yaml").read_text())


def test_all_variants_exist_and_match_factorization():
    for name, (has_dapt, stages) in VARIANTS.items():
        cfg = _cfg(name)
        assert bool(cfg["model"].get("dapt_adapter")) is has_dapt
        assert [x["name"] for x in cfg["stages"]] == stages
        assert cfg["model"]["base_model"] == "Qwen/Qwen3.5-0.8B-Base"
        assert cfg["seed"] == 42


def test_common_training_hyperparameters_are_identical():
    keys = [
        "method", "mixed_precision", "per_device_batch_size", "eval_batch_size",
        "gradient_accumulation_steps", "learning_rate", "weight_decay",
        "warmup_ratio", "gradient_checkpointing",
    ]
    ref = _cfg("full")["training"]
    for name in VARIANTS:
        cur = _cfg(name)["training"]
        assert {k: cur[k] for k in keys} == {k: ref[k] for k in keys}


def test_source_isolations_keep_40k_budget():
    b = _cfg("bionli_only")["stages"][1]
    n = _cfg("nli4ct_only")["stages"][1]
    assert b["epoch_examples"] == n["epoch_examples"] == 40000
    assert b["datasets"] == [{"name": "bionli", "weight": 1.0}]
    assert n["datasets"] == [{"name": "nli4ct", "weight": 1.0}]


def test_model_configs_point_to_expected_final_stage():
    for name in VARIANTS:
        cfg = yaml.safe_load((ROOT / f"configs/sprint6/models/{name}.yaml").read_text())
        ckpt = cfg["params"]["checkpoint"]
        if name in {"general_biomedical","full","bionli_only","nli4ct_only"}:
            assert "/stage-02-biomedical/final" in ckpt
        elif name in {"general_only","dapt_general"}:
            assert "/stage-01-general-nli/final" in ckpt
        else:
            assert "/stage-01-biomedical/final" in ckpt
