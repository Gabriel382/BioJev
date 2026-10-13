"""Offline regression tests for targeted Sprint 8 NLI4CT recovery."""
import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import repair_sprint8_nli4ct_outliers as r

OUTLIERS = {
    "nli4ct:72": 4181, "nli4ct:44": 4573, "nli4ct:144": 4100,
    "nli4ct:151": 4572, "nli4ct:80": 4177, "nli4ct:197": 5044,
    "nli4ct:117": 4456, "nli4ct:175": 5225, "nli4ct:29": 5045,
    "nli4ct:69": 5216, "nli4ct:160": 4445,
}
MODELS = ["biojev-systemone:0.8b", "biojev-systemone:4b", "biojev-systemone:9b"]


def fixture():
    examples, rows = [], []
    for id_, tokens in OUTLIERS.items():
        examples.append({
            "source": "nli4ct", "id": id_, "state": "clinical note",
            "question": "Entailment?", "answer_key": "entailment",
            "options": [
                {"key": "entailment", "description": "Statement supported"},
                {"key": "contradiction", "description": "Statement contradicted"},
            ],
        })
    examples.append({
        "source": "bionli", "id": "bionli:1", "state": "patient with fever",
        "question": "Entailment?", "answer_key": "entailment",
        "options": [{"key": "entailment", "description": "True"}, {"key": "contradiction", "description": "False"}],
    })
    for model in MODELS:
        for id_, tokens in OUTLIERS.items():
            rows.append({
                "model": model, "source": "nli4ct", "id": id_,
                "gold_key": "entailment", "prediction": "", "confidence": "",
                "gold_probability": "", "elapsed_seconds": "0.003", "error":
                    f'HTTP 400: {{"error":"prompt 0 has {tokens} tokens; expected 1–2048 (input is never truncated)"}}'
                    ' [requires > 4096 context tokens; not retried]',
                "model_used": model, "context_limit_tokens": "2048",
                "attempts": "1", "overflow_tokens": "",
            })
        rows.append({
            "model": model, "source": "bionli", "id": "bionli:1",
            "gold_key": "entailment", "prediction": "entailment", "confidence": "0.99",
            "gold_probability": "0.99", "elapsed_seconds": "0.05", "error": "",
            "model_used": model, "context_limit_tokens": "2048", "attempts": "1", "overflow_tokens": "",
        })
    return examples, rows


class RepairsTests(unittest.TestCase):
    def test_context_allocation(self):
        levels = [r.required_context(v) for v in OUTLIERS.values()]
        self.assertEqual(levels.count(5120), 9)
        self.assertEqual(levels.count(6144), 2)
        self.assertEqual(max(levels), 6144)

    def test_parse_escaped_unicode(self):
        self.assertEqual(r.prompt_requirement('HTTP 400: {"error":"prompt 0 has 5225 tokens; expected 1–2048"}'), (5225, 2048))
        self.assertEqual(r.prompt_requirement('HTTP 400: {"error":"prompt 0 has 5225 tokens; expected 1\\u20132048"}'), (5225, 2048))
        self.assertIsNone(r.prompt_requirement('HTTP 500: GPU out of memory'))

    def test_only_specific_failed_rows(self):
        data, rows = fixture()
        targeted = r.discover_repairs(rows, r.index_examples(data))
        self.assertEqual(sum(map(len, targeted.values())), 33)
        self.assertEqual({x["context"] for tasks in targeted.values() for x in tasks}, {5120, 6144})
        self.assertEqual(len([x for x in rows if not x["error"]]), 3)

    def test_refuse_unexpected_non_overflow(self):
        data, rows = fixture()
        rows[0]["error"] = "HTTP 500: Inference failure"
        with self.assertRaisesRegex(ValueError, "Non-overflow"):
            r.discover_repairs(rows, r.index_examples(data))

    def test_refuse_too_large(self):
        data, rows = fixture()
        rows[0]["error"] = 'HTTP 400: {"error":"prompt 0 has 8200 tokens; expected 1–2048"}'
        with self.assertRaisesRegex(ValueError, "exceeding"):
            r.discover_repairs(rows, r.index_examples(data))

    def test_full_cli_mock_no_base_modifications_and_no_repeat(self):
        data, rows = fixture()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            origin, dest = root / "original", root / "new"
            origin.mkdir()
            dev = root / "dev.jsonl"
            dev.write_text("".join(json.dumps(x) + "\n" for x in data), encoding="utf-8")
            r.save_csv(origin / "predictions.csv", rows, r.PREDICTION_FIELDS)
            (origin / "run_config.json").write_text(json.dumps({"data": str(dev)}), encoding="utf-8")
            content_before = (origin / "predictions.csv").read_bytes()
            called = []
            created, deleted, checked = [], [], []

            def fake_create(model, ctx, run_id):
                tag = f"{model}-s8repair{ctx // 1024}k-{run_id}"
                created.append(tag)
                return tag

            def fake_score(url, model, example, timeout, keep_alive):
                called.append((model, example["id"]))
                return {"prediction": "entailment", "confidence": 0.9,
                        "gold_probability": 0.95, "elapsed_seconds": 0.2, "error": ""}

            saved = sys.argv
            sys.argv = ["repair_sprint8_nli4ct_outliers.py", "--input", str(origin),
                        "--output", str(dest)]
            try:
                with patch.object(r, "create_variant", side_effect=fake_create), \
                     patch.object(r, "remove_variant", side_effect=deleted.append), \
                     patch.object(r, "check_model_context", side_effect=lambda *x: checked.append(x)), \
                     patch.object(r, "score_one", side_effect=fake_score), \
                     contextlib.redirect_stdout(io.StringIO()):
                    r.main()
            finally:
                sys.argv = saved

            self.assertEqual(len(called), 33)
            self.assertEqual(len(set(k for _, k in called)), 11)
            self.assertEqual(len(created), 6)
            self.assertEqual(set(created), set(deleted))
            self.assertEqual(len(checked), 3)
            self.assertTrue(all(ctx == 2048 for _, ctx in checked))
            self.assertEqual((origin / "predictions.csv").read_bytes(), content_before)
            new_rows = r.read_csv(dest / "predictions.csv")
            self.assertEqual(len(new_rows), 36)
            self.assertEqual(sum(bool(x["error"]) for x in new_rows), 0)
            self.assertEqual(sum(x["id"] == "bionli:1" and x["elapsed_seconds"] == "0.05"
                                 for x in new_rows), 3)
            self.assertEqual(len(r.read_csv(dest / "repairs.csv")), 33)
            sums = r.read_csv(dest / "summary.csv")
            for model in MODELS:
                self.assertEqual(next(x for x in sums if x["model"] == model and x["source"] == "nli4ct")["n_success"], "11")


if __name__ == "__main__":
    unittest.main()
