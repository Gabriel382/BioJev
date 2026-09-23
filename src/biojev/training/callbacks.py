from __future__ import annotations

import json
from pathlib import Path

from transformers import TrainerCallback


class JsonlLogCallback(TrainerCallback):
    def __init__(self, output_dir: str | Path):
        self.path = Path(output_dir) / "trainer_log.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        payload = {"step": state.global_step, **logs}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
