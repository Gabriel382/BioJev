#!/usr/bin/env python
import argparse, json
from pathlib import Path
from biojev.systemone.adapter import BioJevSystemOneAdapter

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--request", required=True)
p.add_argument("--model-name", default="biojev")
p.add_argument("--device", choices=["cuda", "cpu"], default=None)
p.add_argument("--load-in-4bit", action="store_true")
p.add_argument("--max-length", type=int, default=2048)
a = p.parse_args()
payload = json.loads(Path(a.request).read_text())
adapter = BioJevSystemOneAdapter(a.checkpoint, a.model_name, a.device, a.load_in_4bit, a.max_length)
print(json.dumps(adapter.decide(payload), indent=2, ensure_ascii=False))
