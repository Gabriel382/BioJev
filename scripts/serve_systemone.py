#!/usr/bin/env python
import argparse
from biojev.systemone.adapter import BioJevSystemOneAdapter
from biojev.systemone.api import create_app

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--model-name", default="biojev")
p.add_argument("--host", default="127.0.0.1")
p.add_argument("--port", type=int, default=8000)
p.add_argument("--device", choices=["cuda", "cpu"], default=None)
p.add_argument("--load-in-4bit", action="store_true")
p.add_argument("--max-length", type=int, default=2048)
a = p.parse_args()

adapter = BioJevSystemOneAdapter(a.checkpoint, a.model_name, a.device, a.load_in_4bit, a.max_length)
app = create_app(adapter)

import uvicorn
uvicorn.run(app, host=a.host, port=a.port)
