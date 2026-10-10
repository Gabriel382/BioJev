#!/usr/bin/env python3
"""Create optional Ollama 4K-context variants; never modify base BioJev tags.

Run in the BioJev repository on the machine with Ollama installed.
"""
import argparse
import re
import subprocess
import tempfile
from pathlib import Path

DEFAULT_MODELS = (
    "biojev-systemone:0.8b", "biojev-systemone:4b", "biojev-systemone:9b"
)


def context_setting(modelfile):
    matches = re.findall(r"(?im)^\s*PARAMETER\s+num_ctx\s+(\d+)\s*$", modelfile)
    return int(matches[-1]) if matches else None


def ollama_modelfile(model):
    completed = subprocess.run(
        ["ollama", "show", "--modelfile", model],
        text=True, capture_output=True, check=True,
    )
    return completed.stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    ap.add_argument("--extended-context", type=int, default=4096)
    ap.add_argument("--suffix", default="-4k", help="Extended variant suffix after tag")
    ap.add_argument("--base-context", type=int, default=2048,
                    help="Require that original tag keeps this num_ctx")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.extended_context <= args.base_context:
        ap.error("--extended-context must be greater than --base-context")
    if not args.suffix or any(char.isspace() for char in args.suffix):
        ap.error("--suffix must be nonempty and have no spaces")

    for model in args.models:
        if ":" not in model:
            ap.error(f"A specific tagged model is required: {model}")
        extended = model + args.suffix
        modelfile = (
            f"FROM {model}\n"
            "CAPABILITY decision\n"
            f"PARAMETER num_ctx {args.extended_context}\n"
        )
        print(f"\n{model} -> {extended}\n{modelfile}", flush=True)
        if args.dry_run:
            continue

        # Refuse to proceed if original settings have changed unexpectedly.
        base_file = ollama_modelfile(model)
        original_context = context_setting(base_file)
        if original_context != args.base_context:
            raise SystemExit(
                f"Refusing to create {extended}: {model} has num_ctx="
                f"{original_context!r}, expected {args.base_context}. "
                "Original model was not modified. Check `ollama show --modelfile`."
            )

        with tempfile.TemporaryDirectory(prefix="biojev-context-") as temp:
            path = Path(temp) / "Modelfile"
            path.write_text(modelfile, encoding="utf-8")
            subprocess.run(["ollama", "create", extended, "-f", str(path)], check=True)

        actual_context = context_setting(ollama_modelfile(extended))
        if actual_context != args.extended_context:
            raise SystemExit(
                f"{extended} created, but Ollama reports num_ctx="
                f"{actual_context!r} (expected {args.extended_context})."
            )
        print(f"OK: {extended} configured for {actual_context} tokens; original unchanged")


if __name__ == "__main__":
    main()
