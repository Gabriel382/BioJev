#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            pass
    return total


def human(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(n)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TB"


def main() -> None:
    home = Path.home()
    locations = {
        "BioJev data/": Path("data"),
        "BioJev runs/": Path("runs"),
        "BioJev results/": Path("results"),
        "Hugging Face hub cache": home / ".cache" / "huggingface" / "hub",
        "Hugging Face datasets cache": home / ".cache" / "huggingface" / "datasets",
    }
    total = 0
    for label, path in locations.items():
        size = directory_size(path)
        total += size
        print(f"{label:30} {human(size):>12}  {path}")
    print("-" * 70)
    print(f"{'Total across locations':30} {human(total):>12}")
    print("Note: caches may overlap logically; on Windows without symlinks, physical use can be higher.")


if __name__ == "__main__":
    main()
