from __future__ import annotations

import platform

import torch


def main() -> None:
    print("BioJev GPU environment check")
    print(f"  Python:            {platform.python_version()}")
    print(f"  PyTorch:           {torch.__version__}")
    print(f"  PyTorch CUDA build:{torch.version.cuda}")
    print(f"  CUDA available:    {torch.cuda.is_available()}")
    print(f"  CUDA device count: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        print(f"  Current device:    {idx}")
        print(f"  GPU:               {torch.cuda.get_device_name(idx)}")
        print(f"  Capability:        {torch.cuda.get_device_capability(idx)}")
        print(f"  BF16 supported:    {torch.cuda.is_bf16_supported()}")
    else:
        print()
        print("CUDA is not available to this Python environment.")
        print("BioJev training with device=auto will fall back to CPU + LoRA + float32.")
        print("This is supported but a 4B model will be much slower than GPU training.")


if __name__ == "__main__":
    main()
