# BioJev Sprint 2 status

Implemented in v0.2.0:

- [x] Sprint-1 dataset summary table after `download_datasets.py`
- [x] Qwen3.5-4B-Base biomedical DAPT
- [x] streaming PubMed + PMC corpus
- [x] deterministic 98/1/1 document partitioning
- [x] 100M / 500M / 1B / 3B token configs
- [x] exact token-budget packing
- [x] QLoRA
- [x] LoRA
- [x] full fine-tuning
- [x] bf16/fp16 mixed precision
- [x] gradient accumulation
- [x] gradient checkpointing
- [x] Trainer checkpoints and resume
- [x] Accelerate multi-GPU launch
- [x] JSONL + TensorBoard logging; optional W&B
- [x] PEFT and merged HF export
- [x] held-out biomedical perplexity evaluation
- [x] Qwen-Base vs BioQwen multi-dataset likelihood benchmark
- [x] PowerShell launchers
- [x] Python cell walkthrough
- [x] Jupyter notebook walkthrough
- [x] offline unit tests for corpus partitioning/packing

Intentionally not part of Sprint 2:

- BioJev NLI/decision supervision;
- RAG;
- agent/tool use;
- task-specific biomedical supervised fine-tuning.
