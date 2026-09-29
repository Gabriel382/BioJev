# BioJev Sprint 3 status

Delivered:

- Biomedical decision/NLI trainer with weighted multi-dataset stages.
- Nano-first configs plus a 4B rerun config.
- OpenJev biomedical-adaptation baselines plus a Base-aligned BioQwen -> general NLI -> biomedical NLI main path.
- Generic Qwen -> general NLI -> biomedical NLI path for sizes without OpenJev checkpoints.
- Automatic GPU/CPU runtime selection and QLoRA -> LoRA CPU fallback.
- Sprint-2 DAPT LoRA transplantation into sequence classification.
- Fixed OpenJev labels: contradiction=0, entailment=1, neutral=2.
- Typed `decide(context, hypotheses)` API.
- Sprint-1 benchmark compatibility for the resulting BioJev checkpoint.
- SNLI, MNLI and ANLI adapters for optional general NLI bootstrapping.
- Hugging Face merged export.
- PowerShell and ordered notebook workflows.

Nano DAPT input: `outputs/biojev-nano/final`

Nano Sprint-3 target: `outputs/biojev/nano_full/stage-02-biomedical/final`

Qwen3.5's current dense family uses 9B rather than 8B. If a desired scale has no matching OpenJev
checkpoint, use the generic `qwen_seqcls` two-stage route with that model ID.
