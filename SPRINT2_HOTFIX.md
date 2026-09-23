# Sprint 2 hotfix 0.2.1

- Fixed Transformers v5.2+ compatibility: `TrainingArguments.warmup_ratio` was removed upstream.
- BioJev now detects the installed Transformers API and uses `warmup_ratio` on older releases or float-valued `warmup_steps` on newer releases.
- No model checkpoint redownload is required; Hugging Face will reuse the existing local cache.
