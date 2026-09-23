# Sprint 1 experimental protocol

## Evaluation regimes

Keep two regimes distinct throughout the project.

### Zero-shot / task-general
OpenJev and generative Qwen receive no gradient updates on the target dataset.

### Supervised task-specific
BioBERT, PubMedBERT/BiomedBERT and BioLinkBERT are fine-tuned independently for each target dataset.

Never average the two regimes into a single "overall rank" without clearly separating them.

## Seeds

Use 42, 123 and 456 for trained models. Deterministic inference may be run once, but retaining the same suite structure makes later comparisons easier.

## Relation benchmark definition

Sprint 1 defaults to gold entity-pair relation-type classification. This intentionally isolates whether a model can choose the semantic relation once the entities are known.

An optional `NO_RELATION` setting samples negative pairs. If used, report:

- negative sampling ratio;
- allowed entity-type pair rule;
- random seed;
- whether negatives are sampled per document or globally.

## Required result artifacts

A valid paper run should preserve:

- full model identifier/revision;
- exact dataset/preprocessing version;
- seed;
- metrics;
- example-level predictions;
- environment metadata;
- Git commit.
