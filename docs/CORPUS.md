# Biomedical DAPT corpus

BioJev Sprint 2 intentionally keeps corpus acquisition **streaming**.

## PubMed abstracts

Default source: `slinusc/PubMedAbstractsSubset`.

Fields used:

```text
PMID
Title
Abstract
```

The title and abstract are concatenated with a blank line. The source is useful for a lightweight, reproducible PubMed-scale stream without cloning the complete PubMed baseline locally.

## PMC Open Access

Default source: `aochongoliverli/pmc_openaccess_split`.

The `text` field contains the full article. BioJev asks the PMC loader for its **commercial** subset directly, so non-commercial/other archives are not part of the default stream. By default BioJev:

- accepts only commercial-use license labels (`CC0`, `CC BY`, `CC BY-SA`, `CC BY-ND` variants);
- skips records marked retracted;
- drops very short records.

Change `commercial_only: false` only after checking the licensing implications for your intended model distribution.

## Train/validation/test isolation

Every document ID is hashed into 100 deterministic buckets:

```text
00–97 -> train
98    -> validation
99    -> test
```

This means the held-out perplexity corpus is stable across runs and never used in DAPT training.

## Mixture weights

The default is:

```yaml
pubmed: 0.8
pmc:    0.2
```

Weights are *sampling probabilities*, not fixed byte proportions. The token budget controls when training stops.

## Streaming and disk use

Streaming avoids downloading the complete multi-million-document corpus before training. Hugging Face still caches accessed shards locally, so disk usage grows during long runs. The exact cache size depends on which remote shards are touched and your Hugging Face cache settings.
