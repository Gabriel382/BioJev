# BioJev Sprint 4 status

Sprint 4 adds the multi-dataset and cross-dataset generalization layer.

Implemented:

- frozen multi-dataset runner that loads each model once;
- 4B and Nano size-matched Qwen/OpenJev/BioJev suites;
- BioNLI, NLI4CT, ChemProt, DDI2013 and BioRED evaluation;
- explicit gold-pair relation-typing scope;
- per-class metrics and confusion matrices;
- raw relation entailment scores for error analysis;
- BioNLI-only and NLI4CT-only decision-training configs for cross-dataset NLI;
- conventional supervised BioBERT/PubMedBERT/BioLinkBERT benchmark runner;
- CSV/JSON aggregation and paper-table generation;
- PowerShell and Jupyter/VS Code walkthroughs.

The 4B Sprint-3 checkpoint can be evaluated immediately at:

`outputs/biojev/4b_full/stage-02-biomedical/final`

The Nano suite becomes runnable as soon as:

`outputs/biojev/nano_full/stage-02-biomedical/final`

exists.
