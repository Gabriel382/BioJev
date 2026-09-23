# Dataset notes

## BioNLI
Public HF mirror: `presencesw/bionli`. The currently surfaced mirror has ~36k examples in one training split with `sentence1`, `sentence2`, and binary `gold_label`. The code creates deterministic dev/test partitions when requested.

## NLI4CT
Public HF mirror: `tasksource/nli4ct`. The adapter uses the section indicated by `Section_id` from the embedded primary/secondary clinical-trial structures to build the NLI premise.

## ChemProt
BigBio config: `bigbio/chemprot`, `chemprot_bigbio_kb`.

## DDI2013
BigBio config: `bigbio/ddi_corpus`, `ddi_corpus_bigbio_kb`.

## BioRED
BigBio config: `bigbio/biored`, `biored_bigbio_kb`. The current BigBio loader imports `bioc.pubtator`, so BioJev declares `bioc>=2.1` as a dependency. The project pins `datasets<4` because this repository still uses a dataset loading script. If a future Hugging Face release drops that mechanism, replace this adapter with a direct NCBI PubTator parser.

## MedNLI
Manual only. Do not automate around PhysioNet credentials or redistribute the corpus.
