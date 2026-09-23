# Disk usage expectations

These are practical planning estimates for Sprint 1 on Windows.

## Dataset preparation only

The public source downloads are small compared with the models:

- BioNLI: about 18 MB of source parquet data.
- NLI4CT: about 4.5 MB.
- ChemProt BigBio KB config: about 3 MB of parquet downloads.
- DDI2013: about 1–2 MB in the currently used parquet representation.
- BioRED: a small 600-abstract corpus downloaded by the BigBio loading script.

A reasonable expectation is **roughly 30–50 MB of network dataset payload** for the five public datasets.

However, Hugging Face keeps cache files and BioJev also writes normalized JSONL. On Windows without symlink support, cached duplicates can use more physical space. Reserve **about 150–400 MB** for the complete Sprint-1 dataset/caching layer; **1 GB free** is a comfortable margin.

Use:

```powershell
python scripts/report_disk_usage.py
```

to measure your actual machine after preparation.

## Models are much larger

Dataset size should not be confused with model downloads. At the time this Sprint-1 snapshot was prepared:

- OpenJev Qwen3.5-4B NLI checkpoint: about **9.1 GB**.
- Qwen3.5-4B baseline: about **9.3 GB**.
- BioBERT/BiomedBERT/BioLinkBERT checkpoints are each hundreds of MB.

Therefore a full Sprint-1 model cache can readily exceed **20 GB**, and supervised training checkpoints/results can push total project usage much higher. Keep **30–50 GB free** if you intend to run and retain the entire baseline matrix locally.
