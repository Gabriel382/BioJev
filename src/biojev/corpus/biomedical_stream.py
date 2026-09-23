from __future__ import annotations

import hashlib
import random
import warnings
from dataclasses import dataclass
from typing import Any, Iterator

from torch.utils.data import IterableDataset


COMMERCIAL_PMC_LICENSES = {
    "CC0",
    "CC BY",
    "CC BY-SA",
    "CC BY-ND",
    "CC0-1.0",
    "CC-BY-4.0",
    "CC-BY-SA-4.0",
    "CC-BY-ND-4.0",
}


@dataclass(frozen=True)
class CorpusSource:
    name: str
    dataset: str
    split: str = "train"
    weight: float = 1.0
    text_field: str | None = None
    title_field: str | None = None
    abstract_field: str | None = None
    id_field: str | None = None
    trust_remote_code: bool = False
    commercial_only: bool = False
    skip_retracted: bool = True
    min_chars: int = 100
    shuffle_buffer: int = 10_000
    load_kwargs: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CorpusSource":
        return cls(**data)


def load_corpus_sources(config: dict[str, Any]) -> list[CorpusSource]:
    sources = [CorpusSource.from_dict(x) for x in config.get("sources", [])]
    if not sources:
        raise ValueError("At least one biomedical corpus source is required")
    if any(s.weight <= 0 for s in sources):
        raise ValueError("All source weights must be > 0")
    return sources


def stable_partition(doc_id: str, modulo: int = 100) -> int:
    digest = hashlib.sha1(doc_id.encode("utf-8", errors="ignore")).digest()
    return int.from_bytes(digest[:8], "big") % modulo


def belongs_to_partition(doc_id: str, partition: str) -> bool:
    """Deterministic 98/1/1 train/validation/test split by document id."""
    bucket = stable_partition(doc_id, modulo=100)
    if partition == "train":
        return bucket < 98
    if partition in {"validation", "dev"}:
        return bucket == 98
    if partition == "test":
        return bucket == 99
    if partition == "all":
        return True
    raise ValueError(f"Unknown corpus partition: {partition}")


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    return " ".join(str(value).split())


def _extract_text(row: dict[str, Any], source: CorpusSource) -> tuple[str, str] | None:
    if source.skip_retracted and str(row.get("retracted", "no")).lower() in {"yes", "true", "1"}:
        return None

    if source.commercial_only and "license" in row:
        license_name = _normalize_text(row.get("license")).upper()
        normalized = license_name.replace(" ", "-")
        allowed = {x.upper().replace(" ", "-") for x in COMMERCIAL_PMC_LICENSES}
        if normalized not in allowed:
            return None

    if source.text_field:
        text = _normalize_text(row.get(source.text_field))
    else:
        title = _normalize_text(row.get(source.title_field or "title"))
        abstract = _normalize_text(row.get(source.abstract_field or "abstract"))
        if title and abstract:
            text = f"{title}\n\n{abstract}"
        else:
            text = abstract or title

    if len(text) < source.min_chars:
        return None

    candidate_fields = [source.id_field, "PMID", "pmid", "accession_id", "id"]
    doc_id = ""
    for field in candidate_fields:
        if field and row.get(field) not in {None, ""}:
            doc_id = str(row[field])
            break
    if not doc_id:
        # Stable fallback based on content. We intentionally avoid Python's randomized hash().
        doc_id = hashlib.sha1(text[:4096].encode("utf-8", errors="ignore")).hexdigest()
    return text, doc_id


def probe_corpus_source(source: CorpusSource, *, max_rows: int = 100) -> dict[str, Any]:
    """Verify that a streaming corpus source is reachable and yields usable text.

    This deliberately runs before the 4B model is loaded so broken remote datasets,
    renamed splits, schema drift, or retired loader scripts fail fast.
    """
    from datasets import load_dataset

    try:
        ds = load_dataset(
            source.dataset,
            split=source.split,
            streaming=True,
            trust_remote_code=source.trust_remote_code,
            **(source.load_kwargs or {}),
        )
        for i, row in enumerate(ds):
            extracted = _extract_text(row, source)
            if extracted is not None:
                text, doc_id = extracted
                return {
                    "name": source.name,
                    "dataset": source.dataset,
                    "split": source.split,
                    "doc_id": doc_id,
                    "text_chars": len(text),
                }
            if i + 1 >= max_rows:
                break
    except Exception as exc:
        raise RuntimeError(
            f"BioJev corpus preflight failed for source '{source.name}' "
            f"({source.dataset}, split={source.split!r}): {exc}"
        ) from exc

    raise RuntimeError(
        f"BioJev corpus preflight reached {max_rows} rows for source '{source.name}' "
        f"({source.dataset}, split={source.split!r}) without finding a usable document. "
        "Check its field names and filtering options."
    )


def probe_corpus_sources(sources: list[CorpusSource]) -> list[dict[str, Any]]:
    results = []
    for source in sources:
        result = probe_corpus_source(source)
        results.append(result)
        print(
            f"  corpus OK: {source.name:<8} {source.dataset} "
            f"split={source.split} sample_chars={result['text_chars']:,}"
        )
    return results


class BiomedicalCorpusStream(IterableDataset):
    """Weighted, streaming PubMed/PMC text source.

    The source datasets remain streaming; this class does not materialize the corpus locally.
    A deterministic document-id partition prevents overlap between DAPT train and held-out
    perplexity evaluation.
    """

    def __init__(
        self,
        sources: list[CorpusSource],
        *,
        partition: str = "train",
        seed: int = 42,
        shuffle: bool = True,
    ) -> None:
        super().__init__()
        self.sources = sources
        self.partition = partition
        self.seed = seed
        self.shuffle = shuffle

    def _source_iterator(self, source: CorpusSource, seed: int) -> Iterator[dict[str, str]]:
        from datasets import load_dataset

        ds = load_dataset(
            source.dataset,
            split=source.split,
            streaming=True,
            trust_remote_code=source.trust_remote_code,
            **(source.load_kwargs or {}),
        )
        if self.shuffle and source.shuffle_buffer > 0:
            ds = ds.shuffle(seed=seed, buffer_size=source.shuffle_buffer)

        for row in ds:
            extracted = _extract_text(row, source)
            if extracted is None:
                continue
            text, doc_id = extracted
            if not belongs_to_partition(doc_id, self.partition):
                continue
            yield {"text": text, "doc_id": doc_id, "source": source.name}

    def __iter__(self) -> Iterator[dict[str, str]]:
        # The Trainer/Accelerate dataloader handles process sharding. Every process therefore
        # starts from the same deterministic stream and receives disjoint batches downstream.
        rng = random.Random(self.seed)
        active = {
            source.name: {
                "source": source,
                "iterator": iter(self._source_iterator(source, self.seed + i * 9973)),
            }
            for i, source in enumerate(self.sources)
        }

        while active:
            names = list(active)
            weights = [active[name]["source"].weight for name in names]
            name = rng.choices(names, weights=weights, k=1)[0]
            try:
                yield next(active[name]["iterator"])
            except StopIteration:
                del active[name]
