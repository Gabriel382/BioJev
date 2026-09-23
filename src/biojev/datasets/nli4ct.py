from __future__ import annotations

from typing import Any

from datasets import load_dataset

from biojev.datasets.base import DatasetAdapter
from biojev.schemas import NLIExample
from biojev.utils.io import write_jsonl


def _section_text(trial: Any, section_id: str | None) -> str:
    if trial is None:
        return ""
    if isinstance(trial, dict):
        if section_id and section_id in trial:
            value = trial[section_id]
            if isinstance(value, list):
                return "\n".join(str(x) for x in value)
            return str(value)
        # Fall back to all human-readable trial sections.
        pieces = []
        for key, value in trial.items():
            if str(key).lower() in {"clinical trial id", "id"}:
                continue
            if isinstance(value, list):
                text = "\n".join(str(x) for x in value)
            else:
                text = str(value)
            if text.strip():
                pieces.append(f"[{key}]\n{text}")
        return "\n".join(pieces)
    return str(trial)


class NLI4CTAdapter(DatasetAdapter):
    name = "nli4ct"

    def prepare(self, force: bool = False):
        # This public HF mirror packages each statement together with the linked CTR content,
        # avoiding brittle manual joins against the 999 clinical-trial JSON files.
        ds = load_dataset("tasksource/nli4ct")
        output: dict[str, list[NLIExample]] = {}
        for hf_split, split_ds in ds.items():
            split = "dev" if hf_split in {"validation", "valid", "dev"} else hf_split
            rows: list[NLIExample] = []
            for i, row in enumerate(split_ds):
                label = row.get("Label") or row.get("label")
                statement = row.get("Statement") or row.get("statement")
                if label is None or statement is None:
                    # Public test sets can intentionally omit labels.
                    continue
                section = row.get("Section_id") or row.get("section_id")
                primary = _section_text(row.get("Primary_ct") or row.get("primary_ct"), section)
                secondary = _section_text(row.get("Secondary_ct") or row.get("secondary_ct"), section)
                premise = primary
                if secondary.strip():
                    premise += "\n\n[SECONDARY CLINICAL TRIAL]\n" + secondary
                example_id = row.get("__index_level_0__") or row.get("id") or f"{split}-{i}"
                rows.append(
                    NLIExample(
                        id=f"nli4ct-{example_id}",
                        dataset=self.name,
                        split=split,
                        premise=premise,
                        hypothesis=str(statement),
                        label=str(label).lower(),
                        metadata={
                            "type": row.get("Type"),
                            "section_id": section,
                            "primary_id": row.get("Primary_id"),
                            "secondary_id": row.get("Secondary_id"),
                            "primary_evidence_index": row.get("Primary_evidence_index"),
                            "secondary_evidence_index": row.get("Secondary_evidence_index"),
                        },
                    )
                )
            if rows:
                output[split] = rows
                write_jsonl(self.processed_root / self.name / f"{split}.jsonl", rows)
        if not output:
            raise RuntimeError("No labeled NLI4CT examples were found in tasksource/nli4ct.")
        return output
