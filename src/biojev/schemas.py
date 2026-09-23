from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

TaskType = Literal["nli", "relation_classification"]


class BaseExample(BaseModel):
    id: str
    dataset: str
    split: str
    task: TaskType
    metadata: dict[str, Any] = Field(default_factory=dict)


class NLIExample(BaseExample):
    task: Literal["nli"] = "nli"
    premise: str
    hypothesis: str
    label: str


class RelationExample(BaseExample):
    task: Literal["relation_classification"] = "relation_classification"
    context: str
    subject: str
    object: str
    subject_type: str | None = None
    object_type: str | None = None
    label: str
    candidates: list[str] = Field(default_factory=list)


class PredictionRecord(BaseModel):
    id: str
    dataset: str
    split: str
    task: TaskType
    gold: str
    prediction: str
    probabilities: dict[str, float]
    metadata: dict[str, Any] = Field(default_factory=dict)
