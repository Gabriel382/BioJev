from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any


def flatten_passages(passages: list[dict[str, Any]]) -> str:
    pieces: list[str] = []
    for passage in passages or []:
        text = passage.get("text", "")
        if isinstance(text, list):
            pieces.extend(str(x) for x in text)
        elif text:
            pieces.append(str(text))
    return "\n".join(piece.strip() for piece in pieces if piece and piece.strip())


def entity_lookup(entities: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(entity["id"]): entity for entity in entities or [] if "id" in entity}


def entity_text(entity: dict[str, Any]) -> str:
    text = entity.get("text", "")
    if isinstance(text, list):
        return " ".join(str(x) for x in text)
    return str(text)


def observed_relation_labels(rows: list[dict[str, Any]]) -> list[str]:
    labels = set()
    for row in rows:
        for relation in row.get("relations", []) or []:
            if relation.get("type"):
                labels.add(str(relation["type"]))
    return sorted(labels)


def relation_pairs(row: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any], str, str]]:
    entities = entity_lookup(row.get("entities", []))
    output = []
    for relation in row.get("relations", []) or []:
        a = entities.get(str(relation.get("arg1_id")))
        b = entities.get(str(relation.get("arg2_id")))
        if not a or not b:
            continue
        output.append((a, b, str(relation.get("type", "UNKNOWN")), str(relation.get("id", ""))))
    return output


def allowed_type_pairs(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for row in rows:
        for a, b, _, _ in relation_pairs(row):
            ta, tb = str(a.get("type", "")), str(b.get("type", ""))
            pairs.add(tuple(sorted((ta, tb))))
    return pairs


def negative_entity_pairs(row: dict[str, Any], allowed: set[tuple[str, str]]) -> list[tuple[dict, dict]]:
    entities = row.get("entities", []) or []
    gold = {
        frozenset((str(a.get("id")), str(b.get("id"))))
        for a, b, _, _ in relation_pairs(row)
    }
    output = []
    for a, b in combinations(entities, 2):
        ta, tb = str(a.get("type", "")), str(b.get("type", ""))
        if tuple(sorted((ta, tb))) not in allowed:
            continue
        key = frozenset((str(a.get("id")), str(b.get("id"))))
        if key not in gold:
            output.append((a, b))
    return output
