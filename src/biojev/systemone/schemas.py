from __future__ import annotations
import json
from typing import Any

class SystemOneValidationError(ValueError):
    pass

def _content(value: Any, field: str) -> str:
    if isinstance(value, str):
        if not value.strip():
            raise SystemOneValidationError(f"{field} must not be empty")
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    raise SystemOneValidationError(f"{field} must be a string, object, or array")

def validate_request(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise SystemOneValidationError("request body must be an object")
    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        raise SystemOneValidationError("model is required")
    state = _content(payload.get("state"), "state")
    questions = payload.get("questions")
    if not isinstance(questions, dict) or not (1 <= len(questions) <= 64):
        raise SystemOneValidationError("questions must contain 1-64 fields")
    out = {}
    for name, q in questions.items():
        if not isinstance(name, str) or not name.strip():
            raise SystemOneValidationError("question names must not be empty")
        if not isinstance(q, dict):
            raise SystemOneValidationError(f"question {name!r} must be an object")
        typ = q.get("type")
        if typ not in {"choice", "noul", "score"}:
            raise SystemOneValidationError(f"question {name!r}: invalid type")
        instructions = _content(q.get("instructions"), f"question {name!r} instructions")
        if typ == "choice":
            criteria = q.get("criteria")
            if not isinstance(criteria, dict) or not (2 <= len(criteria) <= 26):
                raise SystemOneValidationError(f"question {name!r}: choice criteria must contain 2-26 options")
            options = []
            for key, desc in criteria.items():
                if not isinstance(key, str) or not key.strip():
                    raise SystemOneValidationError(f"question {name!r}: blank choice key")
                if desc is None:
                    desc = key
                if not isinstance(desc, str):
                    raise SystemOneValidationError(f"question {name!r}: descriptions must be strings or null")
                options.append((key, desc))
        elif typ == "noul":
            criteria = q.get("criteria") or {}
            if not isinstance(criteria, dict) or set(criteria) - {"false", "true"}:
                raise SystemOneValidationError(f"question {name!r}: invalid noul criteria")
            f = criteria.get("false", "No")
            t = criteria.get("true", "Yes")
            if not isinstance(f, str) or not isinstance(t, str):
                raise SystemOneValidationError(f"question {name!r}: noul descriptions must be strings")
            options = [("false", f), ("true", t)]
        else:
            criteria = q.get("criteria")
            if not isinstance(criteria, list) or not (2 <= len(criteria) <= 26) or not all(isinstance(x, str) for x in criteria):
                raise SystemOneValidationError(f"question {name!r}: score criteria must contain 2-26 strings")
            options = [(str(i), d) for i, d in enumerate(criteria)]
        out[name] = {"type": typ, "instructions": instructions, "options": options}
    return {"model": model.strip(), "state": state, "questions": out, "keep_alive": payload.get("keep_alive")}
