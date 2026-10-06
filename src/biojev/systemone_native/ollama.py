
from __future__ import annotations
import json, urllib.request
from typing import Any

def systemone_payload(model: str, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
    return {"model": model, "state": state, "questions": questions}

def post_systemone(payload, base_url="http://127.0.0.1:11434", timeout=120):
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/systemone",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)

def validate_systemone_response(result, question_types):
    answers = result.get("answers")
    if not isinstance(answers, dict):
        raise AssertionError("missing answers")
    for name, typ in question_types.items():
        if name not in answers:
            raise AssertionError(f"missing answer {name}")
        a = answers[name]
        if a.get("type") != typ:
            raise AssertionError(f"{name}: expected {typ}, got {a.get('type')}")
        if typ == "choice":
            for k in ("choice", "probabilities", "confidence"):
                if k not in a: raise AssertionError(f"{name}: missing {k}")
        elif typ == "noul":
            p = a.get("noul")
            if not isinstance(p, (int,float)) or not 0 <= p <= 1:
                raise AssertionError(f"{name}: invalid noul")
        elif typ == "score":
            for k in ("score", "probabilities", "confidence"):
                if k not in a: raise AssertionError(f"{name}: missing {k}")
