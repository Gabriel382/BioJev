from __future__ import annotations
from typing import Any
from .schemas import SystemOneValidationError

def create_app(adapter):
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as e:
        raise RuntimeError("Install HTTP deps: pip install fastapi uvicorn") from e
    app = FastAPI(title="BioJev System One Compatibility API", version="0.8.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "backend": "biojev-nli-bridge",
                "model": adapter.model_name, "checkpoint": adapter.checkpoint}

    @app.post("/v1/systemone")
    def systemone(payload: dict[str, Any]):
        try:
            return adapter.decide(payload)
        except SystemOneValidationError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
    return app
