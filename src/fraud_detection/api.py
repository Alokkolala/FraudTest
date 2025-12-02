"""Lightweight FastAPI app for uploading CSVs and receiving fraud scores."""

from __future__ import annotations

import io

import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from .config import DEFAULT_MODEL_CONFIG, DEFAULT_PIPELINE_CONFIG, DEFAULT_RULE_CONFIG
from .pipeline import score_transactions, train_model

app = FastAPI(title="Fraud Detection Prototype")


@app.on_event("startup")
def _load_default_model():
    global DEFAULT_MODEL
    DEFAULT_MODEL = train_model(model_config=DEFAULT_MODEL_CONFIG, pipeline_config=DEFAULT_PIPELINE_CONFIG)


@app.post("/score")
async def score_endpoint(file: UploadFile = File(...), threshold: float = 0.6, compute_importance: bool = False):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    results = score_transactions(
        DEFAULT_MODEL,
        df,
        threshold=threshold,
        rule_config=DEFAULT_RULE_CONFIG,
        pipeline_config=DEFAULT_PIPELINE_CONFIG,
        compute_importance=compute_importance,
    )
    return JSONResponse(results.to_dict(orient="records"))


@app.get("/health")
async def health():
    return {"status": "ok"}
