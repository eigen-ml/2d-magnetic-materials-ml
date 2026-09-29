"""REST API around the composition-only magnetic screening model.

The model, threshold and feature schema are the same artefacts used by
``scripts/04_discover.py``; this module only exposes them over HTTP.
Scores are uncalibrated model outputs, not probabilities of a real phase.
"""
from __future__ import annotations

import importlib.util
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "models"))

# scripts/04_discover.py starts with a digit, so it cannot be imported by name.
_spec = importlib.util.spec_from_file_location(
    "discover", ROOT / "scripts" / "04_discover.py"
)
discover = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(discover)

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    classifier, threshold, metadata = discover.load_model(MODEL_DIR)
    state.update(
        classifier=classifier,
        threshold=threshold,
        metadata=metadata,
        positive_index=discover.positive_class_index(classifier),
    )
    yield
    state.clear()


app = FastAPI(
    title="2D magnetic materials screening API",
    description="Composition-only classifier for magnetic ordering in 2D materials.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    formulas: list[str] = Field(..., min_length=1, max_length=1000)


class Prediction(BaseModel):
    formula: str
    reduced_formula: str | None
    magnetic_score: float | None
    predicted_magnetic: bool | None
    score_band: str | None
    error: str | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "classifier" in state}


@app.get("/model")
def model_info() -> dict:
    meta = state["metadata"]
    return {
        "model_version": meta.get("model_version"),
        "threshold": state["threshold"],
        "n_features": meta.get("n_features"),
        "metrics": meta.get("metrics"),
    }


@app.post("/predict", response_model=list[Prediction])
def predict(request: PredictRequest) -> list[Prediction]:
    if "classifier" not in state:
        raise HTTPException(status_code=503, detail="model not loaded")
    feature_names = state["metadata"]["feature_names"]
    results: list[Prediction] = []
    rows: list[dict] = []
    row_positions: list[int] = []

    for formula in request.formulas:
        reduced = discover.normalize_formula(formula)
        features = discover.featurize(reduced, feature_names) if reduced else None
        if features is None:
            reason = "could not parse formula" if reduced is None else (
                "contains elements the model was not trained on"
            )
            results.append(Prediction(
                formula=formula, reduced_formula=reduced, magnetic_score=None,
                predicted_magnetic=None, score_band=None, error=reason,
            ))
        else:
            row_positions.append(len(results))
            rows.append(features)
            results.append(Prediction(
                formula=formula, reduced_formula=reduced, magnetic_score=0.0,
                predicted_magnetic=False, score_band="",
            ))

    if rows:
        frame = pd.DataFrame(rows, columns=feature_names)
        scores = np.asarray(state["classifier"].predict_proba(frame))[
            :, state["positive_index"]
        ]
        for position, score in zip(row_positions, scores):
            item = results[position]
            item.magnetic_score = float(score)
            item.predicted_magnetic = bool(score >= state["threshold"])
            item.score_band = discover.score_band(float(score))
    return results
