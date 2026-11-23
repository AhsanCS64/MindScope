from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from threading import BoundedSemaphore

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from .schemas.prediction import (
    ExplanationRequest,
    ExplanationResponse,
    PredictionResponse,
    StudentFeatures,
)
from .services.analytics import DatasetAnalytics
from .services.explainer import EnsembleExplainer
from .services.predictor import ARTIFACTS, DISCLAIMER, ROOT, EnsemblePredictor, load_dataset

FIGURES = ROOT / "reports" / "figures"
DEPENDENCE_PLOTS = {
    "academic-pressure": ("Academic Pressure", "dependence_num__Academic_Pressure.png"),
    "financial-stress": ("Financial Stress", "dependence_num__Financial_Stress.png"),
    "age": ("Age", "dependence_num__Age.png"),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor = EnsemblePredictor()
    dataset = load_dataset()
    app.state.predictor = predictor
    app.state.analytics = DatasetAnalytics(dataset, predictor)
    app.state.explainer = EnsembleExplainer(predictor, app.state.analytics.model_population)
    app.state.global_shap_summary = read_artifact("global_shap.json")
    app.state.model_metrics = read_artifact("model_metrics.json")
    app.state.summary = app.state.analytics.summary()
    app.state.features = app.state.analytics.features()
    app.state.correlations = app.state.analytics.correlations()
    app.state.numeric_bounds = {
        item["name"]: {key: item["statistics"][key] for key in ("min", "max")}
        for item in app.state.features if item["kind"] == "numeric"
    }
    app.state.explanation_slot = BoundedSemaphore(1)
    yield


app = FastAPI(
    title="MindScope API",
    description=(
        "Educational research demonstration for the dataset's recorded label. "
        "Not a diagnostic or clinical screening system."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

origins = [value.strip() for value in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if value.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def read_artifact(name: str) -> dict | None:
    path = ARTIFACTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, _exc: RequestValidationError) -> JSONResponse:
    # Do not echo the submitted values or any unexpected user-supplied keys.
    return JSONResponse(status_code=422, content={
        "detail": {"message": "Send all 15 expected features with the required numeric and categorical types."}
    })


def validate_features(payload: StudentFeatures, request: Request) -> dict:
    features = payload.to_feature_dict()
    predictor: EnsemblePredictor = request.app.state.predictor
    for name, bounds in request.app.state.numeric_bounds.items():
        if not bounds["min"] <= features[name] <= bounds["max"]:
            raise HTTPException(status_code=422, detail={
                "field": name, "message": f"{name} must be between {bounds['min']:g} and {bounds['max']:g}.",
            })
    for name, choices in predictor.categories.items():
        if features[name] not in choices:
            raise HTTPException(status_code=422, detail={
                "field": name, "message": f"Choose a supported category for {name}.",
            })
    return features


@app.get("/api/health")
def health(request: Request) -> dict[str, str]:
    return {"status": "ok", "model_version": request.app.state.predictor.model_version}


@app.get("/api/model/info")
def model_info(request: Request) -> dict:
    predictor: EnsemblePredictor = request.app.state.predictor
    return {
        "model_version": predictor.model_version,
        "dataset": f"OpenML {predictor.metadata['openml_dataset_id']}",
        "algorithm": "Five-seed XGBoost probability ensemble",
        "threshold": predictor.threshold,
        "positive_label": "Yes",
        "negative_label": "No",
        "feature_columns": predictor.feature_columns,
        "categorical_columns": predictor.categorical_columns,
        "category_options": predictor.categories,
        "numeric_bounds": request.app.state.numeric_bounds,
        "sensitive_feature_excluded": True,
        "disclaimer": DISCLAIMER,
    }


@app.get("/api/model/metrics")
def model_metrics(request: Request) -> dict:
    metrics = request.app.state.model_metrics
    if metrics is None:
        raise HTTPException(status_code=503, detail="The model verification artifact is unavailable.")
    return metrics


@app.get("/api/dataset/summary")
def dataset_summary(request: Request) -> dict:
    return request.app.state.summary


@app.get("/api/analytics/features")
def feature_analytics(request: Request) -> dict:
    return {"features": request.app.state.features}


@app.get("/api/analytics/correlations")
def correlations(request: Request) -> dict:
    return request.app.state.correlations


@app.get("/api/analytics/features/{name}")
def feature_detail(name: str, request: Request) -> dict:
    try:
        return request.app.state.analytics.feature_summary(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Feature not found or excluded from public analytics.") from exc


@app.post("/api/predict", response_model=PredictionResponse)
def predict(payload: StudentFeatures, request: Request) -> PredictionResponse:
    result = request.app.state.predictor.predict(validate_features(payload, request))
    return PredictionResponse(**{key: value for key, value in result.items() if key in PredictionResponse.model_fields})


@app.get("/api/explain/global")
def explain_global(request: Request) -> dict:
    summary = request.app.state.global_shap_summary
    if summary is None:
        raise HTTPException(status_code=503, detail="The precomputed SHAP summary is unavailable.")
    return {
        **summary,
        "dependence_plots": [
            {"feature": name, "path": f"/api/explain/global/dependence/{key}"}
            for key, (name, filename) in DEPENDENCE_PLOTS.items() if (FIGURES / filename).is_file()
        ],
    }


def shap_figure(filename: str) -> FileResponse:
    image = FIGURES / filename
    if not image.is_file():
        raise HTTPException(status_code=404, detail="The precomputed SHAP plot is unavailable.")
    return FileResponse(image, media_type="image/png")


@app.get("/api/explain/global/summary-plot")
def global_shap_plot() -> FileResponse:
    return shap_figure("global_shap_summary.png")


@app.get("/api/explain/global/dependence/{feature}")
def global_dependence_plot(feature: str) -> FileResponse:
    if feature not in DEPENDENCE_PLOTS:
        raise HTTPException(status_code=404, detail="SHAP dependence feature not found.")
    return shap_figure(DEPENDENCE_PLOTS[feature][1])


@app.post("/api/explain", response_model=ExplanationResponse)
def explain(payload: ExplanationRequest, request: Request) -> ExplanationResponse:
    features = validate_features(payload, request)
    slot = request.app.state.explanation_slot
    if not slot.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="An explanation is already running. Try again shortly.")
    try:
        result = request.app.state.explainer.explain(features)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="The explanation could not be verified.") from exc
    finally:
        slot.release()
    return ExplanationResponse(**result)
