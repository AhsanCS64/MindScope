from __future__ import annotations

import json
import threading
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import HTTPException


ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "artifacts"
METADATA_FILE = ARTIFACTS / "mindscope_openml_xgb_ensemble_metadata.json"
MANIFEST_FILE = ARTIFACTS / "manifest.json"
DISCLAIMER = (
    "Educational demonstration only. This is an estimate of the dataset's recorded "
    "label, not a diagnosis, clinical screening, or treatment recommendation."
)


class EnsemblePredictor:
    """Loads the portable, JSON-serialized five-member XGBoost ensemble once."""

    def __init__(self, artifacts_dir: Path = ARTIFACTS) -> None:
        self.artifacts_dir = artifacts_dir
        self._lock = threading.RLock()
        self.manifest = json.loads((artifacts_dir / "manifest.json").read_text(encoding="utf-8"))
        self.metadata = json.loads(
            (artifacts_dir / "mindscope_openml_xgb_ensemble_metadata.json").read_text(encoding="utf-8")
        )
        self.feature_columns: list[str] = self.manifest["feature_columns"]
        if self.feature_columns != self.metadata["feature_columns"]:
            raise ValueError("Portable artifact feature columns do not match model metadata.")
        self.threshold = float(self.metadata["threshold"])
        if not np.isclose(self.threshold, float(self.manifest["threshold"])):
            raise ValueError("Portable artifact threshold does not match model metadata.")
        if self.metadata.get("class_mapping") != {"No": 0, "Yes": 1}:
            raise ValueError("Unsupported model class mapping.")
        self.categorical_columns = list(self.metadata["categorical_columns"])
        self.preprocessors = [joblib.load(artifacts_dir / name) for name in self.manifest["preprocessors"]]
        self.boosters: list[xgb.Booster] = []
        for name in self.manifest["boosters"]:
            booster = xgb.Booster(params={"nthread": 1})
            booster.load_model(str(artifacts_dir / name))
            booster.set_param({"nthread": 1, "device": "cpu"})
            self.boosters.append(booster)
        if len(self.boosters) != 5 or len(self.preprocessors) != 5:
            raise ValueError("Expected exactly five preprocessors and five XGBoost members.")
        self.model_version = (
            f"OpenML-{self.metadata['openml_dataset_id']}-XGBoost-5-seed"
        )
        self.categories = self._extract_categories()
        self._validate_artifacts()

    def _extract_categories(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        preprocessor = self.preprocessors[0]
        for _, transformer, columns in preprocessor.transformers_:
            if not hasattr(transformer, "named_steps"):
                continue
            encoder = next(
                (step for step in transformer.named_steps.values() if hasattr(step, "categories_")),
                None,
            )
            if encoder is None:
                continue
            for column, values in zip(columns, encoder.categories_):
                result[str(column)] = [str(value) for value in values]
        return result

    def _validate_artifacts(self) -> None:
        if set(self.categories) != set(self.categorical_columns):
            raise ValueError("Could not recover all categorical options from the fitted encoders.")
        first_categories = self.categories
        for preprocessor in self.preprocessors[1:]:
            columns: dict[str, list[str]] = {}
            for _, transformer, input_columns in preprocessor.transformers_:
                if not hasattr(transformer, "named_steps"):
                    continue
                encoder = next(
                    (step for step in transformer.named_steps.values() if hasattr(step, "categories_")),
                    None,
                )
                if encoder is not None:
                    columns.update({str(c): [str(v) for v in values]
                                    for c, values in zip(input_columns, encoder.categories_)})
            if columns != first_categories:
                raise ValueError("Ensemble members do not share the same fitted category schema.")

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        missing = [column for column in self.feature_columns if column not in features]
        extra = [column for column in features if column not in self.feature_columns]
        if missing or extra:
            raise HTTPException(
                status_code=422,
                detail={"missing_features": missing, "unexpected_features": extra},
            )
        for column in self.categorical_columns:
            value = features[column]
            if str(value) not in self.categories[column]:
                raise HTTPException(
                    status_code=422,
                    detail={"field": column, "message": "Choose a category supported by the fitted model."},
                )

        frame = pd.DataFrame([{column: features[column] for column in self.feature_columns}],
                             columns=self.feature_columns)
        member_probabilities: list[float] = []
        started = perf_counter()
        with self._lock:
            for preprocessor, booster in zip(self.preprocessors, self.boosters):
                transformed = preprocessor.transform(frame)
                probability_yes = float(booster.predict(xgb.DMatrix(transformed, nthread=1))[0])
                if not np.isfinite(probability_yes) or not 0 <= probability_yes <= 1:
                    raise RuntimeError("The model returned an invalid probability.")
                member_probabilities.append(probability_yes)
        probability_yes = float(np.mean(member_probabilities))
        return {
            "probability_yes": probability_yes,
            "probability_no": 1.0 - probability_yes,
            "dataset_label_estimate": "Yes" if probability_yes >= self.threshold else "No",
            "threshold": self.threshold,
            "model_version": self.model_version,
            "explanation_available": True,
            "disclaimer": DISCLAIMER,
            "inference_ms": (perf_counter() - started) * 1000,
            "member_probabilities": member_probabilities,
        }


def load_dataset() -> pd.DataFrame:
    path = ROOT / "data" / "raw" / "student_depression_openml_46874.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {path}")
    return pd.read_csv(path)
