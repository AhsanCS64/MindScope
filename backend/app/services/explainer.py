from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from scipy import sparse

from .predictor import EnsemblePredictor


class EnsembleExplainer:
    """Probability-space SHAP for the deployed mean-probability ensemble."""

    def __init__(self, predictor: EnsemblePredictor, population: pd.DataFrame) -> None:
        self.predictor = predictor
        self._lock = predictor._lock
        self.population = population.loc[:, predictor.feature_columns].copy()
        self.background = self.population.sample(n=min(64, len(self.population)), random_state=42)
        self._explainers: list[shap.TreeExplainer] | None = None
        self._feature_to_raw: list[str] | None = None

    @staticmethod
    def _shap_dense(values: Any) -> np.ndarray:
        """Preserve missing positions when converting a sparse frame for SHAP."""
        if not sparse.issparse(values):
            return np.asarray(values, dtype=float)
        entries = values.tocoo()
        dense = np.full(entries.shape, np.nan)
        dense[entries.row, entries.col] = entries.data
        return dense

    def _feature_mapping(self, preprocessor: Any) -> list[str]:
        mapped: list[str] = []
        for _, transformer, columns in preprocessor.transformers_:
            if transformer == "drop" or transformer is None:
                continue
            if hasattr(transformer, "named_steps"):
                encoder = next(
                    (step for step in transformer.named_steps.values() if hasattr(step, "categories_")),
                    None,
                )
                if encoder is None:
                    raise ValueError("Categorical preprocessing does not expose fitted category names.")
                mapped.extend(
                    str(column)
                    for column, categories in zip(columns, encoder.categories_)
                    for _ in categories
                )
            else:
                mapped.extend(str(column) for column in columns)
        expected = len(preprocessor.get_feature_names_out())
        if len(mapped) != expected:
            raise ValueError(f"Could not map all {expected} transformed features to original columns.")
        return mapped

    def _load_explainers(self) -> None:
        if self._explainers is not None:
            return
        explainers: list[shap.TreeExplainer] = []
        with self._lock:
            for preprocessor, booster in zip(self.predictor.preprocessors, self.predictor.boosters):
                background = self._shap_dense(preprocessor.transform(self.background))
                explainers.append(
                    shap.TreeExplainer(
                        booster,
                        data=background,
                        feature_perturbation="interventional",
                        model_output="probability",
                    )
                )
        self._feature_to_raw = self._feature_mapping(self.predictor.preprocessors[0])
        self._explainers = explainers

    def explain(self, features: dict[str, Any]) -> dict[str, Any]:
        self._load_explainers()
        row = pd.DataFrame([{name: features[name] for name in self.predictor.feature_columns}],
                           columns=self.predictor.feature_columns)
        per_member_values: list[np.ndarray] = []
        per_member_bases: list[float] = []
        with self._lock:
            for preprocessor, explainer in zip(self.predictor.preprocessors, self._explainers or []):
                transformed = self._shap_dense(preprocessor.transform(row))
                explanation = explainer(transformed, check_additivity=False)
                values = np.asarray(explanation.values, dtype=float).reshape(-1)
                base = float(np.asarray(explanation.base_values).reshape(-1)[0])
                per_member_values.append(values)
                per_member_bases.append(base)
        mean_values = np.mean(np.vstack(per_member_values), axis=0)
        mean_base = float(np.mean(per_member_bases))
        result = self.predictor.predict(features)
        model_output = float(result["probability_yes"])
        reconstructed = mean_base + float(mean_values.sum())
        error = abs(model_output - reconstructed)
        if error > 2e-4:
            raise RuntimeError(f"SHAP probability-space contributions failed additivity check ({error:.6g}).")

        grouped: dict[str, float] = {}
        for name, value in zip(self._feature_to_raw or [], mean_values):
            grouped[name] = grouped.get(name, 0.0) + float(value)
        contributions = [
            {"feature": name, "contribution": value, "direction": "increases estimate" if value >= 0 else "decreases estimate"}
            for name, value in sorted(grouped.items(), key=lambda item: abs(item[1]), reverse=True)
        ]
        return {
            "available": True,
            "message": "SHAP explains this model's behavior for this input; it does not identify causes.",
            "base_value": mean_base,
            "model_output": model_output,
            "output_scale": "probability of the dataset's recorded Yes label",
            "additivity_error": error,
            "contributions": contributions,
        }

    def global_summary(self, sample_size: int = 200) -> dict[str, Any]:
        self._load_explainers()
        sample = self.population.sample(n=min(sample_size, len(self.population)), random_state=43)
        shap_values = []
        base_values = []
        with self._lock:
            for preprocessor, explainer in zip(self.predictor.preprocessors, self._explainers or []):
                transformed = self._shap_dense(preprocessor.transform(sample))
                explanation = explainer(transformed, check_additivity=False)
                shap_values.append(np.asarray(explanation.values, dtype=float))
                base_values.append(np.asarray(explanation.base_values, dtype=float).reshape(-1))
        mean_values = np.mean(np.stack(shap_values), axis=0)
        mean_bases = np.mean(np.stack(base_values), axis=0)
        member_probabilities = []
        for preprocessor, booster in zip(self.predictor.preprocessors, self.predictor.boosters):
            member_probabilities.append(
                booster.predict(xgb.DMatrix(preprocessor.transform(sample), nthread=1))
            )
        expected_outputs = np.mean(np.vstack(member_probabilities), axis=0)
        additivity_error = float(np.max(np.abs(mean_bases + mean_values.sum(axis=1) - expected_outputs)))
        if additivity_error > 2e-4:
            raise RuntimeError(f"Global SHAP probability-space additivity failed ({additivity_error:.6g}).")
        raw_features = list(self.predictor.feature_columns)
        grouped_values = np.zeros((len(sample), len(raw_features)), dtype=float)
        positions = {name: i for i, name in enumerate(raw_features)}
        for index, raw_name in enumerate(self._feature_to_raw or []):
            grouped_values[:, positions[raw_name]] += mean_values[:, index]
        grouped = {
            name: float(np.mean(np.abs(grouped_values[:, index])))
            for index, name in enumerate(raw_features)
        }
        importance = sorted(grouped.items(), key=lambda item: item[1], reverse=True)
        return {
            "output_scale": "probability of the dataset's recorded Yes label",
            "method": "Mean probability-space SHAP contributions across all five ensemble members; 64-row background sample.",
            "sample_size": int(len(sample)),
            "maximum_additivity_error": additivity_error,
            "features": [{"feature": name, "mean_absolute_shap": value} for name, value in importance],
            "warning": "These values explain model behavior, not causes or clinical risk.",
            "_sample": sample,
            "_transformed_shap": mean_values,
            "_transformed_data": self._shap_dense(self.predictor.preprocessors[0].transform(sample)),
            "_transformed_feature_names": self.predictor.preprocessors[0].get_feature_names_out().tolist(),
        }
