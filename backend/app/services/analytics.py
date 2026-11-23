from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .predictor import EnsemblePredictor


SENSITIVE_COLUMNS = {"Have you ever had suicidal thoughts ?"}


class DatasetAnalytics:
    def __init__(self, frame: pd.DataFrame, predictor: EnsemblePredictor) -> None:
        self.source_rows = len(frame)
        self.predictor = predictor
        profession = next((c for c in frame.columns if c.casefold() == "profession"), None)
        if profession:
            is_student = frame[profession].astype(str).str.strip().str.casefold().eq("student")
            self.model_population = frame.loc[is_student].copy()
        else:
            self.model_population = frame.copy()
        self.model_population = self.model_population.drop_duplicates().reset_index(drop=True)
        self.visible_features = [
            column for column in predictor.feature_columns
            if column not in SENSITIVE_COLUMNS
        ]

    def summary(self) -> dict[str, Any]:
        target = self.model_population[self.predictor.metadata["target_column"]].astype(str).str.strip()
        target_counts = target.value_counts(dropna=False)
        mapping = {str(value): str(label) for label, value in self.predictor.metadata["class_mapping"].items()}
        missing = {
            str(column): int(self.model_population[column].isna().sum())
            for column in self.visible_features + [self.predictor.metadata["target_column"]]
        }
        return {
            "dataset": "OpenML Student Depression Dataset",
            "openml_dataset_id": self.predictor.metadata["openml_dataset_id"],
            "source_rows": self.source_rows,
            "model_population_rows": int(len(self.model_population)),
            "model_feature_count": len(self.predictor.feature_columns),
            "target_column": self.predictor.metadata["target_column"],
            "target_counts": {mapping.get(str(label), str(label)): int(count) for label, count in target_counts.items()},
            "missing_by_column": missing,
            "duplicate_rows_in_model_population": int(self.model_population.duplicated().sum()),
            "sensitive_predictor_excluded_from_app": True,
        }

    def features(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        frame = self.model_population
        for column in self.visible_features:
            values = frame[column]
            item: dict[str, Any] = {
                "name": column,
                "kind": "numeric" if pd.api.types.is_numeric_dtype(values) else "categorical",
                "missing_count": int(values.isna().sum()),
                "unique_count": int(values.nunique(dropna=True)),
            }
            if item["kind"] == "numeric":
                desc = values.describe()
                item["statistics"] = {
                    key: float(desc[key]) for key in ("mean", "std", "min", "25%", "50%", "75%", "max")
                }
                present = values.dropna()
                if present.nunique() <= 15:
                    counts = present.value_counts().sort_index()
                    item["distribution"] = [
                        {"category": f"{value:g}", "count": int(count)} for value, count in counts.items()
                    ]
                else:
                    counts, edges = np.histogram(present, bins=10)
                    item["distribution"] = [
                        {"category": f"{left:.2f} to {right:.2f}", "count": int(count)}
                        for left, right, count in zip(edges, edges[1:], counts)
                    ]
            else:
                counts = values.fillna("<missing>").astype(str).value_counts()
                item["distribution"] = [
                    {"category": str(label), "count": int(count)} for label, count in counts.items()
                ]
            result.append(item)
        return result

    def feature_summary(self, name: str) -> dict[str, Any]:
        if name not in self.visible_features:
            raise KeyError(name)
        values = self.model_population[name]
        if pd.api.types.is_numeric_dtype(values):
            return self.features_by_name(name)
        counts = values.fillna("<missing>").astype(str).value_counts()
        return {
            "name": name,
            "kind": "categorical",
            "distribution": [{"category": str(k), "count": int(v)} for k, v in counts.items()],
        }

    def correlations(self) -> dict[str, Any]:
        numeric = [
            name for name in self.visible_features
            if pd.api.types.is_numeric_dtype(self.model_population[name])
        ]
        matrix = self.model_population[numeric].corr()
        return {
            "features": numeric,
            "matrix": {name: {column: float(value) if pd.notna(value) else None for column, value in row.items()}
                       for name, row in matrix.to_dict().items()},
            "interpretation": "Correlations describe associations in this dataset and do not establish causes.",
        }

    def features_by_name(self, name: str) -> dict[str, Any]:
        return next(feature for feature in self.features() if feature["name"] == name)
