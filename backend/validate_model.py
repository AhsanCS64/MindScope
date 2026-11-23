"""Verify the portable export and score the final refit on Kaggle's original test split."""
from __future__ import annotations

import json
from zipfile import ZipFile

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from app.services.analytics import DatasetAnalytics
from app.services.predictor import ARTIFACTS, ROOT, EnsemblePredictor, load_dataset

# Saved output of execution 62 in student-depression.ipynb, in source row order.
REFERENCE = [
    0.27652499079704285, 0.13792599737644196, 0.14135700464248657, 0.7749750018119812,
    0.31199800968170166, 0.08668100088834763, 0.09542199940776825, 0.10367099940776825,
    0.749068021774292, 0.47008898854255676,
]
# Displayed to six decimals in the final refit's wiring check (execution 53).
REFIT_CHECK = [0.927191, 0.059064, 0.988026, 0.979713, 0.839578]
TOLERANCE = 5e-5


def ensemble_probabilities(predictor: EnsemblePredictor, examples: pd.DataFrame) -> np.ndarray:
    frame = examples.loc[:, predictor.feature_columns].copy()
    for name in predictor.categorical_columns:
        frame[name] = frame[name].fillna("__MISSING__").astype(str)
    member_scores = [
        booster.predict(xgb.DMatrix(preprocessor.transform(frame), nthread=1))
        for preprocessor, booster in zip(predictor.preprocessors, predictor.boosters)
    ]
    return np.mean(np.stack(member_scores).astype(np.float64), axis=0)


def main() -> None:
    predictor = EnsemblePredictor()
    frame = load_dataset()
    with ZipFile(ROOT / "mindscope_portable.zip") as source:
        for name in ["manifest.json", *predictor.manifest["boosters"], *predictor.manifest["preprocessors"]]:
            if name == "manifest.json":
                assert json.loads(source.read(name)) == predictor.manifest
            else:
                assert source.read(name) == (ARTIFACTS / name).read_bytes(), name
    for preprocessor, booster in zip(predictor.preprocessors, predictor.boosters):
        assert booster.num_boosted_rounds() == predictor.metadata["n_estimators"]
        assert preprocessor.feature_names_in_.tolist() == predictor.feature_columns

    actual = [
        predictor.predict(row)["probability_yes"]
        for row in frame.loc[:, predictor.feature_columns].head(len(REFERENCE)).to_dict("records")
    ]
    max_reference_error = float(np.max(np.abs(np.asarray(actual) - REFERENCE)))
    assert max_reference_error <= TOLERANCE, max_reference_error

    # Match the notebook's student filter, full-row deduplication, row order,
    # class encoding and both stratified splits (execution 44).
    population = DatasetAnalytics(frame, predictor).model_population
    target = population[predictor.metadata["target_column"]].astype(str).str.strip().astype(int)
    development, test, y_dev, y_test = train_test_split(
        population, target, test_size=0.20, random_state=42, stratify=target,
    )
    train, validation, _, _ = train_test_split(
        development, y_dev, test_size=0.25, random_state=42, stratify=y_dev,
    )
    assert (len(train), len(validation), len(test)) == (16722, 5574, 5574)
    assert y_test.value_counts().to_dict() == {0: 2312, 1: 3262}
    refit_error = float(np.max(np.abs(ensemble_probabilities(predictor, validation.head(5)) - REFIT_CHECK)))
    assert refit_error <= TOLERANCE, refit_error

    scores = ensemble_probabilities(predictor, test)
    predictions = (scores >= predictor.threshold).astype(int)
    observed, predicted = calibration_curve(y_test, scores, n_bins=10, strategy="quantile")
    held_out = {
        "sample_count": len(test),
        "threshold": round(predictor.threshold, 2),
        "accuracy": accuracy_score(y_test, predictions),
        "yes_precision": precision_score(y_test, predictions, zero_division=0),
        "yes_recall": recall_score(y_test, predictions, zero_division=0),
        "yes_f1": f1_score(y_test, predictions, zero_division=0),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
        "roc_auc": roc_auc_score(y_test, scores),
        "pr_auc": average_precision_score(y_test, scores),
        "brier": brier_score_loss(y_test, scores),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "calibration": [
            {"mean_prediction": float(p), "observed_yes_rate": float(o)}
            for p, o in zip(predicted, observed)
        ],
    }
    evidence = {
        "status": "held_out_verified",
        "message": "The exported ensemble was scored on the original stratified holdout after its refit on training and validation data.",
        "provenance": {
            "notebook": "student-depression.ipynb",
            "split_seed": 42,
            "train_rows": len(train),
            "validation_rows": len(validation),
            "refit_rows": len(development),
            "test_rows": len(test),
        },
        "held_out": held_out,
        "notebook_evaluation": {
            "description": "The notebook scored the train-only tuned ensemble before the final refit.",
            "validation_macro_f1": 0.795653,
            "test_accuracy": 0.7910,
            "test_macro_f1": 0.7851,
            "test_roc_auc": 0.8705,
            "test_pr_auc": 0.8967,
            "test_brier": 0.1420,
        },
        "verification": {
            "reference_probability": REFERENCE[0],
            "reproduced_probability": actual[0],
            "absolute_error": abs(actual[0] - REFERENCE[0]),
            "maximum_reference_error": max_reference_error,
            "reference_profiles": len(REFERENCE),
            "refit_reference_profiles": len(REFIT_CHECK),
            "maximum_refit_reference_error": refit_error,
            "tolerance": TOLERANCE,
        },
    }
    payload = json.dumps(evidence, indent=2, allow_nan=False) + "\n"
    staging = ARTIFACTS / "model_metrics.json.tmp"
    staging.write_text(payload, encoding="utf-8")
    staging.replace(ARTIFACTS / "model_metrics.json")
    print(payload)


if __name__ == "__main__":
    main()

