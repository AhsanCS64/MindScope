"""Measure CPU startup and inference locally with the supplied model."""
import json
from statistics import median
from time import perf_counter

START = perf_counter()

from app.services.analytics import DatasetAnalytics
from app.services.explainer import EnsembleExplainer
from app.services.predictor import ARTIFACTS, EnsemblePredictor, load_dataset


def main() -> None:
    predictor = EnsemblePredictor()
    dataset = load_dataset()
    analytics = DatasetAnalytics(dataset, predictor)
    explainer = EnsembleExplainer(predictor, analytics.model_population)
    analytics.summary()
    analytics.features()
    analytics.correlations()
    startup_ms = (perf_counter() - START) * 1000

    profile = dataset.loc[0, predictor.feature_columns].to_dict()
    predictions = [predictor.predict(profile)["inference_ms"] for _ in range(21)]
    explanations = []
    result = None
    for _ in range(6):
        started = perf_counter()
        result = explainer.explain(profile)
        explanations.append((perf_counter() - started) * 1000)
    files = predictor.manifest["boosters"] + predictor.manifest["preprocessors"]
    stats = {
        "model_bytes": sum((ARTIFACTS / file).stat().st_size for file in files),
        "startup_ms": startup_ms,
        "first_prediction_ms": predictions[0],
        "median_warm_prediction_ms": median(predictions[1:]),
        "first_explanation_ms": explanations[0],
        "median_warm_explanation_ms": median(explanations[1:]),
        "local_shap_additivity_error": result["additivity_error"],
    }
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

