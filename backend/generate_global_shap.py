from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from app.services.analytics import DatasetAnalytics
from app.services.explainer import EnsembleExplainer
from app.services.predictor import EnsemblePredictor, load_dataset


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "reports" / "figures"


def main() -> None:
    predictor = EnsemblePredictor()
    analytics = DatasetAnalytics(load_dataset(), predictor)
    explainer = EnsembleExplainer(predictor, analytics.model_population)
    result = explainer.global_summary(sample_size=200)

    public_result = {key: value for key, value in result.items() if not key.startswith("_")}
    (ARTIFACTS / "global_shap.json").write_text(
        json.dumps(public_result, indent=2), encoding="utf-8"
    )

    values = result["_transformed_shap"]
    data = result["_transformed_data"]
    names = result["_transformed_feature_names"]
    FIGURES.mkdir(parents=True, exist_ok=True)

    shap.summary_plot(values, data, feature_names=names, max_display=20, show=False)
    plt.gcf().set_size_inches(11, 8)
    plt.title("MindScope global SHAP summary\nMean contribution to the dataset-label probability")
    plt.tight_layout()
    plt.savefig(FIGURES / "global_shap_summary.png", dpi=180, bbox_inches="tight")
    plt.close("all")

    mean_abs = abs(values).mean(axis=0)
    top_indexes = mean_abs.argsort()[::-1][:3]
    for index in top_indexes:
        shap.dependence_plot(
            int(index),
            values,
            data,
            feature_names=names,
            interaction_index=None,
            show=False,
        )
        plt.title(f"SHAP dependence: {names[index]}")
        plt.tight_layout()
        safe_name = "".join(ch if ch.isalnum() else "_" for ch in names[index]).strip("_")
        plt.savefig(FIGURES / f"dependence_{safe_name}.png", dpi=180, bbox_inches="tight")
        plt.close("all")

    print(json.dumps(public_result, indent=2))
    print(f"Saved {ARTIFACTS / 'global_shap.json'}")
    print(f"Saved SHAP plots under {FIGURES}")


if __name__ == "__main__":
    main()
