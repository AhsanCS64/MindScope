from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def sample_payload():
    from app.services.predictor import ROOT

    frame = pd.read_csv(ROOT / "data" / "raw" / "student_depression_openml_46874.csv")
    columns = [
            "Gender", "Age", "City", "Profession", "Academic Pressure", "Work Pressure",
            "CGPA", "Study Satisfaction", "Job Satisfaction", "Sleep Duration",
            "Dietary Habits", "Degree", "Work/Study Hours", "Financial Stress",
            "Family History of Mental Illness",
        ]
    return frame.loc[0, columns].to_dict()


def test_health_and_model_info(client):
    health = client.get("/api/health")
    info = client.get("/api/model/info")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert info.status_code == 200
    assert info.json()["threshold"] == pytest.approx(0.56)
    assert len(info.json()["feature_columns"]) == 15
    assert "Have you ever had suicidal thoughts ?" not in info.json()["feature_columns"]


def test_prediction_matches_kaggle_reference(client, sample_payload):
    response = client.post("/api/predict", json=sample_payload)
    assert response.status_code == 200
    result = response.json()
    assert result["probability_yes"] == pytest.approx(0.27652499, abs=5e-5)
    assert result["probability_yes"] + result["probability_no"] == pytest.approx(1.0)
    assert result["dataset_label_estimate"] == "No"
    assert result["threshold"] == pytest.approx(0.56)


def test_invalid_and_sensitive_inputs_are_rejected(client, sample_payload):
    invalid = {**sample_payload, "City": "Unsupported invented city"}
    assert client.post("/api/predict", json=invalid).status_code == 422
    sensitive = {**sample_payload, "Have you ever had suicidal thoughts ?": "Yes"}
    assert client.post("/api/predict", json=sensitive).status_code == 422


def test_dataset_analytics_are_real_and_exclude_sensitive_column(client):
    summary = client.get("/api/dataset/summary")
    features = client.get("/api/analytics/features")
    assert summary.status_code == 200
    assert summary.json()["source_rows"] == 27901
    assert summary.json()["model_population_rows"] == 27870
    names = [item["name"] for item in features.json()["features"]]
    assert len(names) == 15
    assert "Have you ever had suicidal thoughts ?" not in names


def test_metrics_reports_provenance(client):
    response = client.get("/api/model/metrics")
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "held_out_verified"
    assert result["verification"]["reference_profiles"] == 10
    assert result["verification"]["maximum_reference_error"] < result["verification"]["tolerance"]
    held_out = result["held_out"]
    assert held_out["sample_count"] == 5574
    assert held_out["threshold"] == pytest.approx(0.56)
    assert held_out["confusion_matrix"] == [[1747, 565], [597, 2665]]
    assert held_out["roc_auc"] == pytest.approx(0.871382, abs=1e-6)
    assert len(held_out["calibration"]) == 10
    assert result["notebook_evaluation"]["test_macro_f1"] == pytest.approx(0.7851)



def test_local_shap_explanation_reconstructs_probability(client, sample_payload):
    response = client.post("/api/explain", json=sample_payload)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["available"] is True
    assert len(result["contributions"]) == 15
    assert result["additivity_error"] < 2e-4
    assert result["base_value"] + sum(x["contribution"] for x in result["contributions"]) == pytest.approx(
        result["model_output"], abs=2e-4
    )


def test_global_shap_summary_and_plot_exist(client):
    summary = client.get("/api/explain/global")
    plot = client.get("/api/explain/global/summary-plot")
    assert summary.status_code == 200
    assert summary.json()["sample_size"] == 200
    assert summary.json()["maximum_additivity_error"] < 2e-4
    assert plot.status_code == 200
    assert plot.headers["content-type"] == "image/png"
    for item in summary.json()["dependence_plots"]:
        assert client.get(item["path"]).headers["content-type"] == "image/png"
    assert len(summary.json()["dependence_plots"]) == 3
    assert client.get("/api/explain/global/dependence/unknown").status_code == 404

def test_model_rejects_missing_outlier_and_wrong_type_features(client, sample_payload):
    missing = {key: value for key, value in sample_payload.items() if key != "Age"}
    inputs = (missing, {**sample_payload, "Age": -1}, {**sample_payload, "Age": "20"})
    for path in ("/api/predict", "/api/explain"):
        for invalid in inputs:
            response = client.post(path, json=invalid)
            assert response.status_code == 422
    invalid = {**sample_payload, "Age": "PRIVATE_SENTINEL"}
    assert "PRIVATE_SENTINEL" not in client.post("/api/predict", json=invalid).text


def test_concurrent_predictions_are_consistent(client, sample_payload):
    def predict(_):
        response = client.post("/api/predict", json=sample_payload)
        assert response.status_code == 200
        return response.json()

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(predict, range(8)))
    assert all(result == results[0] for result in results)


def test_public_distributions_and_cors(client):
    features = client.get("/api/analytics/features").json()["features"]
    summary = client.get("/api/dataset/summary").json()
    assert sum(summary["target_counts"].values()) == summary["model_population_rows"]
    for feature in features:
        present = sum(item["count"] for item in feature["distribution"] if item["category"] != "<missing>")
        assert present + feature["missing_count"] == summary["model_population_rows"]
    response = client.options("/api/predict", headers={
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type",
    })
    assert response.headers["Access-Control-Allow-Origin"] == "http://localhost:5173"
