import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402


def test_health_and_predict():
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        response = client.post(
            "/predict", json={"formulas": ["CrI3", "not a formula", "Fe2O3"]}
        )
        assert response.status_code == 200
        body = response.json()
        assert body[0]["magnetic_score"] is not None
        assert body[1]["error"] == "could not parse formula"
        assert len(body) == 3


def test_model_endpoint_reports_test_metrics():
    with TestClient(app) as client:
        body = client.get("/model").json()
        assert body["model_version"] == "v3.0_v2db_composition_only"
        assert 0.0 < body["threshold"] < 1.0
        assert "roc_auc" in body["test_metrics"]
