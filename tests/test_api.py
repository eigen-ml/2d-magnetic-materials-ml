from fastapi.testclient import TestClient

from api.main import app


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
