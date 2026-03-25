"""
Tests for the FastAPI endpoints defined in app/api/main.py.

All ML models are replaced by lightweight stubs via the autouse fixture in
conftest.py, so no weights are loaded and no GCS access is needed.
"""
import pytest
from fastapi.testclient import TestClient

from app.api.main import api
from tests.conftest import make_image_bytes, HERB_TOP3, ILLNESS_TOP3


# ---------------------------------------------------------------------------
# Shared test client
# ---------------------------------------------------------------------------
@pytest.fixture
def client():
    with TestClient(api) as c:
        yield c


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------
class TestRoot:
    def test_returns_200(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_returns_hello_world(self, client):
        assert client.get("/").json() == {"message": "Hello World"}


# ---------------------------------------------------------------------------
# POST /predict_herb
# ---------------------------------------------------------------------------
class TestPredictHerb:
    def _post(self, client, suffix=".jpg"):
        img = make_image_bytes(suffix)
        return client.post(
            "/predict_herb",
            files={"file": (f"plant{suffix}", img, "image/jpeg")},
        )

    def test_status_200(self, client):
        assert self._post(client).status_code == 200

    def test_response_has_four_model_keys(self, client):
        data = self._post(client).json()
        assert set(data.keys()) == {"pytorch", "sklearn", "pytorch_large", "tensorflow"}

    def test_each_model_returns_three_predictions(self, client):
        data = self._post(client).json()
        for model_preds in data.values():
            assert len(model_preds) == 3

    def test_prediction_shape(self, client):
        data = self._post(client).json()
        for pred in data["pytorch"]:
            assert "species" in pred
            assert "confidence" in pred

    def test_prediction_values_match_stub(self, client):
        data = self._post(client).json()
        expected = [{"species": s, "confidence": c} for s, c in HERB_TOP3]
        assert data["pytorch"] == expected

    def test_accepts_png(self, client):
        assert self._post(client, suffix=".png").status_code == 200


# ---------------------------------------------------------------------------
# POST /predict_illness
# ---------------------------------------------------------------------------
class TestPredictIllness:
    def _post(self, client):
        img = make_image_bytes()
        return client.post(
            "/predict_illness",
            files={"file": ("plant.jpg", img, "image/jpeg")},
        )

    def test_status_200(self, client):
        assert self._post(client).status_code == 200

    def test_response_has_pytorch_key(self, client):
        data = self._post(client).json()
        assert "pytorch" in data
        assert len(data["pytorch"]) == 3

    def test_prediction_shape(self, client):
        for pred in self._post(client).json()["pytorch"]:
            assert "illness" in pred
            assert "confidence" in pred

    def test_prediction_values_match_stub(self, client):
        data = self._post(client).json()
        expected = [{"illness": s, "confidence": c} for s, c in ILLNESS_TOP3]
        assert data["pytorch"] == expected


# ---------------------------------------------------------------------------
# POST /predict-set
# ---------------------------------------------------------------------------
class TestPredictSet:
    def _post(self, client, n=2):
        img = make_image_bytes()
        files = [
            ("files", (f"img{i}.jpg", img, "image/jpeg"))
            for i in range(n)
        ]
        return client.post("/predict-set", files=files)

    def test_status_200(self, client):
        assert self._post(client).status_code == 200

    def test_returns_one_result_per_image(self, client):
        assert len(self._post(client, n=3).json()) == 3

    def test_result_has_filename(self, client):
        data = self._post(client, n=1).json()
        assert data[0]["filename"] == "img0.jpg"

    def test_result_has_all_model_keys(self, client):
        data = self._post(client, n=1).json()[0]
        assert set(data.keys()) == {"filename", "pytorch", "sklearn", "pytorch_large", "tensorflow"}

    def test_prediction_shape(self, client):
        pred = self._post(client, n=1).json()[0]["pytorch"]
        assert "species" in pred
        assert "confidence" in pred

    def test_prediction_values_match_stub(self, client):
        pred = self._post(client, n=1).json()[0]["pytorch"]
        assert pred == {"species": "Basilic", "confidence": 0.9}

    def test_single_image(self, client):
        assert len(self._post(client, n=1).json()) == 1


# ---------------------------------------------------------------------------
# POST /predict-set_illness
# ---------------------------------------------------------------------------
class TestPredictSetIllness:
    def _post(self, client, n=2):
        img = make_image_bytes()
        files = [
            ("files", (f"img{i}.jpg", img, "image/jpeg"))
            for i in range(n)
        ]
        return client.post("/predict-set_illness", files=files)

    def test_status_200(self, client):
        assert self._post(client).status_code == 200

    def test_returns_one_result_per_image(self, client):
        assert len(self._post(client, n=3).json()) == 3

    def test_result_has_filename(self, client):
        data = self._post(client, n=1).json()
        assert data[0]["filename"] == "img0.jpg"

    def test_prediction_shape(self, client):
        pred = self._post(client, n=1).json()[0]["pytorch"]
        assert "illness" in pred
        assert "confidence" in pred

    def test_prediction_values_match_stub(self, client):
        pred = self._post(client, n=1).json()[0]["pytorch"]
        assert pred == {"illness": "Healthy", "confidence": 0.85}
