import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
PREDICTOR_DIR = REPO_ROOT / "services" / "predictor"


@pytest.fixture
def predictor_app(tmp_path, monkeypatch):
    """Import predictor.py fresh, pointed at an empty temp DATA_DIR, so
    tests don't touch the real trained artifacts under data/."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.syspath_prepend(str(PREDICTOR_DIR))
    sys.modules.pop("predictor", None)
    import predictor as predictor_module

    return predictor_module


@pytest.fixture
def client(predictor_app):
    return TestClient(predictor_app.app)


def test_root_endpoint(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["message"] == "Football Predictor API"


def test_status_reports_untrained_when_no_trace(client):
    resp = client.get("/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_trained"] is False
    assert body["last_modified"] is None


def test_predict_without_training_returns_404(client):
    resp = client.get("/predict")
    assert resp.status_code == 404


def test_gameweeks_empty_dir_returns_empty_list(client):
    resp = client.get("/gameweeks")
    assert resp.status_code == 200
    assert resp.json() == {"rounds": []}


def test_gameweeks_missing_round_returns_404(client):
    resp = client.get("/gameweeks/999/predictions")
    assert resp.status_code == 404


def test_rate_limiter_allows_up_to_threshold_then_blocks(predictor_app):
    predictor_app.request_counts.clear()
    client_id = "unit-test-client"

    for _ in range(predictor_app.MAX_REQUESTS_PER_WINDOW):
        assert predictor_app.check_rate_limit(client_id) is True

    assert predictor_app.check_rate_limit(client_id) is False


def test_gameweeks_endpoint_exempt_from_rate_limit(client, predictor_app):
    # /gameweeks is explicitly carved out of the rate limiter in the
    # middleware, so it should never 429 no matter how many times it's hit.
    for _ in range(predictor_app.MAX_REQUESTS_PER_WINDOW + 5):
        resp = client.get("/gameweeks")
        assert resp.status_code == 200
