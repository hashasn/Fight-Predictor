import json

import pytest
from fastapi.testclient import TestClient

import app as app_module

client = TestClient(app_module.app)


# /health

def test_health():
    """
    health check should always return ok
    """
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# /predictions/upcoming and /predictions/archive

def test_predictions_endpoints(tmp_path, monkeypatch):
    """
    upcoming and archive should return what's in the predictions store.
    if the store file doesn't exist yet, both should return empty lists
    """
    store_path = tmp_path / "predictions_store.json"

    # point the app at a temporary store file instead of the real one
    monkeypatch.setattr(app_module, "STORE_PATH", store_path)

    # no file yet -> empty lists
    assert client.get("/predictions/upcoming").json() == []
    assert client.get("/predictions/archive").json() == []

    # write a store file with one upcoming and one archived fight
    store = {
        "upcoming": [
            {
                "fighter_a": "Joshua Van",
                "fighter_b": "Alexandre Pantoja",
                "fighter_a_win_prob": 0.791,
                "fighter_b_win_prob": 0.209,
                "event": "UFC 331: Van vs. Pantoja 2",
                "date": "September 19, 2026",
            }
        ],
        "archive": [
            {
                "fighter_a": "Jean Silva",
                "fighter_b": "Jose Delgado",
                "fighter_a_win_prob": 0.61,
                "fighter_b_win_prob": 0.39,
                "predicted_winner": "Jean Silva",
                "actual_winner": "Jean Silva",
                "method": "Submission",
                "correct": True,
            }
        ],
    }
    store_path.write_text(json.dumps(store))

    upcoming = client.get("/predictions/upcoming")
    assert upcoming.status_code == 200
    assert upcoming.json() == store["upcoming"]

    archive = client.get("/predictions/archive")
    assert archive.status_code == 200
    assert archive.json() == store["archive"]


    # /predict

@pytest.fixture
def fake_model(monkeypatch):
    """
    skip loading the real model and building the real bundle,
    so these tests only check the endpoint itself
    """
    monkeypatch.setattr(app_module.predict_module, "reload_artifacts", lambda: None)
    monkeypatch.setattr(app_module, "build_bundle", lambda: {})


def test_predict_success(fake_model, monkeypatch):
    """
    a valid matchup should return 200 with both win probabilities
    """
    calls = []

    def fake_predict_matchup(fighter_a, fighter_b, bundle):
        calls.append((fighter_a, fighter_b))
        return {
            "fighter_a": fighter_a,
            "fighter_b": fighter_b,
            "fighter_a_win_prob": 0.791,
            "fighter_b_win_prob": 0.209,
        }

    monkeypatch.setattr(app_module.predict_module, "predict_matchup", fake_predict_matchup)

    response = client.post("/predict", json={
        "fighter_a": "Joshua Van",
        "fighter_b": "Alexandre Pantoja",
    })

    assert response.status_code == 200
    assert response.json() == {
        "fighter_a": "Joshua Van",
        "fighter_b": "Alexandre Pantoja",
        "fighter_a_win_prob": 0.791,
        "fighter_b_win_prob": 0.209,
    }

    # the names from the request were passed to the predictor
    assert calls == [("Joshua Van", "Alexandre Pantoja")]


def test_predict_unknown_fighter(fake_model, monkeypatch):
    """
    a fighter with no history should return 404 with the error message
    """
    def fake_predict_matchup(fighter_a, fighter_b, bundle):
        raise ValueError(f"No history found for fighter: {fighter_a}")

    monkeypatch.setattr(app_module.predict_module, "predict_matchup", fake_predict_matchup)

    response = client.post("/predict", json={
        "fighter_a": "Nobody",
        "fighter_b": "Joshua Van",
    })

    assert response.status_code == 404
    assert response.json()["detail"] == "No history found for fighter: Nobody"


def test_predict_missing_fighter():
    """
    a request missing a fighter should be rejected before reaching the model
    """
    response = client.post("/predict", json={"fighter_a": "Joshua Van"})

    assert response.status_code == 422