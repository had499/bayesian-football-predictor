import pickle
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
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


def _make_synthetic_df_with_future_round(seed=0):
    """One season, 4 teams, a handful of past rounds plus one future round —
    enough for /predict to have something to predict against `datetime.today()`."""
    from football_model.features.add_metadata import (
        add_rounds_to_data, add_match_ids, add_home_away_goals_xg,
    )

    rng = np.random.default_rng(seed)
    rows = []
    teams = ["A", "B", "C", "D"]
    dt = pd.Timestamp.today().normalize() - timedelta(days=6 * 7)
    for rnd in range(1, 9):  # round 8 lands clearly ~1 week in the future
        for i in range(0, len(teams), 2):
            home, away = teams[i], teams[i + 1]
            gh, ga = int(rng.poisson(1.3)), int(rng.poisson(1.0))
            for team, opp, is_home, goals, goals_against in [
                (home, away, 1, gh, ga), (away, home, 0, ga, gh),
            ]:
                rows.append(dict(
                    team=team, opp_team=opp, is_home=is_home, goals=goals,
                    goals_against=goals_against, xG=goals + 0.1, xGA=goals_against + 0.1,
                    datetime=dt, season="2025", round=rnd,
                ))
        dt += timedelta(days=7)

    df = pd.DataFrame(rows)
    df = add_rounds_to_data(df)
    df = add_match_ids(df)
    df = add_home_away_goals_xg(df)
    return df


def test_predict_end_to_end_with_real_trained_artifacts(client, predictor_app, tmp_path):
    """Trains a tiny real model directly (bypassing /train's live Understat
    fetch) and saves the same artifacts /train would, including config.pkl —
    then hits /predict for real. Exercises the actual predict_match_lambdas
    wiring end-to-end, not just that the module imports."""
    from football_model.data.prepare_model_data import prepare_model_data
    from football_model.model.model import build_model
    from football_model.types.model_data import ModelConfig

    df = _make_synthetic_df_with_future_round()
    max_round = df.loc[df["datetime"] <= pd.Timestamp.today(), "round"].max()

    input_model_data = prepare_model_data(df, max_round=max_round)
    config = ModelConfig(clip_theta=5.0, center_team_strength=False)
    model = build_model(input_model_data, config)
    with model:
        trace = pm.sample(draws=5, tune=5, chains=1, cores=1, progressbar=False, random_seed=0)

    with open(predictor_app.TRACE_PATH, "wb") as f:
        pickle.dump(trace, f)
    with open(predictor_app.MODEL_DATA_PATH, "wb") as f:
        pickle.dump(input_model_data, f)
    with open(predictor_app.DATAFRAME_PATH, "wb") as f:
        pickle.dump(df, f)
    with open(predictor_app.CONFIG_PATH, "wb") as f:
        pickle.dump(config, f)

    resp = client.get("/predict")
    assert resp.status_code == 200
    body = resp.json()
    assert body["predictions"], "expected at least one match prediction"

    for match in body["predictions"]:
        probs = match["outcome_probabilities"]
        total = probs["home_win"] + probs["draw"] + probs["away_win"]
        assert 0.99 <= total <= 1.01
        assert match["expected_goals_home"]["mean"] > 0
        assert match["expected_goals_away"]["mean"] > 0
        assert np.isfinite(match["expected_goals_home"]["mean"])
        assert np.isfinite(match["expected_goals_away"]["mean"])


def test_predict_falls_back_to_default_config_when_config_pkl_missing(client, predictor_app):
    """Traces saved before config.pkl existed must still work — /predict
    should fall back to a sane default (clip_theta=5.0) rather than error."""
    from football_model.data.prepare_model_data import prepare_model_data
    from football_model.model.model import build_model
    from football_model.types.model_data import ModelConfig

    df = _make_synthetic_df_with_future_round(seed=1)
    max_round = df.loc[df["datetime"] <= pd.Timestamp.today(), "round"].max()

    input_model_data = prepare_model_data(df, max_round=max_round)
    config = ModelConfig(clip_theta=5.0, center_team_strength=False)
    model = build_model(input_model_data, config)
    with model:
        trace = pm.sample(draws=5, tune=5, chains=1, cores=1, progressbar=False, random_seed=0)

    with open(predictor_app.TRACE_PATH, "wb") as f:
        pickle.dump(trace, f)
    with open(predictor_app.MODEL_DATA_PATH, "wb") as f:
        pickle.dump(input_model_data, f)
    with open(predictor_app.DATAFRAME_PATH, "wb") as f:
        pickle.dump(df, f)
    # deliberately no config.pkl written

    assert not predictor_app.CONFIG_PATH.exists()
    resp = client.get("/predict")
    assert resp.status_code == 200
    assert resp.json()["predictions"]
