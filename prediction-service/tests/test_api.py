from datetime import datetime, timedelta

import pytest
import torch
from fastapi.testclient import TestClient

from app.engine import ForecastEngine
from app.main import app
from app.router import BranchHealth, ERROR_LIMIT_FACTOR, MIN_SAMPLES, bracket, select_branch

AS_OF = datetime(2026, 5, 5, 8)
HORIZONS = (1, 3, 5, 8, 12, 24)
LSTM_VALUE = 0.8
TRANSFORMER_VALUE = 0.6
VAL_MAE = 0.05


class FixedModel:
    """Doble de prueba: predice un valor constante conocido."""

    def __init__(self, value: float):
        self.value = value

    def __call__(self, window, context):
        size = len(window)
        return torch.full((size,), self.value), torch.full((size, 24), 1 / 24)


@pytest.fixture()
def client():
    models, names, mae = {}, {}, {}
    for horizon in HORIZONS:
        models[("lstm", horizon)] = FixedModel(LSTM_VALUE)
        models[("transformer", horizon)] = FixedModel(TRANSFORMER_VALUE)
        for branch in ("lstm", "transformer"):
            names[(branch, horizon)] = f"{branch}-h{horizon}-v1"
            mae[(branch, horizon)] = VAL_MAE
    app.state.engine = ForecastEngine(models, names, ["A", "B"], tau=1, health=BranchHealth(mae))
    with TestClient(app) as test_client:
        yield test_client


def query(horizon: float, zone: str = "A", observed: float = 0.4, week_ago: float = 0.33) -> dict:
    history = [{"timestamp": (AS_OF - timedelta(hours=23 - i)).isoformat(), "occupancy": observed,
                "inflow": 0.1, "outflow": 0.05, "class_change": 0, "no_class_day": 0, "rain": 0}
               for i in range(24)]
    return {"zone": zone, "as_of": AS_OF.isoformat(), "horizon_hours": horizon, "history": history,
            "future_calendar": [{"class_change": 0, "no_class_day": 0}] * 24,
            "week_ago_occupancy": week_ago}


def forecast(client, *queries, include_attention=False) -> list[dict]:
    response = client.post("/v1/forecasts", json={"queries": list(queries), "include_attention": include_attention})
    assert response.status_code == 200, response.text
    return response.json()["results"]


def test_select_branch_uses_tau_inclusively():
    assert select_branch(1, 1) == "lstm"
    assert select_branch(1.01, 1) == "transformer"


def test_bracket_interpolation_points():
    assert bracket(0.5) == (0.0, 1.0, 0.5)
    assert bracket(3) == (1.0, 3.0, 1.0)
    assert bracket(4) == (3.0, 5.0, 0.5)


def test_short_horizon_routes_to_lstm(client):
    result = forecast(client, query(1))[0]
    assert result["branch"] == "lstm" and result["model"] == "lstm-h1-v1"
    assert result["occupancy"] == pytest.approx(LSTM_VALUE)


def test_long_horizon_routes_to_transformer(client):
    result = forecast(client, query(5))[0]
    assert result["branch"] == "transformer" and result["model"] == "transformer-h5-v1"
    assert result["occupancy"] == pytest.approx(TRANSFORMER_VALUE)


def test_minutes_interpolate_between_observed_and_one_hour(client):
    result = forecast(client, query(0.25, observed=0.4))[0]
    assert result["branch"] == "lstm"
    assert result["occupancy"] == pytest.approx(0.75 * 0.4 + 0.25 * LSTM_VALUE)


def test_batch_keeps_order_across_branches(client):
    results = forecast(client, query(5, zone="B"), query(1, zone="A"), query(24, zone="A"))
    assert [(r["zone"], r["branch"]) for r in results] == [("B", "transformer"), ("A", "lstm"), ("A", "transformer")]


def test_attention_is_optional(client):
    assert forecast(client, query(1))[0]["attention"] is None
    assert len(forecast(client, query(1), include_attention=True)[0]["attention"]) == 24


def test_branch_down_falls_back_only_for_that_branch(client):
    assert client.post("/v1/admin/branches/lstm", json={"available": False}).status_code == 200
    short, long = forecast(client, query(1), query(5))
    assert (short["branch"], short["degraded"], short["reason"]) == ("baseline", True, "branch_down")
    assert short["occupancy"] == pytest.approx(0.33)
    assert long["branch"] == "transformer" and not long["degraded"]


def test_branch_recovers_after_being_enabled(client):
    client.post("/v1/admin/branches/lstm", json={"available": False})
    client.post("/v1/admin/branches/lstm", json={"available": True})
    assert forecast(client, query(1))[0]["branch"] == "lstm"


def feedback(client, error: float, count: int, branch: str = "lstm", horizon: int = 1) -> None:
    items = [{"branch": branch, "horizon": horizon, "predicted": error, "observed": 0.0}] * count
    assert client.post("/v1/feedback", json={"items": items}).status_code == 200


def test_error_above_limit_triggers_baseline(client):
    feedback(client, VAL_MAE * ERROR_LIMIT_FACTOR * 1.5, MIN_SAMPLES)
    result = forecast(client, query(1))[0]
    assert (result["branch"], result["reason"]) == ("baseline", "error_limit")


def test_error_limit_needs_minimum_samples(client):
    feedback(client, 0.9, MIN_SAMPLES - 1)
    assert forecast(client, query(1))[0]["branch"] == "lstm"


def test_error_below_limit_keeps_branch(client):
    feedback(client, VAL_MAE, MIN_SAMPLES)
    assert forecast(client, query(1))[0]["branch"] == "lstm"


def test_shadow_query_ignores_error_limit_but_not_outage(client):
    feedback(client, 0.9, MIN_SAMPLES)
    shadow = {**query(1), "ignore_error_limit": True}
    assert forecast(client, shadow)[0]["branch"] == "lstm"
    client.post("/v1/admin/branches/lstm", json={"available": False})
    assert forecast(client, shadow)[0]["branch"] == "baseline"


def test_tau_can_change_while_running(client):
    assert client.post("/v1/admin/router", json={"tau": 5}).status_code == 200
    assert forecast(client, query(5))[0]["branch"] == "lstm"
    assert client.post("/v1/admin/router", json={"tau": 0}).status_code == 200
    assert forecast(client, query(1))[0]["branch"] == "transformer"
    assert client.post("/v1/admin/router", json={"tau": 2}).status_code == 422


def test_drift_injection_and_reset(client):
    client.post("/v1/admin/branches/transformer/drift")
    assert forecast(client, query(5))[0]["reason"] == "error_limit"
    client.post("/v1/admin/reset")
    assert forecast(client, query(5))[0]["branch"] == "transformer"


@pytest.mark.parametrize("change", [
    {"horizon_hours": 25}, {"horizon_hours": 0}, {"zone": "Z"}, {"week_ago_occupancy": 1.5}, {"history": []},
    {"as_of": (AS_OF + timedelta(hours=1)).isoformat()},
])
def test_invalid_queries_are_rejected(client, change):
    response = client.post("/v1/forecasts", json={"queries": [{**query(1), **change}]})
    assert response.status_code == 422
