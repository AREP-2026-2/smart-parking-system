from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from parking_ml.dataset import build_samples, chronological_split, load_hourly, split_samples
from parking_ml.metrics import compute
from parking_ml.models import build_model
from parking_ml.simulator import simulate
from parking_ml.train import calibrate_tau


@pytest.fixture(scope="module")
def csv_path(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("data") / "occupancy.csv"
    simulate(weeks=4, seed=1).to_csv(path, index=False)
    return path


def test_simulator_respects_capacity():
    frame = simulate(weeks=2, seed=3)
    assert (frame["occupied"] >= 0).all()
    assert (frame["occupied"] <= frame["capacity"]).all()
    assert frame["occupied"].max() > 0


def test_simulator_is_reproducible():
    pd.testing.assert_frame_equal(simulate(weeks=1, seed=5), simulate(weeks=1, seed=5))


def test_sunday_has_no_classes():
    frame = simulate(weeks=1, seed=2)
    sunday = frame[frame["timestamp"].dt.day_of_week == 6]
    assert (sunday["no_class_day"] == 1).all()
    assert sunday["class_change"].sum() == 0


def test_load_hourly_aggregates_flows_and_keeps_last_occupancy(csv_path):
    raw = pd.read_csv(csv_path, parse_dates=["timestamp"])
    hourly = load_hourly(csv_path)
    moment = pd.Timestamp("2026-01-27 08:00")
    selected = raw[(raw["zone"] == "A") & (raw["timestamp"].dt.floor("h") == moment)]
    row = hourly[(hourly["zone"] == "A") & (hourly["timestamp"] == moment)].iloc[0]
    assert row["inflow"] == selected["inflow"].sum()
    assert row["occupied"] == selected.sort_values("timestamp")["occupied"].iloc[-1]


def test_load_hourly_accepts_required_columns_only(tmp_path):
    times = pd.date_range("2026-01-01", periods=48, freq="h")
    path = tmp_path / "minimal.csv"
    pd.DataFrame({"timestamp": times, "zone": "A", "occupied": 1, "capacity": 10}).to_csv(path, index=False)
    assert load_hourly(path)["rain"].sum() == 0


def test_load_hourly_rejects_gaps(tmp_path):
    times = pd.date_range("2026-01-01", periods=48, freq="h").delete(5)
    path = tmp_path / "gaps.csv"
    pd.DataFrame({"timestamp": times, "zone": "A", "occupied": 1, "capacity": 10}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="faltantes"):
        load_hourly(path)


def test_load_hourly_rejects_occupied_over_capacity(tmp_path):
    times = pd.date_range("2026-01-01", periods=2, freq="h")
    path = tmp_path / "bad.csv"
    pd.DataFrame({"timestamp": times, "zone": "A", "occupied": 11, "capacity": 10}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="capacity"):
        load_hourly(path)


@pytest.mark.parametrize("horizon", [1, 24])
def test_window_ends_exactly_horizon_hours_before_target(csv_path, horizon):
    hourly = load_hourly(csv_path)
    zones = sorted(hourly["zone"].unique().tolist())
    samples = build_samples(hourly, horizon, zones)
    zone_a = hourly[hourly["zone"] == "A"].set_index("timestamp")
    index = np.flatnonzero(samples.zone == "A")[100]
    target = pd.Timestamp(samples.target_time[index])
    origin = target - pd.Timedelta(hours=horizon)
    expected = zone_a.loc[origin, "occupied"] / zone_a.loc[origin, "capacity"]
    assert samples.window[index, -1, 0] == pytest.approx(expected)
    assert samples.y[index] == pytest.approx(zone_a.loc[target, "occupied"] / zone_a.loc[target, "capacity"])
    week_ago = target - pd.Timedelta(hours=168)
    assert samples.week_ago[index] == pytest.approx(zone_a.loc[week_ago, "occupied"] / zone_a.loc[week_ago, "capacity"])


def test_split_is_chronological_by_target(csv_path):
    hourly = load_hourly(csv_path)
    zones = sorted(hourly["zone"].unique().tolist())
    split = chronological_split(hourly)
    train, val, test = split_samples(build_samples(hourly, 3, zones), split)
    assert train.target_time.max() < np.datetime64(split.val_start) <= val.target_time.min()
    assert val.target_time.max() < np.datetime64(split.test_start) <= test.target_time.min()


def test_mape_ignores_near_zero_actuals():
    result = compute(np.array([0.0, 0.5]), np.array([0.2, 0.25]))
    assert result["mape"] == pytest.approx(50.0)
    assert result["mae"] == pytest.approx(0.225)


def test_r2_is_one_for_perfect_prediction():
    actual = np.array([0.1, 0.4, 0.9])
    assert compute(actual, actual)["r2"] == pytest.approx(1.0)


@pytest.mark.parametrize("branch", ["lstm", "transformer"])
def test_models_output_valid_occupancy_and_attention(branch):
    model = build_model(branch, 11, 12).eval()
    prediction, weights = model(torch.rand(4, 24, 11), torch.rand(4, 12))
    assert prediction.shape == (4,)
    assert ((prediction >= 0) & (prediction <= 1)).all()
    assert weights.shape == (4, 24)
    assert weights.sum(dim=1).detach().numpy() == pytest.approx(np.ones(4), abs=1e-5)


def test_calibrate_tau_picks_crossover():
    lstm = {1: 0.05, 3: 0.20, 5: 0.20, 8: 0.20, 12: 0.20, 24: 0.20}
    transformer = {1: 0.10, 3: 0.10, 5: 0.10, 8: 0.10, 12: 0.10, 24: 0.10}
    tau, _ = calibrate_tau({"lstm": lstm, "transformer": transformer})
    assert tau == 1


def test_calibrate_tau_can_select_a_single_branch():
    lstm = dict.fromkeys((1, 3, 5, 8, 12, 24), 0.05)
    transformer = dict.fromkeys((1, 3, 5, 8, 12, 24), 0.10)
    assert calibrate_tau({"lstm": lstm, "transformer": transformer})[0] == 24
    assert calibrate_tau({"lstm": transformer, "transformer": lstm})[0] == 0
