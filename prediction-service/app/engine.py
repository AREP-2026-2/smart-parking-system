"""Carga los modelos versionados y resuelve lotes de consultas de pronóstico."""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Protocol, Sequence

import numpy as np
import torch

from parking_ml.features import STEP_FEATURES, TARGET_FEATURES, step_vector, target_vector, zone_one_hot
from parking_ml.models import build_model

from .router import BranchHealth, bracket, select_branch
from .schemas import ForecastQuery, ForecastResult


class Predictor(Protocol):
    def __call__(self, window: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...


class ForecastEngine:
    def __init__(self, models: dict[tuple[str, int], Predictor], names: dict[tuple[str, int], str],
                 zones: list[str], tau: int, health: BranchHealth):
        self.models = models
        self.names = names
        self.zones = zones
        self.tau = tau  # Umbral activo; se puede cambiar en caliente.
        self.health = health

    @classmethod
    def from_artifacts(cls, directory: Path, initial_tau: int) -> "ForecastEngine":
        registry = json.loads((directory / "registry.json").read_text(encoding="utf-8"))
        zones = registry["zones"]
        context_size = len(TARGET_FEATURES) + len(zones)
        models, names, validation_mae = {}, {}, {}
        for meta in registry["models"].values():
            key = (meta["branch"], meta["horizon"])
            model = build_model(meta["branch"], len(STEP_FEATURES), context_size)
            model.load_state_dict(torch.load(directory / meta["file"], map_location="cpu", weights_only=True))
            model.eval()
            models[key], names[key], validation_mae[key] = model, meta["name"], meta["val_mae"]
        return cls(models, names, zones, initial_tau, BranchHealth(validation_mae))

    def _inputs(self, query: ForecastQuery, knot: int) -> tuple[list[list[float]], np.ndarray]:
        window = [step_vector(s.timestamp, s.occupancy, s.inflow, s.outflow,
                              s.class_change, s.no_class_day, s.rain) for s in query.history]
        calendar = query.future_calendar[knot - 1]
        target = target_vector(query.as_of + timedelta(hours=knot), calendar.class_change, calendar.no_class_day)
        return window, np.concatenate([np.array(target, dtype=np.float32), zone_one_hot(query.zone, self.zones)])

    @torch.no_grad()
    def _run(self, key: tuple[str, int], items: list[tuple[int, ForecastQuery]],
             predictions: dict, attention: dict) -> None:
        pairs = [self._inputs(query, key[1]) for _, query in items]
        window = torch.tensor(np.array([pair[0] for pair in pairs]), dtype=torch.float32)
        context = torch.tensor(np.array([pair[1] for pair in pairs]), dtype=torch.float32)
        values, weights = self.models[key](window, context)
        for position, (index, _) in enumerate(items):
            predictions[(index, key[1])] = float(values[position])
            attention[(index, key[1])] = weights[position].tolist()

    def forecast(self, queries: Sequence[ForecastQuery], include_attention: bool = False) -> list[ForecastResult]:
        plans: list[Optional[tuple]] = []
        reasons: list[Optional[str]] = []
        pending: dict[tuple[str, int], list[tuple[int, ForecastQuery]]] = defaultdict(list)
        for index, query in enumerate(queries):
            branch = select_branch(query.horizon_hours, self.tau)
            lower, upper, weight = bracket(query.horizon_hours)
            problem = self.health.problem(branch, int(upper), query.ignore_error_limit)
            if problem:
                plans.append(None)
                reasons.append(problem)
                continue
            plans.append((branch, int(lower), int(upper), weight))
            reasons.append(None)
            pending[(branch, int(upper))].append((index, query))
            if lower > 0:
                pending[(branch, int(lower))].append((index, query))

        predictions: dict[tuple[int, int], float] = {}
        attention: dict[tuple[int, int], list[float]] = {}
        for key, items in pending.items():
            self._run(key, items, predictions, attention)

        return [self._compose(index, query, plans[index], reasons[index], predictions, attention, include_attention)
                for index, query in enumerate(queries)]

    def _compose(self, index: int, query: ForecastQuery, plan: Optional[tuple], reason: Optional[str],
                 predictions: dict, attention: dict, include_attention: bool) -> ForecastResult:
        target_time: datetime = query.as_of + timedelta(hours=query.horizon_hours)
        if plan is None:
            return ForecastResult(zone=query.zone, horizon_hours=query.horizon_hours, target_time=target_time,
                                  occupancy=query.week_ago_occupancy, branch="baseline",
                                  model="same-hour-last-week", degraded=True, reason=reason)
        branch, lower, upper, weight = plan
        low = predictions[(index, lower)] if lower > 0 else query.history[-1].occupancy
        value = (1 - weight) * low + weight * predictions[(index, upper)]
        return ForecastResult(zone=query.zone, horizon_hours=query.horizon_hours, target_time=target_time,
                              occupancy=min(1.0, max(0.0, value)), branch=branch,
                              model=self.names[(branch, upper)], degraded=False,
                              attention=attention[(index, upper)] if include_attention else None)
