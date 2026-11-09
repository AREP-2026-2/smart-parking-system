"""Convierte el CSV canónico en ventanas supervisadas con partición cronológica.

Esquema del CSV de entrada (15 minutos u horario):
  timestamp, zone, occupied, capacity            obligatorias
  inflow, outflow, class_change, no_class_day, rain   opcionales (se asumen 0)
Por eso el cargador sirve igual para el simulador y para los conteos reales.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import WINDOW_HOURS
from .features import step_vector, target_vector, zone_one_hot

REQUIRED_COLUMNS = {"timestamp", "zone", "occupied", "capacity"}
OPTIONAL_COLUMNS = ("inflow", "outflow", "class_change", "no_class_day", "rain")
WEEK_HOURS = 168


def load_hourly(path: Path) -> pd.DataFrame:
    """Valida el CSV y lo agrega a resolución horaria."""
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {sorted(missing)}")
    for column in OPTIONAL_COLUMNS:
        if column not in frame:
            frame[column] = 0.0
    if frame[["timestamp", "zone"]].duplicated().any():
        raise ValueError("Hay filas duplicadas para el mismo instante y zona")
    if (frame["occupied"] < 0).any() or (frame["occupied"] > frame["capacity"]).any():
        raise ValueError("La columna occupied debe estar entre 0 y capacity")

    frame = frame.sort_values("timestamp")
    frame["timestamp"] = frame["timestamp"].dt.floor("h")
    hourly = frame.groupby(["zone", "timestamp"], as_index=False).agg(
        occupied=("occupied", "last"), capacity=("capacity", "last"),
        inflow=("inflow", "sum"), outflow=("outflow", "sum"),
        class_change=("class_change", "max"), no_class_day=("no_class_day", "max"),
        rain=("rain", "max"))

    expected = pd.date_range(hourly["timestamp"].min(), hourly["timestamp"].max(), freq="h")
    for zone, group in hourly.groupby("zone"):
        if len(group) != len(expected):
            raise ValueError(f"La zona {zone} tiene {len(expected) - len(group)} horas faltantes")
    return hourly.sort_values(["zone", "timestamp"]).reset_index(drop=True)


@dataclass(frozen=True)
class Split:
    val_start: pd.Timestamp
    test_start: pd.Timestamp
    end: pd.Timestamp


def chronological_split(hourly: pd.DataFrame, train: float = 0.70, val: float = 0.15) -> Split:
    timestamps = hourly["timestamp"].drop_duplicates().sort_values().reset_index(drop=True)
    return Split(timestamps[int(len(timestamps) * train)],
                 timestamps[int(len(timestamps) * (train + val))],
                 timestamps.iloc[-1])


@dataclass
class Samples:
    window: np.ndarray       # (n, 24, variables de paso)
    target_context: np.ndarray  # (n, variables objetivo + zonas)
    y: np.ndarray            # ocupación real en el objetivo, 0-1
    week_ago: np.ndarray     # línea base: misma hora de la semana anterior
    zone: np.ndarray
    target_time: np.ndarray

    def __len__(self) -> int:
        return len(self.y)


def build_samples(hourly: pd.DataFrame, horizon: int, zones: list[str]) -> Samples:
    """Una muestra por zona y origen. Se descarta la primera semana (sin línea base)."""
    windows, contexts, ys, week_agos, sample_zones, target_times = [], [], [], [], [], []
    for zone, group in hourly.groupby("zone"):
        group = group.sort_values("timestamp")
        times = [moment.to_pydatetime() for moment in group["timestamp"]]
        capacity = group["capacity"].to_numpy(dtype=float)
        occupancy = group["occupied"].to_numpy(dtype=float) / capacity
        inflow = group["inflow"].to_numpy(dtype=float) / capacity
        outflow = group["outflow"].to_numpy(dtype=float) / capacity
        class_change = group["class_change"].to_numpy(dtype=float)
        no_class = group["no_class_day"].to_numpy(dtype=float)
        rain = group["rain"].to_numpy(dtype=float)
        one_hot = zone_one_hot(zone, zones)

        steps = np.array([
            step_vector(times[i], occupancy[i], inflow[i], outflow[i],
                        class_change[i], no_class[i], rain[i])
            for i in range(len(group))], dtype=np.float32)

        for target in range(max(WEEK_HOURS, WINDOW_HOURS + horizon - 1), len(group)):
            origin = target - horizon  # Último paso observado.
            windows.append(steps[origin - WINDOW_HOURS + 1:origin + 1])
            contexts.append(np.concatenate([
                np.array(target_vector(times[target], class_change[target], no_class[target]),
                         dtype=np.float32), one_hot]))
            ys.append(occupancy[target])
            week_agos.append(occupancy[target - WEEK_HOURS])
            sample_zones.append(zone)
            target_times.append(np.datetime64(times[target]))

    return Samples(np.stack(windows), np.stack(contexts), np.array(ys, dtype=np.float32),
                   np.array(week_agos, dtype=np.float32), np.array(sample_zones),
                   np.array(target_times, dtype="datetime64[ns]"))


def subset(samples: Samples, mask: np.ndarray) -> Samples:
    return Samples(samples.window[mask], samples.target_context[mask], samples.y[mask],
                   samples.week_ago[mask], samples.zone[mask], samples.target_time[mask])


def split_samples(samples: Samples, split: Split) -> tuple[Samples, Samples, Samples]:
    """Parte por instante objetivo: ninguna etiqueta de validación entra en train."""
    val_start = np.datetime64(split.val_start)
    test_start = np.datetime64(split.test_start)
    times = samples.target_time
    return (subset(samples, times < val_start),
            subset(samples, (times >= val_start) & (times < test_start)),
            subset(samples, times >= test_start))
