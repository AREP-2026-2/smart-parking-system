"""Definición única de variables, compartida por entrenamiento e inferencia."""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np

STEP_FEATURES = ("occupancy", "inflow", "outflow", "class_change", "no_class_day", "rain",
                 "hour_sin", "hour_cos", "dow_sin", "dow_cos", "weekend")
TARGET_FEATURES = ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "weekend",
                   "class_change", "no_class_day")


def time_features(moment: datetime) -> tuple[float, float, float, float, float]:
    """Hora y día codificados de forma cíclica, más indicador de fin de semana."""
    hour_angle = 2 * math.pi * moment.hour / 24
    dow_angle = 2 * math.pi * moment.weekday() / 7
    return (math.sin(hour_angle), math.cos(hour_angle),
            math.sin(dow_angle), math.cos(dow_angle),
            float(moment.weekday() >= 5))


def step_vector(moment: datetime, occupancy: float, inflow: float, outflow: float,
                class_change: float, no_class_day: float, rain: float) -> list[float]:
    """Paso de la ventana histórica. Los flujos llegan divididos por la capacidad."""
    return [occupancy, inflow, outflow, class_change, no_class_day, rain, *time_features(moment)]


def target_vector(moment: datetime, class_change: float, no_class_day: float) -> list[float]:
    """Solo variables conocidas de antemano; el clima futuro no se incluye."""
    return [*time_features(moment), class_change, no_class_day]


def zone_one_hot(zone: str, zones: list[str]) -> np.ndarray:
    vector = np.zeros(len(zones), dtype=np.float32)
    vector[zones.index(zone)] = 1.0
    return vector
