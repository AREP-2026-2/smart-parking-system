"""Métricas de la Sección 7.2: MAPE, MAE, MSE y R²."""
from __future__ import annotations

import numpy as np

MAPE_MIN_OCCUPANCY = 0.05


def compute(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """El MAPE omite ocupaciones menores al 5%, donde el porcentaje se vuelve inestable."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    errors = predicted - actual
    valid = actual >= MAPE_MIN_OCCUPANCY
    mape = float(np.mean(np.abs(errors[valid]) / actual[valid]) * 100) if valid.any() else float("nan")
    total = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1.0 - float(np.sum(errors ** 2)) / total if total > 0 else float("nan")
    return {"mape": mape, "mae": float(np.mean(np.abs(errors))),
            "mse": float(np.mean(errors ** 2)), "r2": r2}
