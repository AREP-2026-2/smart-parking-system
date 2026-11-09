"""Enrutador por horizonte y estado de salud de las ramas (Secciones 4.1 y 4.4)."""
from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Optional

KNOTS = (0.0, 1.0, 3.0, 5.0, 8.0, 12.0, 24.0)
# Un día completo de 5 zonas: el límite parte de un MAE que promedia días enteros,
# de modo que comparar solo horas pico produciría falsas alarmas.
ROLLING_WINDOW = 120
MIN_SAMPLES = 10
ERROR_LIMIT_FACTOR = 2.0


def select_branch(horizon: float, tau: float) -> str:
    """Regla del paper: si h ≤ τ responde el LSTM; si no, el Transformer."""
    return "lstm" if horizon <= tau else "transformer"


def bracket(horizon: float) -> tuple[float, float, float]:
    """Horizontes entrenados que encierran a h y peso del superior.

    El nodo 0 es la ocupación observada. Así un conductor a pocos minutos de
    llegar recibe una interpolación entre el presente y el pronóstico a 1 h.
    """
    for lower, upper in zip(KNOTS, KNOTS[1:]):
        if horizon <= upper:
            return lower, upper, (horizon - lower) / (upper - lower)
    raise ValueError(f"Horizonte fuera de rango: {horizon}")


class BranchHealth:
    """Disponibilidad manual y error observado móvil por rama y horizonte."""

    def __init__(self, validation_mae: dict[tuple[str, int], float]):
        self._limits = {key: mae * ERROR_LIMIT_FACTOR for key, mae in validation_mae.items()}
        self._errors = {key: deque(maxlen=ROLLING_WINDOW) for key in validation_mae}
        self._down: set[str] = set()
        self._lock = Lock()

    def set_available(self, branch: str, available: bool) -> None:
        with self._lock:
            (self._down.discard if available else self._down.add)(branch)

    def record(self, branch: str, horizon: int, absolute_error: float) -> None:
        with self._lock:
            self._errors[(branch, horizon)].append(absolute_error)

    def reset(self) -> None:
        with self._lock:
            self._down.clear()
            for errors in self._errors.values():
                errors.clear()

    def _rolling(self, key: tuple[str, int]) -> Optional[float]:
        errors = self._errors[key]
        return sum(errors) / len(errors) if len(errors) >= MIN_SAMPLES else None

    def problem(self, branch: str, horizon: int, ignore_error_limit: bool = False) -> Optional[str]:
        """Motivo por el que la rama no debe responder, o None si está sana."""
        with self._lock:
            if branch in self._down:
                return "branch_down"
            if ignore_error_limit:
                return None
            rolling = self._rolling((branch, horizon))
            if rolling is not None and rolling > self._limits[(branch, horizon)]:
                return "error_limit"
            return None

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [{"branch": branch, "horizon": horizon,
                     "available": branch not in self._down,
                     "rolling_mae": self._rolling((branch, horizon)),
                     "samples": len(self._errors[(branch, horizon)]),
                     "error_limit": self._limits[(branch, horizon)]}
                    for branch, horizon in sorted(self._errors)]
