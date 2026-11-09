"""Entrenamiento y evaluación del motor predictivo híbrido (Módulo 4 del paper)."""

HORIZONS = (1, 3, 5, 8, 12, 24)
WINDOW_HOURS = 24
BRANCHES = ("lstm", "transformer")
