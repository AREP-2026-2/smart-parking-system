"""Simulador de ocupación de un parqueadero universitario.

Genera datos SINTÉTICOS para que el prototipo funcione mientras no existan los
datos de campo de la Sección 7.3 del paper. Los horarios, capacidades y tasas son
supuestos plausibles, no mediciones de la ECI: cualquier métrica obtenida con
estos datos describe al simulador y no valida nada sobre el campus real.

Modelo: proceso de stock y flujo a 15 minutos. Los vehículos llegan antes de cada
bloque de clases, eligen zona por preferencia, se desbordan a otra zona si la
preferida está llena y se van al terminar un bloque posterior.

Uso: python -m parking_ml.simulator --out ../data --weeks 18 --seed 7
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

STEP_MINUTES = 15
STEPS_PER_DAY = 24 * 60 // STEP_MINUTES


@dataclass(frozen=True)
class Zone:
    id: str
    name: str
    capacity: int
    morning_preference: float
    afternoon_preference: float


ZONES = (
    Zone("A", "Zona A - Entrada principal", 120, 0.42, 0.26),
    Zone("B", "Zona B - Bloque central", 100, 0.28, 0.24),
    Zone("C", "Zona C - Laboratorios", 80, 0.14, 0.24),
    Zone("D", "Zona D - Coliseo", 90, 0.10, 0.16),
    Zone("E", "Zona E - Periferia", 110, 0.06, 0.10),
)

# Hora de inicio de cada bloque e intensidad relativa de llegadas.
WEEKDAY_BLOCKS = ((7.0, 1.00), (8.5, 0.55), (10.0, 0.75), (11.5, 0.45),
                  (13.0, 0.70), (14.5, 0.50), (16.0, 0.55), (17.5, 0.40), (19.0, 0.30))
SATURDAY_BLOCKS = ((7.0, 0.40), (8.5, 0.25), (10.0, 0.30), (11.5, 0.15))

# Festivos de Colombia dentro del primer semestre de 2026 y Semana Santa.
HOLIDAYS = {date(2026, 3, 23), date(2026, 4, 2), date(2026, 4, 3),
            date(2026, 5, 1), date(2026, 5, 18)}
EXAM_WEEKS = {6, 12, 17}  # Semanas de parciales, numeradas desde 1.

ARRIVALS_PER_BLOCK = 170.0
CLOSING_HOUR = 22


def blocks_for(day: date, week_number: int) -> tuple:
    """Bloques académicos del día. Vacío si no hay clases."""
    if day in HOLIDAYS or day.weekday() == 6:
        return ()
    blocks = SATURDAY_BLOCKS if day.weekday() == 5 else WEEKDAY_BLOCKS
    if week_number in EXAM_WEEKS:
        return tuple((start, weight * 0.8) for start, weight in blocks)
    return blocks


def _rain_series(n_steps: int, rng: np.random.Generator) -> np.ndarray:
    """Lluvia como cadena de Markov de dos estados a 15 minutos."""
    rain = np.zeros(n_steps, dtype=np.int8)
    for i in range(1, n_steps):
        probability = 0.90 if rain[i - 1] else 0.02
        rain[i] = int(rng.random() < probability)
    return rain


def _zone_probabilities(hour: float) -> np.ndarray:
    key = "morning_preference" if hour < 12 else "afternoon_preference"
    weights = np.array([getattr(zone, key) for zone in ZONES], dtype=float)
    return weights / weights.sum()


def _arrival_rates(blocks: tuple, day_factor: float, event: tuple | None) -> np.ndarray:
    """Tasa esperada de llegadas por paso del día (campo completo)."""
    rates = np.zeros(STEPS_PER_DAY)
    for start, weight in blocks:
        center = (start - 0.25) * 60 / STEP_MINUTES
        for step in range(STEPS_PER_DAY):
            rates[step] += weight * np.exp(-0.5 * ((step - center) / 1.3) ** 2)
    rates *= ARRIVALS_PER_BLOCK / (1.3 * np.sqrt(2 * np.pi)) * day_factor
    if event is not None:  # Evento no presente en el calendario académico.
        center, size = event
        for step in range(STEPS_PER_DAY):
            rates[step] += size * np.exp(-0.5 * ((step - center) / 3.0) ** 2) / (3.0 * np.sqrt(2 * np.pi))
    return rates


def _sample_departure(arrival_step: int, blocks: tuple, rng: np.random.Generator) -> int:
    """Paso del día en el que sale un vehículo, alineado al final de un bloque."""
    arrival_hour = arrival_step * STEP_MINUTES / 60
    ends = [start + 1.5 for start, _ in blocks if start + 1.5 > arrival_hour + 1.0]
    if ends:
        # Las estadías típicas abarcan de dos a cuatro bloques.
        index = min(len(ends) - 1, rng.choice([1, 2, 3, 4], p=[0.2, 0.35, 0.3, 0.15]))
        leave_hour = ends[index] + rng.normal(0.2, 0.15)
    else:
        leave_hour = arrival_hour + rng.uniform(0.5, 2.0)
    leave_step = min(int(leave_hour * 60 / STEP_MINUTES), CLOSING_HOUR * 4)
    return max(leave_step, arrival_step + 1)


def simulate(weeks: int = 18, seed: int = 7, start: date = date(2026, 1, 26)) -> pd.DataFrame:
    """Devuelve filas por zona y paso de 15 minutos con el esquema canónico."""
    rng = np.random.default_rng(seed)
    n_zones = len(ZONES)
    capacities = np.array([zone.capacity for zone in ZONES])
    n_days = weeks * 7
    n_steps = n_days * STEPS_PER_DAY
    rain = _rain_series(n_steps, rng)

    occupied = np.zeros(n_zones, dtype=int)
    occupied_hist = np.zeros((n_steps, n_zones), dtype=int)
    inflow_hist = np.zeros((n_steps, n_zones), dtype=int)
    outflow_hist = np.zeros((n_steps, n_zones), dtype=int)
    class_change = np.zeros(n_steps)
    no_class_day = np.zeros(n_steps, dtype=np.int8)

    # departures[paso][zona] = vehículos que salen en ese paso.
    departures = np.zeros((n_steps + STEPS_PER_DAY, n_zones), dtype=int)

    for day_index in range(n_days):
        day = start + timedelta(days=day_index)
        week_number = day_index // 7 + 1
        blocks = blocks_for(day, week_number)
        day_offset = day_index * STEPS_PER_DAY
        no_class_day[day_offset:day_offset + STEPS_PER_DAY] = int(not blocks)

        for block_start, weight in blocks:
            step = int(block_start * 60 / STEP_MINUTES)
            class_change[day_offset + step] = weight

        day_factor = float(rng.lognormal(0.0, 0.08))
        event = None
        if blocks and rng.random() < 0.06:
            event = (rng.integers(40, 76), rng.uniform(60, 140))
        rates = _arrival_rates(blocks, day_factor, event)

        for step_of_day in range(STEPS_PER_DAY):
            step = day_offset + step_of_day
            leaving = np.minimum(departures[step, :], occupied)
            occupied -= leaving
            outflow_hist[step] = leaving

            rate = rates[step_of_day] * (1.15 if rain[step] else 1.0)
            arrivals = int(rng.poisson(rate))
            hour = step_of_day * STEP_MINUTES / 60
            probabilities = _zone_probabilities(hour)
            for _ in range(arrivals):
                zone = int(rng.choice(n_zones, p=probabilities))
                if occupied[zone] >= capacities[zone]:
                    free = np.flatnonzero(occupied < capacities)
                    if free.size == 0:
                        continue  # Parqueadero lleno: el vehículo se va.
                    spill = probabilities[free] / probabilities[free].sum()
                    zone = int(rng.choice(free, p=spill))
                occupied[zone] += 1
                inflow_hist[step, zone] += 1
                leave_step = _sample_departure(step_of_day, blocks, rng)
                departures[day_offset + leave_step, zone] += 1

            occupied_hist[step] = occupied

        # Vehículos de personal y vigilancia: piso de ocupación durante todo el día.
        staff_cars = np.minimum(rng.poisson(capacities * 0.02), capacities)
        day_range = slice(day_offset, day_offset + STEPS_PER_DAY)
        occupied_hist[day_range] = np.maximum(occupied_hist[day_range], staff_cars)

    timestamps = pd.date_range(pd.Timestamp(start), periods=n_steps, freq=f"{STEP_MINUTES}min")
    frames = []
    for index, zone in enumerate(ZONES):
        frames.append(pd.DataFrame({
            "timestamp": timestamps,
            "zone": zone.id,
            "occupied": occupied_hist[:, index],
            "capacity": zone.capacity,
            "inflow": inflow_hist[:, index],
            "outflow": outflow_hist[:, index],
            "class_change": class_change,
            "no_class_day": no_class_day,
            "rain": rain,
        }))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("../data"))
    parser.add_argument("--weeks", type=int, default=18)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    frame = simulate(weeks=args.weeks, seed=args.seed)
    path = args.out / "occupancy_15min.csv"
    frame.to_csv(path, index=False)
    (args.out / "zones.json").write_text(json.dumps(
        [{"id": z.id, "name": z.name, "capacity": z.capacity} for z in ZONES],
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(frame)} filas en {path}")


if __name__ == "__main__":
    main()
