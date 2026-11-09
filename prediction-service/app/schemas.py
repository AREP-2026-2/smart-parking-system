"""Contrato público del servicio de inferencia."""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

Branch = Literal["lstm", "transformer"]


class HistoryStep(BaseModel):
    timestamp: datetime
    occupancy: float = Field(ge=0, le=1)
    inflow: float = Field(ge=0, description="Entradas de la hora divididas por la capacidad")
    outflow: float = Field(ge=0, description="Salidas de la hora divididas por la capacidad")
    class_change: float = Field(ge=0)
    no_class_day: float = Field(ge=0, le=1)
    rain: float = Field(ge=0, le=1)


class CalendarStep(BaseModel):
    class_change: float = Field(ge=0)
    no_class_day: float = Field(ge=0, le=1)


class ForecastQuery(BaseModel):
    zone: str
    as_of: datetime
    horizon_hours: float = Field(gt=0, le=24)
    history: list[HistoryStep] = Field(min_length=24, max_length=24)
    future_calendar: list[CalendarStep] = Field(min_length=24, max_length=24,
                                                description="Calendario de as_of+1h a as_of+24h")
    week_ago_occupancy: float = Field(ge=0, le=1)
    ignore_error_limit: bool = Field(default=False, description="Evaluación en sombra para observabilidad")

    @model_validator(mode="after")
    def history_ends_at_as_of(self) -> "ForecastQuery":
        if self.history[-1].timestamp != self.as_of:
            raise ValueError("El último paso de history debe coincidir con as_of")
        return self


class ForecastRequest(BaseModel):
    queries: list[ForecastQuery] = Field(min_length=1, max_length=2000)
    include_attention: bool = False


class ForecastResult(BaseModel):
    zone: str
    horizon_hours: float
    target_time: datetime
    occupancy: float
    branch: Literal["lstm", "transformer", "baseline"]
    model: str
    degraded: bool
    reason: Optional[str] = None
    attention: Optional[list[float]] = None


class ForecastResponse(BaseModel):
    results: list[ForecastResult]


class FeedbackItem(BaseModel):
    branch: Branch
    horizon: Literal[1, 3, 5, 8, 12, 24]
    predicted: float = Field(ge=0, le=1)
    observed: float = Field(ge=0, le=1)


class FeedbackRequest(BaseModel):
    items: list[FeedbackItem] = Field(max_length=5000)


class AvailabilityRequest(BaseModel):
    available: bool


class RouterRequest(BaseModel):
    tau: Literal[0, 1, 3, 5, 8, 12, 24]
