"""Las dos ramas del motor predictivo (Sección 4.1 del paper).

Ambas reciben lo mismo y exponen los pesos de atención sobre las 24 horas de la
ventana, lo que permite repetir el análisis de la Sección 3.2.2.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class AttentionLSTM(nn.Module):
    """LSTM con atención aditiva sobre los pasos de la ventana."""

    def __init__(self, step_size: int, context_size: int, hidden: int = 64,
                 layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(step_size, hidden, layers, batch_first=True, dropout=dropout)
        self.score = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.head = nn.Sequential(nn.Linear(hidden * 2 + context_size, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1), nn.Sigmoid())

    def forward(self, window: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs, (state, _) = self.lstm(window)
        last = state[-1]
        query = last.unsqueeze(1).expand(-1, outputs.size(1), -1)
        weights = torch.softmax(self.score(torch.cat([outputs, query], dim=-1)).squeeze(-1), dim=1)
        attended = torch.sum(weights.unsqueeze(-1) * outputs, dim=1)
        prediction = self.head(torch.cat([attended, last, context], dim=-1)).squeeze(-1)
        return prediction, weights


class PositionalEncoding(nn.Module):
    def __init__(self, size: int, length: int = 64):
        super().__init__()
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(torch.arange(0, size, 2, dtype=torch.float32) * (-math.log(10000.0) / size))
        encoding = torch.zeros(length, size)
        encoding[:, 0::2] = torch.sin(position * scale)
        encoding[:, 1::2] = torch.cos(position * scale)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.encoding[:, :values.size(1)]


class TransformerForecaster(nn.Module):
    """Codificador con autoatención y un paso de decodificación por atención cruzada.

    La consulta del decodificador es el contexto conocido del instante objetivo.
    """

    def __init__(self, step_size: int, context_size: int, size: int = 64, heads: int = 4,
                 layers: int = 2, feedforward: int = 128, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Linear(step_size, size)
        self.position = PositionalEncoding(size)
        layer = nn.TransformerEncoderLayer(size, heads, feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.query = nn.Linear(context_size, size)
        self.cross = nn.MultiheadAttention(size, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(size)
        self.head = nn.Sequential(nn.Linear(size + context_size, size), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(size, 1), nn.Sigmoid())

    def forward(self, window: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        memory = self.encoder(self.position(self.embed(window)))
        query = self.query(context).unsqueeze(1)
        attended, weights = self.cross(query, memory, memory, need_weights=True,
                                       average_attn_weights=True)
        decoded = self.norm(query + attended).squeeze(1)
        prediction = self.head(torch.cat([decoded, context], dim=-1)).squeeze(-1)
        return prediction, weights.squeeze(1)


def build_model(branch: str, step_size: int, context_size: int) -> nn.Module:
    if branch == "lstm":
        return AttentionLSTM(step_size, context_size)
    if branch == "transformer":
        return TransformerForecaster(step_size, context_size)
    raise ValueError(f"Rama desconocida: {branch}")
