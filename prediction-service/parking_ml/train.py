"""Entrena las dos ramas por horizonte, calibra τ y publica los artefactos.

Uso: python -m parking_ml.train --data ../data/occupancy_15min.csv --out artifacts

Protocolo: τ se elige solo con validación; prueba se usa una única vez para el
reporte. Cada combinación rama-horizonte se entrena con varias semillas para
distinguir una diferencia real entre ramas del ruido de la inicialización.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from . import BRANCHES, HORIZONS, WINDOW_HOURS
from .dataset import Samples, build_samples, chronological_split, load_hourly, split_samples
from .features import STEP_FEATURES, TARGET_FEATURES
from .metrics import compute
from .models import build_model

BATCH_SIZE = 128
MAX_EPOCHS = 60
PATIENCE = 8
LEARNING_RATE = 1e-3
VERSION = "v1"


def tensors(samples: Samples, device: torch.device) -> tuple[torch.Tensor, ...]:
    return (torch.from_numpy(samples.window).to(device),
            torch.from_numpy(samples.target_context).to(device),
            torch.from_numpy(samples.y).to(device))


@torch.no_grad()
def predict(model: nn.Module, window: torch.Tensor, context: torch.Tensor,
            batch_size: int = 2048) -> np.ndarray:
    model.eval()
    outputs = [model(window[i:i + batch_size], context[i:i + batch_size])[0].cpu().numpy()
               for i in range(0, len(window), batch_size)]
    return np.concatenate(outputs)


def train_one(branch: str, train: Samples, val: Samples, seed: int,
              device: torch.device) -> tuple[nn.Module, float, int]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(branch, train.window.shape[2], train.target_context.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    loss_fn = nn.L1Loss()
    train_window, train_context, train_y = tensors(train, device)
    val_window, val_context, _ = tensors(val, device)
    loader = DataLoader(TensorDataset(train_window, train_context, train_y),
                        batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    best_mae, best_state, best_epoch, waited = float("inf"), None, 0, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for window, context, target in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(window, context)[0], target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_mae = float(np.mean(np.abs(predict(model, val_window, val_context) - val.y)))
        if val_mae < best_mae:
            best_mae, best_epoch, waited = val_mae, epoch + 1, 0
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
        else:
            waited += 1
            if waited >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best_mae, best_epoch


def permutation_importance(model: nn.Module, samples: Samples, device: torch.device,
                           seed: int = 0) -> dict[str, float]:
    """Aumento del MAE al desordenar cada variable de la ventana entre muestras."""
    rng = np.random.default_rng(seed)
    window, context, _ = tensors(samples, device)
    base = float(np.mean(np.abs(predict(model, window, context) - samples.y)))
    importance = {}
    for index, name in enumerate(STEP_FEATURES):
        shuffled = samples.window.copy()
        shuffled[:, :, index] = shuffled[rng.permutation(len(shuffled)), :, index]
        mae = np.mean(np.abs(predict(model, torch.from_numpy(shuffled).to(device), context) - samples.y))
        importance[name] = float(mae - base)
    return importance


@torch.no_grad()
def mean_attention(model: nn.Module, samples: Samples, device: torch.device) -> list[float]:
    """Atención media por hora de la ventana, de la más antigua a la más reciente."""
    model.eval()
    window, context, _ = tensors(samples, device)
    collected = [model(window[i:i + 2048], context[i:i + 2048])[1].cpu().numpy()
                 for i in range(0, len(window), 2048)]
    return np.concatenate(collected).mean(axis=0).round(5).tolist()


def calibrate_tau(val_mae: dict[str, dict[int, float]]) -> tuple[int, dict[str, float]]:
    """Elige τ minimizando el MAE medio en validación. τ=0: todo Transformer; τ=24: todo LSTM."""
    scores = {}
    for tau in (0, *HORIZONS):
        errors = [val_mae["lstm"][h] if h <= tau else val_mae["transformer"][h] for h in HORIZONS]
        scores[str(tau)] = float(np.mean(errors))
    best = min(scores, key=lambda key: (round(scores[key], 6), int(key)))
    return int(best), scores


def publish_serving_data(hourly: pd.DataFrame, directory: Path, kind: str, split: dict) -> None:
    """Deja en el directorio de datos lo que necesita el Módulo 3 para servir el demo."""
    hourly.to_csv(directory / "occupancy_hourly.csv", index=False)
    (directory / "metadata.json").write_text(json.dumps({"dataset_kind": kind, **split}, indent=2), encoding="utf-8")
    zones_path = directory / "zones.json"
    if not zones_path.exists():  # Con datos reales no lo genera el simulador.
        capacities = hourly.groupby("zone")["capacity"].last()
        zones_path.write_text(json.dumps(
            [{"id": zone, "name": f"Zona {zone}", "capacity": int(capacity)} for zone, capacity in capacities.items()],
            ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("../data/occupancy_15min.csv"))
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--dataset-kind", choices=["synthetic", "field", "public"], default="synthetic")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    hourly = load_hourly(args.data)
    zones = sorted(hourly["zone"].unique().tolist())
    split = chronological_split(hourly)
    data_hash = hashlib.sha256(args.data.read_bytes()).hexdigest()[:16]
    print(f"Dispositivo: {device} | zonas: {zones} | val desde {split.val_start} | test desde {split.test_start}")

    report: dict = {"horizons": {}, "importance": {}, "attention": {}}
    registry: dict = {"models": {}}
    val_mae: dict[str, dict[int, float]] = {branch: {} for branch in BRANCHES}
    test_predictions: dict[str, dict[int, np.ndarray]] = {branch: {} for branch in BRANCHES}
    samples_by_horizon: dict[int, Samples] = {}

    for horizon in HORIZONS:
        train, val, test = split_samples(build_samples(hourly, horizon, zones), split)
        samples_by_horizon[horizon] = test
        entry: dict = {"baseline": compute(test.y, test.week_ago),
                       "samples": {"train": len(train), "val": len(val), "test": len(test)}}
        val_baseline = float(np.mean(np.abs(val.week_ago - val.y)))

        for branch in BRANCHES:
            runs = []
            for seed in range(args.seeds):
                model, mae, epoch = train_one(branch, train, val, 1000 * horizon + seed, device)
                window, context, _ = tensors(test, device)
                predicted = predict(model, window, context)
                runs.append({"model": model, "val_mae": mae, "epoch": epoch, "predicted": predicted,
                             "test": compute(test.y, predicted)})
            best = min(runs, key=lambda run: run["val_mae"])
            name = f"{branch}-h{horizon}-{VERSION}"
            torch.save(best["model"].state_dict(), args.out / f"{name}.pt")

            val_mae[branch][horizon] = best["val_mae"]
            test_predictions[branch][horizon] = best["predicted"]
            test_maes = [run["test"]["mae"] for run in runs]
            entry[branch] = {**best["test"], "val_mae": best["val_mae"],
                             "test_mae_seed_mean": float(np.mean(test_maes)),
                             "test_mae_seed_std": float(np.std(test_maes)),
                             "best_epoch": best["epoch"]}
            registry["models"][f"{branch}:{horizon}"] = {
                "name": name, "branch": branch, "horizon": horizon, "file": f"{name}.pt",
                "version": VERSION, "val_mae": best["val_mae"], "data_sha256": data_hash}
            report["importance"][f"{branch}:{horizon}"] = permutation_importance(best["model"], val, device)
            report["attention"][f"{branch}:{horizon}"] = mean_attention(best["model"], test, device)
            print(f"h={horizon:>2} {branch:<11} val MAE={best['val_mae']:.4f} "
                  f"test MAE={best['test']['mae']:.4f} (semillas {np.mean(test_maes):.4f}±{np.std(test_maes):.4f})")
        print(f"h={horizon:>2} baseline    val MAE={val_baseline:.4f} test MAE={entry['baseline']['mae']:.4f}")
        report["horizons"][str(horizon)] = entry

    tau, tau_scores = calibrate_tau(val_mae)
    for horizon in HORIZONS:
        test = samples_by_horizon[horizon]
        branch = "lstm" if horizon <= tau else "transformer"
        report["horizons"][str(horizon)]["router"] = {
            **compute(test.y, test_predictions[branch][horizon]), "branch": branch}

    means = {name: {metric: float(np.mean([report["horizons"][str(h)][name][metric] for h in HORIZONS]))
                    for metric in ("mape", "mae", "mse", "r2")}
             for name in ("lstm", "transformer", "router", "baseline")}
    crossover = next((h for h in HORIZONS if all(
        report["horizons"][str(later)]["transformer"]["mae"] < report["horizons"][str(later)]["lstm"]["mae"]
        for later in HORIZONS if later >= h)), None)
    report.update({
        "dataset_kind": args.dataset_kind,
        "tau": tau, "tau_scores_val_mae": tau_scores, "means": means,
        "hypotheses": {
            "h1_crossover_horizon": crossover,
            "h1_holds": crossover is not None,
            "h2_holds": means["router"]["mae"] < min(means["lstm"]["mae"], means["transformer"]["mae"],
                                                     means["baseline"]["mae"]),
        },
        "split": {"val_start": str(split.val_start), "test_start": str(split.test_start), "end": str(split.end)},
    })

    registry.update({
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_kind": args.dataset_kind, "data_sha256": data_hash, "tau": tau, "zones": zones,
        "window_hours": WINDOW_HOURS, "horizons": list(HORIZONS),
        "step_features": list(STEP_FEATURES), "target_features": list(TARGET_FEATURES),
        "split": report["split"],
    })
    (args.out / "registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    publish_serving_data(hourly, args.data.parent, args.dataset_kind, report["split"])
    print(f"tau calibrado: {tau} | medias: {json.dumps(means, indent=1)}")
    print(f"Hipotesis: {report['hypotheses']}")


if __name__ == "__main__":
    main()
