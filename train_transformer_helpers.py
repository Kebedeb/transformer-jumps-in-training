"""Settings, config overrides, evaluation, and checkpoint helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from transformer_model import IGNORE_INDEX


@dataclass(frozen=True)
class ModelSettings:
    n_layers: int = 2
    n_heads: int = 4
    n_embd: int = 64
    mlp_multiplier: int = 4


@dataclass(frozen=True)
class TrainingSettings:
    batch_size: int = 64
    train_steps: int = 10000
    eval_interval: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999


@dataclass(frozen=True)
class CheckpointSettings:
    """Schedule: step 0, every `dense_every` up to `dense_until`, then every
    `sparse_every`, plus the final step. Set an interval to 0 to disable it."""
    dense_until: int = 1000
    dense_every: int = 50
    sparse_every: int = 500
    save_optimizer: bool = False


@dataclass(frozen=True)
class OutputSettings:
    create_run_dir: bool = True
    save_checkpoints: bool = True


@dataclass(frozen=True)
class ExperimentSettings:
    seed: int
    device: str
    task: dict
    model: ModelSettings
    training: TrainingSettings
    checkpoints: CheckpointSettings
    output: OutputSettings


def parse_override_value(raw_value: str):
    """Parse a raw override value and convert it to the appropriate type."""
    lowered = raw_value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    try:
        return int(raw_value)
    except ValueError:
        pass
    try:
        return float(raw_value)
    except ValueError:
        pass
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return raw_value


def apply_config_overrides(config: dict[str, Any],
                           overrides: list[str]) -> list[tuple[str, object]]:
    """Apply dotted-key overrides (e.g. training.train_steps=500) in place."""
    applied: list[tuple[str, object]] = []
    for raw_override in overrides:
        if "=" not in raw_override:
            raise ValueError(
                f"Invalid override '{raw_override}'. Expected format key=value.")
        key_path, raw_value = raw_override.split("=", 1)
        key_path = key_path.strip()
        if not key_path:
            raise ValueError(
                f"Invalid override '{raw_override}'. Key cannot be empty.")
        value = parse_override_value(raw_value.strip())
        keys = [k.strip() for k in key_path.split(".") if k.strip()]
        if not keys:
            raise ValueError(
                f"Invalid override '{raw_override}'. Key cannot be empty.")
        target = config
        for key in keys[:-1]:
            existing = target.get(key)
            if existing is None:
                target[key] = {}
                existing = target[key]
            if not isinstance(existing, dict):
                raise ValueError(
                    f"Override path '{key_path}' conflicts with non-dict key '{key}'.")
            target = existing
        target[keys[-1]] = value
        applied.append((key_path, value))
    return applied


def load_experiment_settings(config: dict[str, Any], device: str) -> ExperimentSettings:
    """Build typed settings from a config dictionary."""
    if "task" not in config or "name" not in config["task"]:
        raise ValueError("Config must contain task.name (induction|grokking|icl).")
    m = dict(config.get("model", {}))
    t = dict(config.get("training", {}))
    c = dict(config.get("checkpoints", {}))
    o = dict(config.get("output", {}))
    return ExperimentSettings(
        seed=int(config.get("seed", 1234)),
        device=device,
        task=dict(config["task"]),
        model=ModelSettings(
            n_layers=int(m.get("n_layers", 2)),
            n_heads=int(m.get("n_heads", 4)),
            n_embd=int(m.get("n_embd", 64)),
            mlp_multiplier=int(m.get("mlp_multiplier", 4)),
        ),
        training=TrainingSettings(
            batch_size=int(t.get("batch_size", 64)),
            train_steps=int(t.get("train_steps", 10000)),
            eval_interval=int(t.get("eval_interval", 100)),
            learning_rate=float(t.get("learning_rate", 1e-3)),
            weight_decay=float(t.get("weight_decay", 0.0)),
            beta1=float(t.get("beta1", 0.9)),
            beta2=float(t.get("beta2", 0.999)),
        ),
        checkpoints=CheckpointSettings(
            dense_until=int(c.get("dense_until", 1000)),
            dense_every=int(c.get("dense_every", 50)),
            sparse_every=int(c.get("sparse_every", 500)),
            save_optimizer=bool(c.get("save_optimizer", False)),
        ),
        output=OutputSettings(
            create_run_dir=bool(o.get("create_run_dir", True)),
            save_checkpoints=bool(o.get("save_checkpoints", True)),
        ),
    )


def make_checkpoint_steps(train_steps: int, s: CheckpointSettings) -> list[int]:
    """Sorted steps at which to checkpoint (always includes 0 and the end)."""
    steps = {0, train_steps}
    if s.dense_every > 0:
        steps.update(range(s.dense_every,
                           min(s.dense_until, train_steps) + 1, s.dense_every))
    if s.sparse_every > 0:
        steps.update(range(s.sparse_every, train_steps + 1, s.sparse_every))
    return sorted(steps)


@torch.no_grad()
def evaluate(model: torch.nn.Module,
             eval_sets: dict[str, tuple[torch.Tensor, torch.Tensor]],
             device: str) -> dict[str, float]:
    """Loss and accuracy (supervised positions only) for each eval set."""
    was_training = model.training
    model.eval()
    out: dict[str, float] = {}
    for name, (x, y) in eval_sets.items():
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        supervised = y != IGNORE_INDEX
        acc = (logits.argmax(dim=-1)[supervised] == y[supervised]).float().mean()
        out[f"{name}_loss"] = float(loss.item())
        out[f"{name}_acc"] = float(acc.item())
    model.train(was_training)
    return out


def save_metrics_csv(output_path: Path, rows: list[dict[str, float]]) -> None:
    """Write dict rows to CSV (columns = union of keys, 'step' first)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    if "step" in columns:
        columns.remove("step")
        columns.insert(0, "step")
    with output_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(checkpoint_path: Path, model: torch.nn.Module, step: int,
                    model_kwargs: dict[str, int], config: dict[str, Any],
                    optimizer: torch.optim.Optimizer | None = None) -> None:
    """Save weights (+ optionally optimizer) with everything needed to rebuild."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "model_kwargs": model_kwargs,
        "config": config,
        "step": step,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, checkpoint_path)