"""Thin wrapper around Weights & Biases. Does nothing if disabled/unavailable."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class Tracker:
    def __init__(self, run=None) -> None:
        self.run = run

    @property
    def enabled(self) -> bool:
        return self.run is not None

    def log(self, metrics: dict[str, Any]) -> None:
        if self.run is not None:
            self.run.log(metrics)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
            self.run = None


def init_tracker(config: dict[str, Any], name: str, job_type: str,
                 run_dir: Path | None = None) -> Tracker:
    """Start a W&B run if config['wandb']['enabled'] is true.

    Runs are grouped by config['experiment_name'] (so all seeds of one task
    overlay in the UI); job_type is 'train' or 'analysis'.
    """
    cfg = dict(config.get("wandb", {}))
    if not cfg.get("enabled", False):
        return Tracker()
    try:
        import wandb
    except ImportError:
        print("wandb is not installed; continuing without tracking.")
        return Tracker()

    run = wandb.init(
        project=cfg.get("project", "precursor-emergence"),
        entity=cfg.get("entity"),
        name=name,
        group=str(config.get("experiment_name", "experiment")),
        job_type=job_type,
        tags=cfg.get("tags"),
        config=config,
        mode=cfg.get("mode", "online"),      # "online" | "offline" | "disabled"
        dir=str(run_dir) if run_dir is not None else None,
    )
    # Use our own 'step' column as the x-axis for every metric.
    run.define_metric("step")
    run.define_metric("*", step_metric="step")
    return Tracker(run)