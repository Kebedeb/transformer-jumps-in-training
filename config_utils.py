"""Utility functions for managing experiment configurations and run
directories."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load experiment configuration from JSON."""
    resolved_path = Path(config_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Config file not found: {resolved_path}")
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Config file is not a file: {resolved_path}")
    with resolved_path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def create_run_dir(config: dict[str, Any], base_dir: Path | None = None) -> Path:
    """Create a unique directory under runs/ for the current training run."""
    base = PROJECT_ROOT / "runs" if base_dir is None else Path(base_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    experiment_name = str(config.get("experiment_name", "experiment"))
    run_dir = base / f"{timestamp}-{experiment_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_config_copy(config: dict[str, Any], run_dir: Path) -> None:
    """Persist the exact config used for a run into the run directory."""
    with (run_dir / "config.json").open("w", encoding="utf-8") as file_obj:
        json.dump(config, file_obj, indent=2)
        file_obj.write("\n")