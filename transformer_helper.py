"""Helper functions for devices, checkpoints, and model loading."""

from __future__ import annotations

from pathlib import Path

import torch

from transformer_model import TinyTransformerLM

PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_device(device_setting: str) -> str:
    """Resolve 'auto' / 'cuda' / 'cpu' to an available device string."""
    normalized = device_setting.strip().lower()
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        print("Requested device 'cuda' is unavailable. Falling back to cpu.")
        return "cpu"
    return normalized


def find_latest_run_dir(runs_dir: Path | None = None) -> Path:
    """Most recently modified run directory under runs/."""
    runs_dir = PROJECT_ROOT / "runs" if runs_dir is None else Path(runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
    run_dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No runs found under {runs_dir}")
    return max(run_dirs, key=lambda p: p.stat().st_mtime)


def list_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    """All step checkpoints (ckpt_step_XXXXXXX.pt) in a run, sorted by step."""
    ckpt_dir = Path(run_dir) / "checkpoints"
    found = [(int(p.stem.split("_")[-1]), p)
             for p in ckpt_dir.glob("ckpt_step_*.pt")]
    return sorted(found)


def load_model_from_checkpoint(path: Path, device: str = "cpu"):
    """Rebuild a model from a checkpoint (uses the saved model_kwargs)."""
    ckpt = torch.load(path, map_location=device)
    model = TinyTransformerLM(**ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt