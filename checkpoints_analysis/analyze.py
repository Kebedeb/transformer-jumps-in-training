"""Compute behavioural + internal signals for every checkpoint in a run.

    python -m checkpoints_analysis.analyze runs/<run_dir>
Writes <run_dir>/signals.csv (one row per checkpoint step).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from checkpoints_analysis.attention_stats import attention_stats
from checkpoints_analysis.representation_stats import representation_stats, weight_stats
from config_utils import load_config
from tasks import build_task
from train_transformer_helpers import evaluate, save_metrics_csv
from transformer_helper import (
    find_latest_run_dir, list_checkpoints, load_model_from_checkpoint, resolve_device,
)


@torch.no_grad()
def analyze_run(run_dir: Path, device: str = "auto", probe_size: int = 256) -> Path:
    run_dir = Path(run_dir)
    device = resolve_device(device)
    config = load_config(run_dir / "config.json")
    task = build_task(config["task"], int(config.get("seed", 1234)))
    eval_sets = task.eval_sets()
    probe_x = task.probe_batch()[:probe_size]
    target_mask = task.attention_target_mask(probe_x)

    checkpoints = list_checkpoints(run_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {run_dir / 'checkpoints'}")

    rows, prev_resid = [], None
    for step, path in checkpoints:
        model, _ = load_model_from_checkpoint(path, device)
        row: dict[str, float] = {"step": step}
        row.update(evaluate(model, eval_sets, device))
        _, _, internals = model(probe_x.to(device), return_internals=True)
        row.update(attention_stats(internals["attn"], target_mask))
        row.update(representation_stats(internals["resid"], prev_resid))
        row.update(weight_stats(model))
        prev_resid = [r.detach().cpu() for r in internals["resid"]]
        rows.append(row)
        if len(rows) % 20 == 0:
            print(f"analyzed {len(rows)}/{len(checkpoints)} checkpoints")

    out_path = run_dir / "signals.csv"
    save_metrics_csv(out_path, rows)
    print(f"Wrote {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?", default=None,
                        help="Run directory (default: most recent under runs/)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run_dir()
    analyze_run(run_dir, args.device)


if __name__ == "__main__":
    main()