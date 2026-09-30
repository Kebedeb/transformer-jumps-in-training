"""Shared driver: train, then analyze every checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from checkpoints_analysis.analyze import analyze_run
from config_utils import PROJECT_ROOT, load_config
from train_transformer import train
from train_transformer_helpers import apply_config_overrides


def run_experiment(default_config: str) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config",
                        default=str(PROJECT_ROOT / "configs" / default_config))
    parser.add_argument("-O", "--override", action="append", default=[],
                        metavar="KEY=VALUE")
    parser.add_argument("--skip-analysis", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    apply_config_overrides(config, args.override)
    run_dir = train(config)
    if run_dir is not None and not args.skip_analysis:
        analyze_run(Path(run_dir), device=str(config.get("device", "auto")))