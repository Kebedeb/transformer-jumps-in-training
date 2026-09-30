"""Train a tiny causal transformer on a synthetic task, saving checkpoints.

Run from the repo root:
    python train_transformer.py -c configs/induction_v1.json -O training.train_steps=2000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config_utils import create_run_dir, load_config, save_config_copy
from seed_utils import get_seed, make_torch_generator, seed_everything
from tasks import build_task
from train_transformer_helpers import (
    apply_config_overrides, evaluate, load_experiment_settings,
    make_checkpoint_steps, save_checkpoint, save_metrics_csv,
)
from transformer_helper import resolve_device
from transformer_model import TinyTransformerLM

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "induction_v1.json"


def train(config: dict) -> Path | None:
    """Train according to `config`. Returns the run directory (or None)."""
    seed = int(config.get("seed", get_seed()))
    seed_everything(seed)
    device = resolve_device(str(config.get("device", "auto")))
    s = load_experiment_settings(config, device)

    task = build_task(s.task, seed)
    model_kwargs = dict(
        vocab_size=task.vocab_size, block_size=task.block_size,
        n_layers=s.model.n_layers, n_heads=s.model.n_heads,
        n_embd=s.model.n_embd, mlp_multiplier=s.model.mlp_multiplier,
    )
    model = TinyTransformerLM(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=s.training.learning_rate,
        betas=(s.training.beta1, s.training.beta2),
        weight_decay=s.training.weight_decay,
    )

    run_dir = None
    if s.output.create_run_dir:
        run_dir = create_run_dir(config)
        save_config_copy(config, run_dir)

    ckpt_steps = set(make_checkpoint_steps(s.training.train_steps, s.checkpoints))
    save_ckpts = run_dir is not None and s.output.save_checkpoints
    eval_sets = task.eval_sets()
    data_gen = make_torch_generator(seed)   # separate stream for batch sampling

    print(f"Task: {task.name} | device: {device} | seed: {seed} | run dir: {run_dir}")
    print(f"Model: {model_kwargs} | params: "
          f"{sum(p.numel() for p in model.parameters())}")

    def maybe_checkpoint(step: int) -> None:
        if save_ckpts and step in ckpt_steps:
            save_checkpoint(
                run_dir / "checkpoints" / f"ckpt_step_{step:07d}.pt", model, step,
                model_kwargs, config,
                optimizer if s.checkpoints.save_optimizer else None)

    rows: list[dict[str, float]] = []

    def log_row(step: int, batch_loss: float, grad_norm: float) -> None:
        weight_norm = float(torch.sqrt(sum((p.detach() ** 2).sum()
                                           for p in model.parameters())).item())
        row = {"step": step, "train_batch_loss": batch_loss,
               "grad_norm": grad_norm, "weight_norm": weight_norm,
               **evaluate(model, eval_sets, device)}
        rows.append(row)
        shown = " | ".join(f"{k} {v:.4f}" for k, v in row.items() if k != "step")
        print(f"step {step:6d} | {shown}")

    model.train()
    maybe_checkpoint(0)
    try:
        log_row(0, float("nan"), float("nan"))
        for step in range(1, s.training.train_steps + 1):
            x, y = task.train_batch(s.training.batch_size, data_gen)
            x, y = x.to(device), y.to(device)

            # Cross-entropy over supervised positions only (y == -100 ignored).
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float("inf")))
            optimizer.step()

            if step % s.training.eval_interval == 0 or step == s.training.train_steps:
                log_row(step, float(loss.item()), grad_norm)
            maybe_checkpoint(step)
    finally:
        if run_dir is not None:
            save_metrics_csv(run_dir / "metrics.csv", rows)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train on a synthetic task.")
    parser.add_argument("-c", "--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("-O", "--override", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="Dotted override, repeatable "
                             "(e.g. training.train_steps=500)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    applied = apply_config_overrides(config, args.override)
    print(f"Config: {args.config}")
    for key, value in applied:
        print(f"  override {key}={value}")
    train(config)


if __name__ == "__main__":
    main()