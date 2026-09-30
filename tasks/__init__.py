"""Synthetic task generators."""

from __future__ import annotations

import torch


class Task:
    name: str = "task"
    vocab_size: int
    block_size: int  # length of x

    def train_batch(self, batch_size: int, generator: torch.Generator):
        raise NotImplementedError

    def eval_sets(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Fixed held-out sets for behavioural metrics."""
        raise NotImplementedError

    def probe_batch(self) -> torch.Tensor:
        """Fixed inputs used for attention / representation signals."""
        raise NotImplementedError

    def attention_target_mask(self, x: torch.Tensor):
        """Optional (B, T, T) float mask: 1 where query q should attend to
        key k if a head implements the task's mechanism. None if N/A."""
        return None


def build_task(task_cfg: dict, seed: int) -> Task:
    """Construct a task from config['task']; 'name' selects the class."""
    params = {k: v for k, v in task_cfg.items() if k != "name"}
    name = task_cfg["name"]
    if name == "induction":
        from tasks.induction_task import InductionTask
        return InductionTask(seed=seed, **params)
    if name == "grokking":
        from tasks.grokking_task import GrokkingTask
        return GrokkingTask(seed=seed, **params)
    if name == "icl":
        from tasks.icl_task import ICLTask
        return ICLTask(seed=seed, **params)
    raise ValueError(f"Unknown task '{name}'. Use induction, grokking, or icl.")