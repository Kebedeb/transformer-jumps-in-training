"""Grokking: modular addition (a + b) mod p, input [a, b, '='], loss only on
the answer position. A fraction of all p^2 pairs trains; the rest validates."""

from __future__ import annotations

import torch

from tasks import Task
from transformer_model import IGNORE_INDEX


class GrokkingTask(Task):
    name = "grokking"

    def __init__(self, p: int = 113, train_frac: float = 0.3,
                 seed: int = 1234) -> None:
        self.p = p
        self.vocab_size = p + 1          # tokens 0..p-1 plus '=' (id p)
        self.block_size = 3
        a = torch.arange(p).repeat_interleave(p)
        b = torch.arange(p).repeat(p)
        eq = torch.full_like(a, p)
        self._x = torch.stack([a, b, eq], dim=1)                # (p^2, 3)
        self._y = torch.full_like(self._x, IGNORE_INDEX)
        self._y[:, 2] = (a + b) % p
        gen = torch.Generator().manual_seed(seed + 1)
        perm = torch.randperm(p * p, generator=gen)
        n_train = int(train_frac * p * p)
        self.train_idx = perm[:n_train]
        self.val_idx = perm[n_train:]

    def train_batch(self, batch_size, generator):
        n = len(self.train_idx)
        if batch_size >= n:                       # full-batch training
            idx = self.train_idx
        else:
            idx = self.train_idx[torch.randint(0, n, (batch_size,), generator=generator)]
        return self._x[idx], self._y[idx]

    def eval_sets(self):
        return {
            "train": (self._x[self.train_idx], self._y[self.train_idx]),
            "val": (self._x[self.val_idx], self._y[self.val_idx]),
        }

    def probe_batch(self):
        return self._x[self.val_idx[:512]]