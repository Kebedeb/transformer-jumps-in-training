"""Induction: a random sequence of distinct tokens shown twice
([t0..t_{L-1}, t0..t_{L-1}]). The second copy is predictable only by looking
up what followed the same token the first time."""

from __future__ import annotations

import torch

from tasks import Task
from transformer_model import IGNORE_INDEX


class InductionTask(Task):
    name = "induction"

    def __init__(self, vocab_size: int = 64, half_len: int = 16,
                 n_eval: int = 512, seed: int = 1234) -> None:
        if half_len > vocab_size:
            raise ValueError("half_len must be <= vocab_size (tokens are distinct).")
        self.vocab_size = vocab_size
        self.half_len = half_len
        self.block_size = 2 * half_len - 1
        gen = torch.Generator().manual_seed(seed + 1)
        self._eval = self._sample(n_eval, gen)

    def _sample(self, n: int, gen: torch.Generator):
        L = self.half_len
        first = torch.rand(n, self.vocab_size, generator=gen).argsort(dim=1)[:, :L]
        seq = torch.cat([first, first], dim=1)          # (n, 2L)
        x = seq[:, :-1]                                  # (n, 2L-1)
        y = seq[:, 1:].clone()
        y[:, :L] = IGNORE_INDEX   # first copy (and first repeated token) unpredictable
        return x, y

    def train_batch(self, batch_size, generator):
        return self._sample(batch_size, generator)

    def eval_sets(self):
        return {"heldout": self._eval}

    def probe_batch(self):
        return self._eval[0]

    def attention_target_mask(self, x):
        """At query position q = L + j (token t_j, second copy), an induction
        head attends to key position j + 1 (token after the first t_j)."""
        B, T = x.shape
        L = self.half_len
        mask = torch.zeros(B, T, T)
        j = torch.arange(L - 1)
        mask[:, L + j, j + 1] = 1.0
        return mask