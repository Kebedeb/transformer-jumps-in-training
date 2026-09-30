"""In-context learning: [item_1, label_1, ..., item_N, label_N, query_item]
-> label of query_item. Each sequence uses either a FRESH random mapping
(answerable only from context: ICL) or one GLOBAL fixed mapping (answerable
from weights or context: IWL). icl_fraction sets the training mix."""

from __future__ import annotations

import torch

from tasks import Task
from transformer_model import IGNORE_INDEX


class ICLTask(Task):
    name = "icl"

    def __init__(self, n_items: int = 32, n_labels: int = 8, n_shots: int = 8,
                 icl_fraction: float = 0.5, n_eval: int = 512,
                 seed: int = 1234) -> None:
        self.n_items, self.n_labels, self.n_shots = n_items, n_labels, n_shots
        self.icl_fraction = icl_fraction
        self.vocab_size = n_items + n_labels     # items first, then labels
        self.block_size = 2 * n_shots + 1
        gen = torch.Generator().manual_seed(seed + 1)
        self.fixed_map = torch.randint(0, n_labels, (n_items,), generator=gen)
        self._eval_fresh = self._sample(n_eval, gen, 1.0)
        self._eval_fixed = self._sample(n_eval, gen, 0.0)

    def _sample(self, n: int, gen: torch.Generator, icl_fraction: float):
        N = self.n_shots
        items = torch.randint(0, self.n_items, (n, N), generator=gen)
        use_fresh = torch.rand(n, generator=gen) < icl_fraction
        fresh_maps = torch.randint(0, self.n_labels, (n, self.n_items), generator=gen)
        maps = torch.where(use_fresh[:, None], fresh_maps,
                           self.fixed_map[None, :].expand(n, -1))
        labels = maps.gather(1, items)                              # (n, N)
        qpos = torch.randint(0, N, (n,), generator=gen)
        q_item = items.gather(1, qpos[:, None]).squeeze(1)
        q_label = labels.gather(1, qpos[:, None]).squeeze(1)
        context = torch.stack([items, labels + self.n_items], dim=2).reshape(n, 2 * N)
        x = torch.cat([context, q_item[:, None]], dim=1)            # (n, 2N+1)
        y = torch.full_like(x, IGNORE_INDEX)
        y[:, -1] = q_label + self.n_items
        return x, y

    def train_batch(self, batch_size, generator):
        return self._sample(batch_size, generator, self.icl_fraction)

    def eval_sets(self):
        return {"fresh_map": self._eval_fresh, "fixed_map": self._eval_fixed}

    def probe_batch(self):
        return self._eval_fresh[0]

    def attention_target_mask(self, x):
        """The query (last position) should attend to the label positions that
        follow earlier occurrences of the same item in the context."""
        B, T = x.shape
        N = self.n_shots
        mask = torch.zeros(B, T, T)
        match = (x[:, 0:2 * N:2] == x[:, -1:]).float()              # (B, N)
        mask[:, -1, 1:2 * N:2] = match
        return mask