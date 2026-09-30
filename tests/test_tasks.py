import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seed_utils import make_torch_generator, seed_everything
from tasks import build_task
from train_transformer_helpers import (
    CheckpointSettings, apply_config_overrides, make_checkpoint_steps,
)
from transformer_model import IGNORE_INDEX


class TaskTests(unittest.TestCase):
    def test_induction(self):
        L = 8
        t = build_task({"name": "induction", "vocab_size": 32, "half_len": L}, 0)
        x, y = t.train_batch(16, make_torch_generator(0))
        self.assertEqual(x.shape, (16, 2 * L - 1))
        for i in range(L, 2 * L - 1):
            self.assertTrue(torch.equal(y[:, i], x[:, i + 1 - L]))
        self.assertTrue((y[:, :L] == IGNORE_INDEX).all())
        self.assertEqual(t.attention_target_mask(x)[0].sum().item(), L - 1)

    def test_grokking(self):
        t = build_task({"name": "grokking", "p": 11, "train_frac": 0.5}, 0)
        for _, (x, y) in t.eval_sets().items():
            self.assertTrue(torch.equal(y[:, 2], (x[:, 0] + x[:, 1]) % 11))
        self.assertEqual(
            len(set(t.train_idx.tolist()) & set(t.val_idx.tolist())), 0)
        x, _ = t.train_batch(10 ** 6, make_torch_generator(0))
        self.assertEqual(len(x), len(t.train_idx))

    def test_icl(self):
        t = build_task({"name": "icl", "n_items": 10, "n_labels": 4,
                        "n_shots": 5}, 0)
        for _, (x, y) in t.eval_sets().items():
            mask = t.attention_target_mask(x)
            for b in range(len(x)):
                keys = mask[b, -1].nonzero().flatten()
                self.assertGreater(len(keys), 0)   # query always in context
                self.assertTrue((x[b, keys] == y[b, -1]).all())
        x, y = t.eval_sets()["fixed_map"]
        self.assertTrue(torch.equal(y[:, -1] - 10, t.fixed_map[x[:, -1]]))

    def test_seed_reproducible(self):
        a = build_task({"name": "icl"}, 5).train_batch(4, make_torch_generator(1))
        b = build_task({"name": "icl"}, 5).train_batch(4, make_torch_generator(1))
        self.assertTrue(torch.equal(a[0], b[0]))


class ConfigTests(unittest.TestCase):
    def test_overrides(self):
        cfg = {}
        apply_config_overrides(cfg, ["training.train_steps=5", "model.lr=1e-3"])
        self.assertEqual(cfg["training"]["train_steps"], 5)
        self.assertAlmostEqual(cfg["model"]["lr"], 1e-3)

    def test_checkpoint_steps(self):
        s = make_checkpoint_steps(1000, CheckpointSettings(200, 100, 400, False))
        self.assertEqual(s, [0, 100, 200, 400, 800, 1000])

    def test_seed_everything_seeds_torch(self):
        seed_everything(3)
        a = torch.rand(1)
        seed_everything(3)
        b = torch.rand(1)
        self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()