import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformer_model import IGNORE_INDEX, TinyTransformerLM


def make(**kw):
    args = dict(vocab_size=16, block_size=8, n_layers=2, n_heads=2,
                n_embd=8, mlp_multiplier=2)
    args.update(kw)
    return TinyTransformerLM(**args)


class ModelTests(unittest.TestCase):
    def test_shapes_and_scalar_loss(self):
        m = make()
        x = torch.randint(0, 16, (4, 6))
        logits, loss = m(x, torch.randint(0, 16, (4, 6)))
        self.assertEqual(logits.shape, (4, 6, 16))
        self.assertEqual(loss.shape, torch.Size([]))

    def test_internals_shapes(self):
        m = make()
        x = torch.randint(0, 16, (3, 5))
        _, _, it = m(x, return_internals=True)
        self.assertEqual(len(it["attn"]), 2)
        self.assertEqual(it["attn"][0].shape, (3, 2, 5, 5))
        self.assertEqual(len(it["resid"]), 3)
        self.assertEqual(it["resid"][0].shape, (3, 5, 8))
        self.assertTrue(torch.allclose(it["attn"][0].sum(-1),
                                       torch.ones(3, 2, 5), atol=1e-5))
        # causal: position 0 attends to nothing after itself
        self.assertEqual(float(it["attn"][0][:, :, 0, 1:].abs().sum()), 0.0)

    def test_bad_head_split_asserts(self):
        with self.assertRaises(AssertionError):
            make(n_embd=10, n_heads=3)

    def test_causality(self):
        m = make().eval()
        x = torch.randint(0, 16, (1, 6))
        x2 = x.clone()
        x2[0, -1] = (x2[0, -1] + 1) % 16
        a, _ = m(x)
        b, _ = m(x2)
        self.assertTrue(torch.allclose(a[:, :-1], b[:, :-1], atol=1e-6))

    def test_ignore_index(self):
        m = make()
        x = torch.randint(0, 16, (2, 4))
        y = torch.full((2, 4), IGNORE_INDEX)
        y[:, -1] = 3
        _, loss = m(x, y)
        logits, _ = m(x)
        ref = torch.nn.functional.cross_entropy(logits[:, -1], y[:, -1])
        self.assertAlmostEqual(float(loss), float(ref), places=5)

    def test_generate(self):
        m = make().eval()
        self.assertEqual(m.generate(torch.randint(0, 16, (1, 3)), 5).shape, (1, 8))


if __name__ == "__main__":
    unittest.main()