"""Tiny causal Transformer with access to attention weights and residual stream."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Targets equal to this value are skipped by the loss.
IGNORE_INDEX = -100


class TransformerBlock(nn.Module):
    """Pre-LayerNorm block: x + Attn(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, n_embd: int, n_heads: int, block_size: int,
                 mlp_multiplier: int) -> None:
        super().__init__()
        assert n_embd % n_heads == 0, (
            f"n_embd ({n_embd}) must be divisible by n_heads ({n_heads})")
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, mlp_multiplier * n_embd),
            nn.ReLU(),
            nn.Linear(mlp_multiplier * n_embd, n_embd),
        )
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(block_size, block_size, dtype=torch.bool),
                       diagonal=1),
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        """Returns (x, attn_weights). attn_weights is (B, H, T, T) if
        return_attn else None."""
        seq_len = x.size(1)
        mask = self.causal_mask[:seq_len, :seq_len]

        attn_input = self.ln1(x)
        attn_out, attn_weights = self.attn(
            attn_input, attn_input, attn_input,
            attn_mask=mask,
            need_weights=return_attn,
            average_attn_weights=False,
        )
        # Residual (skip) connection: each block adds a correction to the
        # running "residual stream" instead of overwriting it.
        x = x + attn_out

        x = x + self.ff(self.ln2(x))
        return x, attn_weights


class TinyTransformerLM(nn.Module):
    """Tiny causal Transformer language model over integer tokens."""

    def __init__(self, vocab_size: int, block_size: int, n_layers: int,
                 n_heads: int, n_embd: int, mlp_multiplier: int) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd=n_embd, n_heads=n_heads,
                             block_size=block_size, mlp_multiplier=mlp_multiplier)
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(n_embd)
        self.lm_unembedding = nn.Linear(n_embd, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                return_internals: bool = False):
        """targets[b, t] is the token to predict from position t (tasks
        pre-shift it); IGNORE_INDEX positions are skipped.

        Returns (logits, loss), or (logits, loss, internals) if
        return_internals, where internals = {
            "attn":  [per-layer (B, H, T, T)],
            "resid": [embedding, after block 0, after block 1, ...]
        }."""
        batch_size, seq_len = idx.shape
        if seq_len > self.block_size:
            raise ValueError(
                f"seq_len {seq_len} exceeds block_size {self.block_size}")
        positions = torch.arange(seq_len, device=idx.device)

        x = self.token_emb(idx) + self.pos_emb(positions)[None, :, :]
        attns, resid = [], [x]
        for block in self.blocks:
            x, attn_weights = block(x, return_attn=return_internals)
            if return_internals:
                attns.append(attn_weights)
                resid.append(x)
        x = self.ln_final(x)

        logits = self.lm_unembedding(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(batch_size * seq_len, -1),
                targets.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
        if return_internals:
            return logits, loss, {"attn": attns, "resid": resid}
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Greedy generation: (batch, seq) -> (batch, seq + max_new_tokens)."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.block_size:])
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            idx = torch.cat([idx, next_token], dim=1)
        return idx