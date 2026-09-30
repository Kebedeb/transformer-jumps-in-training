"""Per-head attention statistics. `attn` tensors have shape (B, H, T, T)."""

from __future__ import annotations

import torch


def pattern_score(attn: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """Mean attention mass each head puts on the task's target keys, averaged
    over queries that have targets. Returns (H,). This is the induction score
    when the mask comes from InductionTask.attention_target_mask."""
    mass = (attn * target_mask[:, None]).sum(dim=-1)              # (B, H, T)
    valid = (target_mask.sum(dim=-1) > 0).float()                 # (B, T)
    denom = valid.sum().clamp(min=1.0)
    return (mass * valid[:, None]).sum(dim=(0, 2)) / denom


def previous_token_score(attn: torch.Tensor) -> torch.Tensor:
    """Mean attention from position q to q-1, per head. Returns (H,)."""
    q = torch.arange(1, attn.shape[-1], device=attn.device)
    return attn[:, :, q, q - 1].mean(dim=(0, 2))


def attention_entropy(attn: torch.Tensor) -> torch.Tensor:
    """Mean entropy of each head's attention distribution. Returns (H,)."""
    ent = -(attn * torch.log(attn + 1e-12)).sum(dim=-1)           # (B, H, T)
    return ent.mean(dim=(0, 2))


@torch.no_grad()
def attention_stats(attn_list, target_mask=None) -> dict[str, float]:
    """Flat dict of per-(layer, head) scores plus attn_max_pattern."""
    out: dict[str, float] = {}
    pattern_values: list[float] = []
    for layer, attn in enumerate(attn_list):
        prev = previous_token_score(attn)
        ent = attention_entropy(attn)
        pat = (pattern_score(attn, target_mask.to(attn.device))
               if target_mask is not None else None)
        for head in range(attn.shape[1]):
            out[f"attn_L{layer}H{head}_prev_token"] = float(prev[head])
            out[f"attn_L{layer}H{head}_entropy"] = float(ent[head])
            if pat is not None:
                out[f"attn_L{layer}H{head}_pattern"] = float(pat[head])
                pattern_values.append(float(pat[head]))
    if pattern_values:
        out["attn_max_pattern"] = max(pattern_values)
    return out