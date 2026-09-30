"""Representation and weight statistics (candidate precursor signals)."""

from __future__ import annotations

import torch


def _flat(h: torch.Tensor) -> torch.Tensor:
    return h.reshape(-1, h.shape[-1]).float()


def effective_rank(h: torch.Tensor) -> float:
    """exp(entropy of normalised singular values) of centred activations."""
    x = _flat(h)
    x = x - x.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(x)
    p = s / s.sum().clamp(min=1e-12)
    return float(torch.exp(-(p * torch.log(p + 1e-12)).sum()))


def participation_ratio(h: torch.Tensor) -> float:
    """(sum lambda)^2 / sum lambda^2 over covariance eigenvalues."""
    x = _flat(h)
    x = x - x.mean(dim=0, keepdim=True)
    lam = torch.linalg.svdvals(x) ** 2
    return float(lam.sum() ** 2 / (lam ** 2).sum().clamp(min=1e-12))


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear CKA between two activation sets over the same inputs."""
    x, y = _flat(x), _flat(y)
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    num = torch.linalg.norm(y.T @ x) ** 2
    den = torch.linalg.norm(x.T @ x) * torch.linalg.norm(y.T @ y)
    return float(num / den.clamp(min=1e-12))


@torch.no_grad()
def representation_stats(resid, prev_resid=None) -> dict[str, float]:
    """Per residual stage: effective rank, participation ratio, mean norm,
    and CKA with the previous checkpoint's activations (if given)."""
    out: dict[str, float] = {}
    for i, h in enumerate(resid):
        out[f"resid{i}_eff_rank"] = effective_rank(h)
        out[f"resid{i}_part_ratio"] = participation_ratio(h)
        out[f"resid{i}_mean_norm"] = float(h.norm(dim=-1).mean())
        if prev_resid is not None:
            out[f"resid{i}_cka_prev"] = linear_cka(h, prev_resid[i].to(h.device))
    return out


@torch.no_grad()
def weight_stats(model: torch.nn.Module) -> dict[str, float]:
    """Total L2 weight norm plus per-block attention / MLP norms."""
    out = {"weight_norm_total": float(torch.sqrt(sum(
        (p ** 2).sum() for p in model.parameters())))}
    for name, p in model.named_parameters():
        if name.startswith("blocks."):
            layer = name.split(".")[1]
            group = "attn" if ".attn." in name else "mlp" if ".ff." in name else None
            if group:
                key = f"weight_norm_block{layer}_{group}"
                out[key] = float(torch.sqrt(out.get(key, 0.0) ** 2 + (p ** 2).sum()))
    return out