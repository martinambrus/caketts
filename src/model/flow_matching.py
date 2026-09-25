"""
OT-CFM decoder (Matcha-TTS convention) with a fixed-width 1-D U-Net estimator.

Fixes relative to v1:
  * DIRECTION: v1 trained with data at t=0 and noise at t=1 (target x1 - x0) but sampled
    from noise at t=0 integrating forward, i.e. it ran the data->noise flow and could only
    output noise. Here, as in Matcha-TTS: noise z at t=0, data x1 at t=1,
        y_t = (1 - (1 - sigma_min) t) z + t x1,   u = x1 - (1 - sigma_min) z,
    and Euler integration from t=0 to t=1. sigma_min is actually used.
  * WIDTH: v1 doubled channels at every U-Net level (hidden * 2**i); with the config's
    n_layers: 6 that is an 8,192-channel bottleneck (~1e9 parameters). Here the width is
    fixed per level, as in Matcha-TTS.
  * every convolution and norm is masked; lengths are padded to a multiple of 2**levels.
Optional robustness objectives (off by default; enable after the baseline works):
  * condition dropout -> classifier-free guidance at inference and per-sample
    conditional/unconditional losses for Self-Purifying Flow Matching (SPFM)
  * RobustSpeechFlow-style negatives (repeat / skip corruptions of the target), reimplemented
    from the paper's description (no official code) — validate with an ablation.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = 1000.0 * t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, dim: int, t_dim: int, groups: int = 8):
        super().__init__()
        self.n1, self.n2 = nn.GroupNorm(groups, dim), nn.GroupNorm(groups, dim)
        self.c1 = nn.Conv1d(dim, dim, 3, padding=1)
        self.c2 = nn.Conv1d(dim, dim, 3, padding=1)
        self.t = nn.Linear(t_dim, 2 * dim)

    def forward(self, x, mask, temb):
        h = self.c1(F.silu(self.n1(x * mask)) * mask)
        scale, shift = self.t(temb).unsqueeze(-1).chunk(2, dim=1)
        h = self.n2(h * mask) * (1 + scale) + shift
        h = self.c2(F.silu(h) * mask)
        return (x + h) * mask


class UNetEstimator(nn.Module):
    """v(x_t, t | mu_y, cond) with a fixed channel width at every level."""

    def __init__(self, n_mels: int, cond_dim: int = 0, dim: int = 256, levels: int = 2, blocks_per_level: int = 1):
        super().__init__()
        self.levels = levels
        t_dim = dim * 4
        self.t_mlp = nn.Sequential(nn.Linear(dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        self.dim = dim
        self.inp = nn.Conv1d(2 * n_mels + cond_dim, dim, 1)
        self.down = nn.ModuleList([nn.ModuleList([ResBlock(dim, t_dim) for _ in range(blocks_per_level)]) for _ in range(levels)])
        self.downsample = nn.ModuleList([nn.Conv1d(dim, dim, 3, stride=2, padding=1) for _ in range(levels)])
        self.mid = nn.ModuleList([ResBlock(dim, t_dim) for _ in range(2)])
        self.upsample = nn.ModuleList([nn.ConvTranspose1d(dim, dim, 4, stride=2, padding=1) for _ in range(levels)])
        self.up_fuse = nn.ModuleList([nn.Conv1d(2 * dim, dim, 1) for _ in range(levels)])
        self.up = nn.ModuleList([nn.ModuleList([ResBlock(dim, t_dim) for _ in range(blocks_per_level)]) for _ in range(levels)])
        self.out = nn.Sequential(nn.GroupNorm(8, dim), nn.SiLU(), nn.Conv1d(dim, n_mels, 1))
        nn.init.zeros_(self.out[-1].weight); nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, mask, mu, t, cond: Optional[torch.Tensor] = None):
        T = x.shape[-1]
        mult = 2 ** self.levels
        pad = (mult - T % mult) % mult
        if pad:
            x, mu, mask = (F.pad(a, (0, pad)) for a in (x, mu, mask))
        h = [x, mu]
        if cond is not None:
            h.append(cond.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        h = self.inp(torch.cat(h, dim=1)) * mask
        temb = self.t_mlp(timestep_embedding(t, self.dim))
        skips, masks = [], [mask]
        for blocks, ds in zip(self.down, self.downsample):
            for b in blocks:
                h = b(h, masks[-1], temb)
            skips.append(h)
            h = ds(h * masks[-1])
            masks.append(masks[-1][:, :, ::2])
            h = h * masks[-1]
        for b in self.mid:
            h = b(h, masks[-1], temb)
        for i, (us, fuse, blocks) in enumerate(zip(self.upsample, self.up_fuse, self.up)):
            masks.pop()
            h = us(h * 1.0)[:, :, : masks[-1].shape[-1]] * masks[-1]
            h = fuse(torch.cat([h, skips.pop()], dim=1)) * masks[-1]
            for b in blocks:
                h = b(h, masks[-1], temb)
        v = self.out(h * mask) * mask
        return v[..., :T]


class CFMDecoder(nn.Module):
    def __init__(self, n_mels: int, cond_dim: int = 0, dim: int = 256, levels: int = 2,
                 sigma_min: float = 1e-4, p_uncond: float = 0.0):
        super().__init__()
        self.n_mels = n_mels
        self.sigma_min = sigma_min
        self.p_uncond = p_uncond  # > 0 enables CFG / SPFM (optional, Step 7.3)
        self.estimator = UNetEstimator(n_mels, cond_dim, dim, levels)

    # ---- training ------------------------------------------------------------------------
    def loss(self, x1: torch.Tensor, mask: torch.Tensor, mu: torch.Tensor, cond: Optional[torch.Tensor] = None,
             drop_cond: Optional[torch.Tensor] = None, per_sample: bool = False,
             negatives: Optional[torch.Tensor] = None, neg_weight: float = 0.0):
        """
        x1: target mel (B, n_mels, T); mu: aligned prior mean mu_y (B, n_mels, T).
        drop_cond: (B,) bool, replace mu with zeros for those items (unconditional branch).
        negatives: optional corrupted targets (B, n_mels, T) for contrastive robustness.
        """
        B = x1.shape[0]
        t = torch.rand(B, device=x1.device)
        z = torch.randn_like(x1)
        tt = t[:, None, None]
        y = (1 - (1 - self.sigma_min) * tt) * z + tt * x1
        u = x1 - (1 - self.sigma_min) * z
        if drop_cond is None and self.training and self.p_uncond > 0:
            drop_cond = torch.rand(B, device=x1.device) < self.p_uncond
        if drop_cond is not None:
            mu = mu * (~drop_cond).float()[:, None, None]
        v = self.estimator(y, mask, mu, t, cond)
        denom = mask.sum(dim=(1, 2)) * self.n_mels
        l_pos = ((v - u) ** 2 * mask).sum(dim=(1, 2)) / denom
        out = l_pos
        if negatives is not None and neg_weight > 0:
            u_neg = negatives - (1 - self.sigma_min) * z
            l_neg = ((v - u_neg) ** 2 * mask).sum(dim=(1, 2)) / denom
            out = l_pos - neg_weight * l_neg
        return out if per_sample else out.mean()

    # ---- inference -----------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, mu: torch.Tensor, mask: torch.Tensor, n_steps: int = 10, temperature: float = 1.0,
               cond: Optional[torch.Tensor] = None, guidance: float = 0.0) -> torch.Tensor:
        """Euler from t=0 (noise) to t=1 (data). guidance > 0 needs a model trained with p_uncond > 0."""
        x = torch.randn_like(mu) * temperature
        ts = torch.linspace(0, 1, n_steps + 1, device=mu.device)
        for i in range(n_steps):
            t = ts[i].expand(mu.shape[0])
            v = self.estimator(x, mask, mu, t, cond)
            if guidance > 0:
                v_u = self.estimator(x, mask, torch.zeros_like(mu), t, cond)
                v = v + guidance * (v - v_u)
            x = x + (ts[i + 1] - ts[i]) * v
        return x * mask


# ---- optional robustness helpers --------------------------------------------------------

def repeat_skip_negatives(x1: torch.Tensor, lengths: torch.Tensor, floor: float = 0.0,
                          repeat_frac=(0.2, 0.4)) -> torch.Tensor:
    """
    Length-preserving corruptions of the target that look like TTS failures
    (RobustSpeechFlow, May 2026): half the batch gets a 'repeat' (a 20-40% span overwritten
    with content from another span), half a 'skip' (the tail shifted forward, end padded
    with silence `floor`, i.e. the normalised value of a silent frame).
    """
    neg = x1.clone()
    for b in range(x1.shape[0]):
        n = int(lengths[b])
        span = max(1, int(n * torch.empty(1).uniform_(*repeat_frac).item()))
        if n <= span + 1:
            continue
        if b % 2 == 0:  # repeat
            src = torch.randint(0, n - span, (1,)).item()
            dst = torch.randint(0, n - span, (1,)).item()
            neg[b, :, dst:dst + span] = x1[b, :, src:src + span]
        else:           # skip
            start = torch.randint(0, n - span, (1,)).item()
            tail = x1[b, :, start + span:n]
            neg[b, :, start:start + tail.shape[-1]] = tail
            neg[b, :, start + tail.shape[-1]:n] = floor
    return neg


def spfm_route(loss_cond: torch.Tensor, loss_uncond: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """
    Self-Purifying Flow Matching (arXiv 2509.19091): items whose conditional loss is not better
    than their unconditional loss look mislabeled (text does not match audio) -> train them
    unconditionally. Returns a (B,) bool mask of items to route to the unconditional branch.
    """
    return loss_cond > loss_uncond - margin
