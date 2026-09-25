"""
Durations: predictor, Monotonic Alignment Search (MAS) and length regulation.

Fixes relative to v1:
  * LENGTH REGULATION: v1 did round(d).clamp(min=0) and dropped tokens with 0 frames —
    torch.round rounds half to even, so every token predicted at <= 0.5 frames vanished
    (a skipped phoneme; several in a row = a skipped word). Here every real token gets
    at least 1 frame AFTER any length scaling: ceil(exp(logw) * length_scale).clamp(min=1).
  * DURATION PREDICTOR: v1 applied nn.LayerNorm(hidden) to a (B, hidden, T) tensor (it
    normalises the LAST dim, so it crashed whenever T != hidden) and ran unmasked convs.
    Here: channel LayerNorm, masking, detached encoder input (as in Matcha-TTS), and FiLM
    conditioning on narrator / tempo / boundary so tempo is an explicit input.
  * MAS: v1 aligned with a learned attention module that no loss trained. Here MAS maximises
    the Gaussian log-likelihood of the mel frames under the encoder's prior means (Glow-TTS /
    Matcha-TTS), with an O(T_text * T_mel) numba DP instead of a pure-Python double loop.
"""
from __future__ import annotations

import math
from typing import Optional

import numba
import numpy as np
import torch
import torch.nn as nn


class ChannelLayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T)
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class DurationPredictor(nn.Module):
    """Predicts log-durations (log frames). Tempo enters ONLY here (see Step 6 / pace rules)."""

    def __init__(self, in_dim: int, hidden: int = 256, kernel: int = 3, cond_dim: int = 0,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.films = nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(nn.Conv1d(in_dim if i == 0 else hidden, hidden, kernel, padding=kernel // 2))
            self.norms.append(ChannelLayerNorm(hidden))
            self.films.append(nn.Linear(cond_dim, 2 * hidden) if cond_dim else None)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Conv1d(hidden, 1, 1)
        for film in self.films:  # start as identity: gamma = 1, beta = 0
            if film is not None:
                nn.init.zeros_(film.weight); nn.init.zeros_(film.bias)

    def forward(self, h: torch.Tensor, x_mask: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = h.detach()  # duration loss must not shape the encoder (Glow-TTS / Matcha practice)
        for conv, norm, film in zip(self.convs, self.norms, self.films):
            x = norm(torch.relu(conv(x * x_mask)))
            if film is not None and cond is not None:
                g, b = film(cond).unsqueeze(-1).chunk(2, dim=1)
                x = x * (1 + g) + b
            x = self.drop(x)
        return self.out(x * x_mask) * x_mask  # (B, 1, T) log-durations


# ---------------------------------------------------------------------------------------
# Monotonic Alignment Search
# ---------------------------------------------------------------------------------------

def gaussian_log_prior(mu_x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """log N(y_j | mu_x_i, I) for every (text i, frame j): (B, T_x, T_y). Matcha-TTS formulation."""
    n_feats = mu_x.shape[1]
    const = -0.5 * math.log(2 * math.pi) * n_feats
    factor = -0.5 * torch.ones_like(mu_x)
    y_sq = torch.matmul(factor.transpose(1, 2), y ** 2)
    y_mu = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
    mu_sq = torch.sum(factor * mu_x ** 2, 1).unsqueeze(-1)
    return y_sq - y_mu + mu_sq + const


@numba.njit(cache=True)
def _maximum_path_single(value: np.ndarray, t_x: int, t_y: int) -> np.ndarray:
    """Best monotonic path: every frame -> exactly one token, every token >= 1 frame."""
    neg_inf = -1e9
    q = np.full((t_x, t_y), neg_inf, dtype=np.float64)
    for j in range(t_y):
        for i in range(max(0, t_x + j - t_y), min(t_x, j + 1)):
            stay = q[i, j - 1] if j > 0 else (0.0 if i == 0 else neg_inf)
            move = q[i - 1, j - 1] if (i > 0 and j > 0) else neg_inf
            q[i, j] = value[i, j] + max(stay, move)
    path = np.zeros((t_x, t_y), dtype=np.float32)
    i = t_x - 1
    for j in range(t_y - 1, -1, -1):
        path[i, j] = 1.0
        if i > 0 and (i == j or q[i, j - 1] < q[i - 1, j - 1]):
            i -= 1
    return path


def maximum_path(log_prior: torch.Tensor, x_lengths: torch.Tensor, y_lengths: torch.Tensor) -> torch.Tensor:
    """Hard alignment (B, T_x, T_y) from log-likelihoods. Requires T_y >= T_x (see filter_manifest)."""
    lp = log_prior.detach().float().cpu().numpy()
    out = np.zeros(lp.shape, dtype=np.float32)
    for b in range(lp.shape[0]):
        tx, ty = int(x_lengths[b]), int(y_lengths[b])
        if ty < tx:
            raise ValueError(f"item {b}: {ty} frames < {tx} tokens; MAS needs >= 1 frame per token")
        out[b, :tx, :ty] = _maximum_path_single(lp[b, :tx, :ty].astype(np.float64), tx, ty)
    return torch.from_numpy(out).to(log_prior.device)


# ---------------------------------------------------------------------------------------
# Length regulation (inference)
# ---------------------------------------------------------------------------------------

def durations_from_logw(logw: torch.Tensor, x_mask: torch.Tensor, length_scale: float = 1.0,
                        token_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Integer frame counts; every real token gets >= 1 frame AFTER scaling (the v1 fix).
    length_scale: one scalar per language/book (>1 = slower). token_scale: optional (B, T_x)
    per-token multipliers, used by the verify-and-retry loop to slow down suspect words only.
    """
    w = torch.exp(logw) * length_scale
    if token_scale is not None:
        w = w * token_scale.unsqueeze(1)
    return torch.clamp(torch.ceil(w), min=1.0) * x_mask  # (B, 1, T_x)


def path_from_durations(durations: torch.Tensor, x_mask: torch.Tensor, max_frames: Optional[int] = None) -> torch.Tensor:
    """Monotonic hard path (B, T_x, T_y) from integer durations (B, 1, T_x)."""
    d = durations.squeeze(1).long()
    cum = torch.cumsum(d, dim=1)
    t_y = int(max_frames or cum[:, -1].max())
    frames = torch.arange(t_y, device=d.device)[None, None, :]
    end = cum.unsqueeze(-1)
    start = (cum - d).unsqueeze(-1)
    path = ((frames >= start) & (frames < end)).float()
    return path * x_mask.transpose(1, 2)
