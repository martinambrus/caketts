"""
Text encoder (Matcha-TTS style) with the v2 additions.

Changes relative to v1:
  * per-token LANGUAGE embedding (cs/sk/en) added to the phone embedding, so English names
    inside Slovak text are encoded as English (the IndexTTS 2.5 approach)
  * every convolution is MASKED: v1 added positional encodings to padded positions and ran
    unmasked convolutions over them, so a sentence's encoding changed with batch padding
  * outputs the prior mean mu_x in (normalised) mel space for Monotonic Alignment Search and
    the prior loss, as in Matcha-TTS; v1 had an alignment module nothing ever trained
Mask convention everywhere in the model: float tensor (B, 1, T), 1 = valid, 0 = padding.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def sequence_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    max_len = int(max_len or lengths.max())
    return (torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]).float().unsqueeze(1)


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 7, expansion: int = 4):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, dim * expansion)
        self.pw2 = nn.Linear(dim * expansion, dim)
        self.gamma = nn.Parameter(torch.full((dim,), 1e-6))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # x: (B, D, T)
        h = self.dwconv(x * mask).transpose(1, 2)
        h = self.pw2(torch.nn.functional.gelu(self.pw1(self.norm(h))))
        return (x + (self.gamma * h).transpose(1, 2)) * mask


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, ff_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # x: (B, D, T)
        h = x.transpose(1, 2)
        pad = mask.squeeze(1) == 0
        a, _ = self.attn(h, h, h, key_padding_mask=pad, need_weights=False)
        h = self.norm1(h + self.drop(a))
        h = self.norm2(h + self.drop(self.ff(h)))
        return h.transpose(1, 2) * mask


class TextEncoder(nn.Module):
    def __init__(self, n_vocab: int, n_langs: int = 3, n_mels: int = 100, dim: int = 192,
                 n_heads: int = 2, ff_dim: int = 768, n_layers: int = 6, n_convnext: int = 2,
                 cond_dim: int = 0, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.emb = nn.Embedding(n_vocab, dim, padding_idx=0)
        self.lang_emb = nn.Embedding(n_langs, dim)
        nn.init.normal_(self.emb.weight, 0.0, dim ** -0.5)
        nn.init.normal_(self.lang_emb.weight, 0.0, dim ** -0.5)
        self.cond_proj = nn.Linear(cond_dim, dim) if cond_dim else None
        self.prenet = nn.ModuleList([nn.Conv1d(dim, dim, 5, padding=2) for _ in range(3)])
        self.prenet_norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])
        self.blocks = nn.ModuleList([TransformerBlock(dim, n_heads, ff_dim, dropout) for _ in range(n_layers)])
        self.convnext = nn.ModuleList([ConvNeXtBlock(dim) for _ in range(n_convnext)])
        self.proj_mu = nn.Conv1d(dim, n_mels, 1)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def positional(T: int, dim: int, device) -> torch.Tensor:
        pos = torch.arange(T, device=device, dtype=torch.float32)[:, None]
        div = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
        pe = torch.zeros(T, dim, device=device)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        return pe.t().unsqueeze(0)  # (1, D, T)

    def forward(self, phoneme_ids: torch.Tensor, lang_ids: torch.Tensor, lengths: torch.Tensor,
                cond: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns hidden h (B, D, T), prior mean mu_x (B, n_mels, T) and x_mask (B, 1, T)."""
        mask = sequence_mask(lengths, phoneme_ids.shape[1])
        x = (self.emb(phoneme_ids) + self.lang_emb(lang_ids)) * math.sqrt(self.dim)
        x = x.transpose(1, 2)
        if self.cond_proj is not None and cond is not None:
            x = x + self.cond_proj(cond).unsqueeze(-1)
        x = (x + self.positional(x.shape[-1], self.dim, x.device)) * mask
        for conv, norm in zip(self.prenet, self.prenet_norm):
            x = x + self.drop(torch.relu(norm(conv(x * mask).transpose(1, 2)).transpose(1, 2)))
            x = x * mask
        for blk in self.blocks:
            x = blk(x, mask)
        for blk in self.convnext:
            x = blk(x, mask)
        return x, self.proj_mu(x) * mask, mask
