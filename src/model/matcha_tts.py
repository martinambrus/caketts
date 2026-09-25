"""
MatchaTTS assembly: encoder -> (MAS | duration predictor) -> CFM decoder, plus the three
Matcha-TTS losses. v1 referenced `MatchaTTS` and `compute_total_loss` in its tests but never
specified them, and nothing trained its alignment.

Conditioning (v2):
  * narrator: a learned speaker table (one row per pretraining speaker + the narrator);
    replaces per-utterance ECAPA vectors, which carry session/language information
  * tempo: utterance articulation rate + pause ratio -> DURATION PREDICTOR ONLY, so tempo is
    an explicit input that is fixed at inference (the ElevenLabs slow-mode problem)
  * boundary: mid-paragraph / paragraph-final / chapter-final sentence flag -> durations and
    decoder, so paragraph cadence comes from the flag, never from "end of input"
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from .duration import DurationPredictor, durations_from_logw, gaussian_log_prior, maximum_path, path_from_durations
from .flow_matching import CFMDecoder
from .text_encoder import TextEncoder, sequence_mask


@dataclass
class MatchaConfig:
    n_vocab: int
    n_langs: int = 3
    n_speakers: int = 1
    n_boundaries: int = 3
    n_mels: int = 100
    enc_dim: int = 192
    enc_layers: int = 6
    enc_heads: int = 2
    enc_ff: int = 768
    spk_dim: int = 64
    cond_dim: int = 64
    dur_hidden: int = 256
    dec_dim: int = 256
    dec_levels: int = 2
    sigma_min: float = 1e-4
    p_uncond: float = 0.0
    dropout: float = 0.1
    rate_mean: float = 12.0   # phones/s, set from the corpus (Step 4.4)
    rate_std: float = 2.0


class MatchaTTS(nn.Module):
    def __init__(self, cfg: MatchaConfig):
        super().__init__()
        self.cfg = cfg
        self.spk = nn.Embedding(cfg.n_speakers, cfg.spk_dim)
        self.boundary = nn.Embedding(cfg.n_boundaries, cfg.cond_dim)
        self.rate_mlp = nn.Sequential(nn.Linear(2, cfg.cond_dim), nn.SiLU(), nn.Linear(cfg.cond_dim, cfg.cond_dim))
        self.encoder = TextEncoder(cfg.n_vocab, cfg.n_langs, cfg.n_mels, cfg.enc_dim, cfg.enc_heads, cfg.enc_ff,
                                   cfg.enc_layers, cond_dim=cfg.spk_dim, dropout=cfg.dropout)
        self.duration = DurationPredictor(cfg.enc_dim, cfg.dur_hidden, cond_dim=cfg.spk_dim + 2 * cfg.cond_dim,
                                          dropout=cfg.dropout)
        self.decoder = CFMDecoder(cfg.n_mels, cond_dim=cfg.spk_dim + cfg.cond_dim, dim=cfg.dec_dim,
                                  levels=cfg.dec_levels, sigma_min=cfg.sigma_min, p_uncond=cfg.p_uncond)

    # ---- conditioning --------------------------------------------------------------------
    def _conds(self, speaker_ids, rates, boundary_ids):
        spk = self.spk(speaker_ids)
        rate_n = torch.stack([(rates[:, 0] - self.cfg.rate_mean) / self.cfg.rate_std, rates[:, 1]], dim=1)
        dur_cond = torch.cat([spk, self.rate_mlp(rate_n), self.boundary(boundary_ids)], dim=1)
        dec_cond = torch.cat([spk, self.boundary(boundary_ids)], dim=1)
        return spk, dur_cond, dec_cond

    # ---- training ------------------------------------------------------------------------
    def forward(self, batch: Dict[str, torch.Tensor], negatives_weight: float = 0.0,
                negatives: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        x, xl, y, yl = batch["phoneme_ids"], batch["phoneme_lengths"], batch["mels"], batch["mel_lengths"]
        spk, dur_cond, dec_cond = self._conds(batch["speaker_ids"], batch["rates"], batch["boundary_ids"])
        h, mu_x, x_mask = self.encoder(x, batch["lang_ids"], xl, spk)
        y_mask = sequence_mask(yl, y.shape[-1])

        with torch.no_grad():
            attn = maximum_path(gaussian_log_prior(mu_x, y), xl, yl)  # (B, T_x, T_y)

        logw = self.duration(h, x_mask, dur_cond)
        logw_target = torch.log(1e-8 + attn.sum(-1)).unsqueeze(1) * x_mask
        dur_loss = torch.sum((logw - logw_target) ** 2) / torch.sum(xl)

        mu_y = torch.matmul(attn.transpose(1, 2), mu_x.transpose(1, 2)).transpose(1, 2)  # (B, n_mels, T_y)
        prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask) / (
            torch.sum(y_mask) * self.cfg.n_mels)

        cfm_loss = self.decoder.loss(y, y_mask, mu_y, dec_cond, negatives=negatives, neg_weight=negatives_weight)
        return {"dur_loss": dur_loss, "prior_loss": prior_loss, "cfm_loss": cfm_loss,
                "loss": dur_loss + prior_loss + cfm_loss, "attn": attn}

    # ---- inference -----------------------------------------------------------------------
    @torch.no_grad()
    def synthesize(self, phoneme_ids, lang_ids, lengths, speaker_ids, rates, boundary_ids,
                   n_steps: int = 10, temperature: float = 0.667, length_scale: float = 1.0,
                   guidance: float = 0.0, token_scale: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        `rates` is the FIXED target tempo for every sentence of a book (narrator median),
        not something inferred per sentence. `length_scale` is one scalar per language/book.
        """
        spk, dur_cond, dec_cond = self._conds(speaker_ids, rates, boundary_ids)
        h, mu_x, x_mask = self.encoder(phoneme_ids, lang_ids, lengths, spk)
        logw = self.duration(h, x_mask, dur_cond)
        durations = durations_from_logw(logw, x_mask, length_scale, token_scale)  # >= 1 frame per token
        y_lengths = durations.sum(dim=(1, 2)).long()
        path = path_from_durations(durations, x_mask)                       # (B, T_x, T_y)
        y_mask = sequence_mask(y_lengths, path.shape[-1])
        mu_y = torch.matmul(path.transpose(1, 2), mu_x.transpose(1, 2)).transpose(1, 2)
        mel = self.decoder.sample(mu_y, y_mask, n_steps, temperature, dec_cond, guidance)
        return {"mel": mel, "mel_lengths": y_lengths, "durations": durations.squeeze(1), "path": path}
