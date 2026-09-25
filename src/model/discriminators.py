"""
Multilingual SSL discriminator (replaces the English-only WavLM-Large backbone).

WavLM Base+/Large were pretrained on 94k hours of ENGLISH audio (Libri-Light, GigaSpeech,
English VoxPopuli; the paper: "our focus is English-only audio"). For Czech/Slovak the
default backbone is XLS-R-300M (436k h, 128 languages incl. cs/sk, Apache-2.0), which takes
the same 16 kHz waveform input, so it is a drop-in. Alternatives for the ablation (Step 9):
Omnilingual wav2vec 2.0 (Meta, Nov 2025, 1,600+ languages, Apache-2.0; loads via fairseq2)
and w2v-BERT 2.0 (4.5M h, MIT; needs a DIFFERENTIABLE 80-bin fbank front end).

StyleTTS2 recipe: frozen SSL encoder, trainable head over ALL hidden layers, LSGAN losses,
plus an L1 feature-matching term between real/generated SSL features for the generator.
Gradients must flow: generated waveform -> resample(24k->16k) -> SSL encoder -> head.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF


class SSLDiscriminator(nn.Module):
    def __init__(self, ssl_model: nn.Module, n_layers: int, hidden: int, head_dim: int = 256,
                 in_sr: int = 24000, ssl_sr: int = 16000, normalize_input: bool = True):
        """
        ssl_model: a HF Wav2Vec2Model-compatible module, e.g.
            Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m")  # n_layers=25, hidden=1024
        n_layers: number of hidden states returned (transformer layers + 1)
        """
        super().__init__()
        self.ssl = ssl_model.eval()
        for p in self.ssl.parameters():
            p.requires_grad_(False)
        self.in_sr, self.ssl_sr, self.normalize_input = in_sr, ssl_sr, normalize_input
        self.head = nn.Sequential(
            nn.Conv1d(n_layers * hidden, head_dim, 1), nn.LeakyReLU(0.2),
            nn.Conv1d(head_dim, head_dim, 5, padding=2), nn.LeakyReLU(0.2),
            nn.Conv1d(head_dim, 1, 3, padding=1),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.ssl.eval()  # the frozen encoder stays in eval mode (no dropout, fixed norms)
        return self

    def features(self, wav: torch.Tensor) -> torch.Tensor:
        """wav (B, T) at in_sr -> stacked hidden states (B, n_layers * hidden, frames)."""
        x = AF.resample(wav, self.in_sr, self.ssl_sr) if self.in_sr != self.ssl_sr else wav
        if self.normalize_input:  # XLS-R / wav2vec2-large feature extractors normalise per utterance
            x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-7)
        hs = self.ssl(x, output_hidden_states=True).hidden_states
        return torch.cat(hs, dim=-1).transpose(1, 2)

    def forward(self, wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.features(wav)
        return self.head(feats), feats

    # ---- losses (LSGAN, as in StyleTTS2) ------------------------------------------------
    def discriminator_loss(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        d_real, _ = self(real)
        d_fake, _ = self(fake.detach())
        return torch.mean((1 - d_real) ** 2) + torch.mean(d_fake ** 2)

    def generator_loss(self, real: torch.Tensor, fake: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            _, f_real = self(real)
        d_fake, f_fake = self(fake)
        adv = torch.mean((1 - d_fake) ** 2)
        fm = F.l1_loss(f_fake, f_real)
        return adv, fm
