"""
Audio processing for training and inference.

Fixes relative to v1:
  * mel features are computed EXACTLY like BigVGAN v2 (the vocoder we use): 100 bands,
    Slaney-scale + Slaney-normalised librosa filterbank, (n_fft - hop)/2 reflect padding,
    center=False, magnitude sqrt(re^2 + im^2 + 1e-9), log(clamp(x, 1e-5)).
    v1 used 80 HTK-scale unnormalised bands with center=True: a 3-6 nat per-band offset
    and a different frame grid than the vocoder expects.
  * loudness normalisation is real EBU R128 (pyloudnorm), applied per SOURCE FILE
    (chapter/session) before segmentation, so within-session dynamics survive.
    v1 was peak normalisation to 0.95 per clip and never used target_lufs.
  * trimming never cuts into speech: bounds come from a forced aligner when available,
    otherwise from an energy detector relative to the recording's NOISE FLOOR (not the
    clip's loudest frame), and a fixed pad (default 150 ms) is kept on both sides.
  * I/O uses soundfile: torchaudio.load/save need TorchCodec since torchaudio 2.9.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from librosa.filters import mel as librosa_mel_fn

LOG_MEL_FLOOR = float(np.log(1e-5))  # value of a silent frame; use it for padding


@dataclass
class MelConfig:
    """Must match the vocoder's config.json (nvidia/bigvgan_v2_24khz_100band_256x)."""
    sample_rate: int = 24000
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 100
    fmin: float = 0.0
    fmax: Optional[float] = None  # None -> sample_rate / 2, exactly as BigVGAN's `fmax: null`

    @classmethod
    def from_bigvgan_config(cls, cfg: dict) -> "MelConfig":
        return cls(cfg["sampling_rate"], cfg["n_fft"], cfg["hop_size"], cfg["win_size"],
                   cfg["num_mels"], cfg["fmin"], cfg["fmax"])


class AudioProcessor:
    def __init__(self, mel: MelConfig = MelConfig(), target_lufs: float = -23.0,
                 peak_ceiling: float = 0.99):
        self.cfg = mel
        self.target_lufs = target_lufs
        self.peak_ceiling = peak_ceiling
        fb = librosa_mel_fn(sr=mel.sample_rate, n_fft=mel.n_fft, n_mels=mel.n_mels,
                            fmin=mel.fmin, fmax=mel.fmax)  # librosa defaults: htk=False, norm="slaney"
        self._mel_basis = torch.from_numpy(fb).float()
        self._window = torch.hann_window(mel.win_length)

    # ---- I/O ------------------------------------------------------------------------
    def load_audio(self, path: str) -> torch.Tensor:
        """Mono float32 tensor (1, T) at cfg.sample_rate."""
        data, sr = sf.read(path, dtype="float32", always_2d=True)  # (T, C)
        wav = torch.from_numpy(data.mean(axis=1)).unsqueeze(0)
        if sr != self.cfg.sample_rate:
            wav = AF.resample(wav, sr, self.cfg.sample_rate)
        return wav

    def save_audio(self, path: str, wav: torch.Tensor) -> None:
        sf.write(path, wav.squeeze(0).cpu().numpy(), self.cfg.sample_rate, subtype="PCM_24")

    # ---- loudness (per source file, before segmentation) -------------------------------
    def normalize_loudness(self, wav: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """EBU R128 integrated loudness -> target_lufs; returns (audio, applied_gain_db)."""
        import pyloudnorm as pyln

        x = wav.squeeze(0).numpy().astype(np.float64)
        if len(x) < int(0.4 * self.cfg.sample_rate):
            raise ValueError("loudness needs >= 400 ms of audio; normalise the whole source file")
        loudness = pyln.Meter(self.cfg.sample_rate).integrated_loudness(x)
        gain_db = self.target_lufs - loudness
        y = x * 10 ** (gain_db / 20)
        peak = np.abs(y).max()
        if peak > self.peak_ceiling:  # never clip: lower the gain instead (and log it)
            reduce = self.peak_ceiling / peak
            y *= reduce
            gain_db += 20 * np.log10(reduce)
        return torch.from_numpy(y.astype(np.float32)).unsqueeze(0), float(gain_db)

    # ---- trimming ------------------------------------------------------------------------
    def trim(self, wav: torch.Tensor, speech_start_s: float, speech_end_s: float,
             pad_s: float = 0.15) -> torch.Tensor:
        """Keep [first speech - pad, last speech + pad]. Bounds from a forced aligner are best."""
        sr = self.cfg.sample_rate
        a = max(0, int(round((speech_start_s - pad_s) * sr)))
        b = min(wav.shape[-1], int(round((speech_end_s + pad_s) * sr)))
        return wav[..., a:b]

    def detect_speech_bounds(self, wav: torch.Tensor, noise_floor_db: Optional[float] = None,
                             margin_db: float = 12.0, frame_s: float = 0.010) -> Tuple[float, float]:
        """
        Energy fallback when no alignment exists. The threshold is `margin_db` above the
        recording's noise floor (pass the SESSION floor; default: 5th percentile of this
        clip's frame RMS), not relative to the clip's loudest frame as in v1, so a quiet
        onset in a loud clip is not cut.
        """
        x = wav.squeeze(0).numpy()
        hop = max(1, int(frame_s * self.cfg.sample_rate))
        n = len(x) // hop
        if n == 0:
            return 0.0, len(x) / self.cfg.sample_rate
        frames = x[: n * hop].reshape(n, hop)
        rms_db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-10)
        floor = np.percentile(rms_db, 5) if noise_floor_db is None else noise_floor_db
        speech = np.nonzero(rms_db > floor + margin_db)[0]
        if len(speech) == 0:
            return 0.0, len(x) / self.cfg.sample_rate
        return speech[0] * hop / self.cfg.sample_rate, (speech[-1] + 1) * hop / self.cfg.sample_rate

    # ---- features ------------------------------------------------------------------------
    def compute_mel(self, wav: torch.Tensor) -> torch.Tensor:
        """(1, T) or (T,) waveform -> (n_mels, T // hop) log-mel, identical to BigVGAN v2."""
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        c = self.cfg
        pad = (c.n_fft - c.hop_length) // 2
        y = torch.nn.functional.pad(wav.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
        spec = torch.stft(y, c.n_fft, hop_length=c.hop_length, win_length=c.win_length,
                          window=self._window.to(y.device), center=False, pad_mode="reflect",
                          normalized=False, onesided=True, return_complex=True)
        spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
        mel = torch.matmul(self._mel_basis.to(y.device), spec)
        return torch.log(torch.clamp(mel, min=1e-5)).squeeze(0)

    def n_frames(self, n_samples: int) -> int:
        """Frames produced by compute_mel for n_samples (>= win_length - hop)."""
        return n_samples // self.cfg.hop_length
