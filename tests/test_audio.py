"""Audio tests: vocoder-compatible mels, real loudness normalisation, trimming that keeps speech."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.audio import AudioProcessor, MelConfig

# git clone https://github.com/NVIDIA/BigVGAN third_party/BigVGAN   (or set BIGVGAN_DIR)
BIGVGAN_DIR = Path(os.environ.get("BIGVGAN_DIR", Path(__file__).resolve().parents[1] / "third_party" / "BigVGAN"))


@pytest.fixture(scope="module")
def ap():
    return AudioProcessor()


def speechlike(seconds=1.0, sr=24000, seed=0):
    """Harmonic 'voice' with a slowly varying pitch plus a little noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    f0 = 140 + 20 * np.sin(2 * np.pi * 1.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    x = sum(np.sin(k * phase) / k for k in range(1, 12)) * 0.2
    return torch.from_numpy((x + 0.003 * rng.standard_normal(len(t))).astype(np.float32)).unsqueeze(0)


class TestMel:
    def test_matches_vocoder_config(self, ap):
        cfg = json.loads((BIGVGAN_DIR / "configs" / "bigvgan_v2_24khz_100band_256x.json").read_text())
        assert MelConfig.from_bigvgan_config(cfg) == ap.cfg

    def test_identical_to_bigvgan_mel(self, ap):
        """The acoustic model's targets must be exactly what BigVGAN was trained on."""
        sys.path.insert(0, str(BIGVGAN_DIR))
        from meldataset import mel_spectrogram
        wav = speechlike(1.3)
        c = ap.cfg
        ref = mel_spectrogram(wav, c.n_fft, c.n_mels, c.sample_rate, c.hop_length, c.win_length,
                              c.fmin, c.fmax, center=False).squeeze(0)
        ours = ap.compute_mel(wav)
        assert ours.shape == ref.shape
        assert torch.allclose(ours, ref, atol=1e-5)

    def test_band_count_and_frame_count(self, ap):
        for n in (24000, 24000 + 100, 48000 - 1):
            mel = ap.compute_mel(speechlike(n / 24000))
            assert mel.shape == (100, ap.n_frames(n))  # v1 test expected 90 frames for 1 s: wrong

    def test_silence_hits_the_log_floor(self, ap):
        mel = ap.compute_mel(torch.zeros(1, 24000))
        assert torch.allclose(mel, torch.full_like(mel, np.log(1e-5)), atol=1e-4)


class TestLoudness:
    def test_reaches_target_lufs(self, ap):
        import pyloudnorm as pyln
        quiet = speechlike(3.0) * 0.05
        out, gain = ap.normalize_loudness(quiet)
        measured = pyln.Meter(24000).integrated_loudness(out.squeeze(0).numpy().astype(np.float64))
        assert abs(measured - ap.target_lufs) < 0.1 and gain > 0

    def test_never_clips(self, ap):
        spiky = speechlike(3.0) * 0.001
        spiky[0, 1000] = 0.9  # one transient: loudness gain alone would clip it
        out, _ = ap.normalize_loudness(spiky)
        assert out.abs().max() <= ap.peak_ceiling + 1e-6

    def test_refuses_short_clips(self, ap):
        with pytest.raises(ValueError):
            ap.normalize_loudness(speechlike(0.2))


class TestTrimming:
    def test_pads_are_kept(self, ap):
        wav = torch.zeros(1, 48000)
        wav[:, 12000:36000] = speechlike(1.0)
        out = ap.trim(wav, 0.5, 1.5, pad_s=0.15)
        assert out.shape[-1] == int(1.3 * 24000)

    def test_quiet_onset_is_not_cut(self, ap):
        """v1 cut at -40 dB below the loudest frame and lost a 200 ms soft onset."""
        sr = 24000
        rng = np.random.default_rng(1)
        wav = torch.from_numpy((1e-4 * rng.standard_normal(2 * sr)).astype(np.float32)).unsqueeze(0)  # -80 dBFS floor
        onset = speechlike(0.2) * (10 ** (-60 / 20) / 0.2)       # ~ -60 dBFS soft onset
        loud = speechlike(0.8)                                    # ~ -20 dBFS body
        wav[:, int(0.5 * sr):int(0.7 * sr)] += onset
        wav[:, int(0.7 * sr):int(1.5 * sr)] += loud
        start, end = ap.detect_speech_bounds(wav)
        assert start <= 0.51 and end >= 1.49
        trimmed = ap.trim(wav, start, end)
        assert trimmed.shape[-1] >= int((1.0 + 0.3) * sr) - 1
