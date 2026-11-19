"""Corpus-preparation audio stage: streamed mono mix, -23 LUFS over the whole part, 24 kHz and 16 kHz PCM_16 wavs."""
import json
import math
import resource
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import torch
import torchaudio.functional as AF
from scipy.signal import lfilter


@dataclass
class AudioConfig:
    target_lufs: float = -23.0
    peak_ceiling: float = 0.99
    clip_sr: int = 24000
    align_sr: int = 16000
    block_s: float = 30.0
    clip_level: float = 0.9999


def iter_resampled(path, rates, block_s):
    """Mono float32 blocks, one array per rate, equal to a one-shot resample of the whole file."""
    with sf.SoundFile(str(path)) as f:
        sr, n = f.samplerate, f.frames
        period = math.lcm(*[sr // math.gcd(sr, r) for r in rates])
        block = max(1, round(block_s * sr / period)) * period
        ctx = math.ceil(0.1 * sr / period) * period
        for s in range(0, n, block):
            lo, hi = max(0, s - ctx), min(n, s + block + ctx)
            f.seek(lo)
            mono = f.read(hi - lo, dtype="float32", always_2d=True).mean(axis=1)
            end = min(n, s + block)
            out = []
            for r in rates:
                y = mono if r == sr else AF.resample(torch.from_numpy(mono).unsqueeze(0), sr, r)[0].numpy()
                a = (s - lo) * r // sr if r != sr else s - lo
                out.append(y[a:a + math.ceil(r * (end - s) / sr)])
            yield tuple(out)


class LoudnessMeter:
    """Streaming equivalent of pyloudnorm's integrated loudness (whole 400 ms blocks, hop 100 ms)."""

    def __init__(self, rate):
        if rate % 10:
            raise ValueError("rate must be a multiple of 10")
        self.hop = rate // 10
        self._stages = [(s.b, s.a, s.passband_gain, np.zeros(max(len(s.a), len(s.b)) - 1))
                        for s in pyln.Meter(rate)._filters.values()]
        self._pending = np.zeros(0)
        self._hops = []

    def update(self, x):
        y = np.asarray(x, dtype=np.float64)
        for i, (b, a, gain, zi) in enumerate(self._stages):
            y, zi = lfilter(b, a, y, zi=zi)
            y = gain * y
            self._stages[i] = (b, a, gain, zi)
        sq = np.concatenate([self._pending, y ** 2])
        whole = len(sq) // self.hop * self.hop
        self._hops.append(sq[:whole].reshape(-1, self.hop).sum(axis=1))
        self._pending = sq[whole:]

    def integrated(self):
        hops = np.concatenate(self._hops) if self._hops else np.zeros(0)
        if len(hops) < 4:
            raise ValueError("loudness needs >= 400 ms of audio")
        z = np.convolve(hops, np.ones(4), mode="valid") / (4 * self.hop)
        with np.errstate(divide="ignore"):
            loud = -0.691 + 10 * np.log10(z)
        z = z[loud >= -70]
        if not len(z):
            raise ValueError("no audio above the absolute loudness gate (silent part)")
        threshold = -0.691 + 10 * np.log10(z.mean()) - 10
        with np.errstate(divide="ignore"):
            loud = -0.691 + 10 * np.log10(z)
        kept = z[(loud > threshold) & (loud > -70)]
        return float(-0.691 + 10 * np.log10(kept.mean()))


def count_clipped_runs(path, block_s, level):
    """Runs of >= 3 consecutive samples with |x| >= level on one channel of the source."""
    total = 0
    with sf.SoundFile(str(path)) as f:
        carry = np.zeros(f.channels, dtype=np.int64)
        step = max(1, round(block_s * f.samplerate))
        while True:
            data = f.read(step, dtype="float32", always_2d=True)
            if not len(data):
                break
            hit = np.abs(data) >= level
            for c in range(f.channels):
                m = np.concatenate([np.ones(carry[c], dtype=bool), hit[:, c], [False]])
                d = np.diff(np.concatenate([[False], m]).astype(np.int8))
                lengths = np.flatnonzero(d == -1) - np.flatnonzero(d == 1)
                if hit[-1, c]:
                    carry[c], lengths = lengths[-1], lengths[:-1]
                else:
                    carry[c] = 0
                total += int((lengths >= 3).sum())
        total += int((carry >= 3).sum())
    return total


def _measure_file(path, block_s):
    meter, peak = LoudnessMeter(24000), 0.0
    with sf.SoundFile(str(path)) as f:
        while True:
            x = f.read(round(block_s * f.samplerate), dtype="float32")
            if not len(x):
                break
            meter.update(x)
            peak = max(peak, float(np.abs(x).max()))
    return meter.integrated(), peak


def process_part(src, out_dir, stem, cfg):
    src, out_dir = Path(src), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = sf.info(str(src))
    meter, peak, n_clip, n_align = LoudnessMeter(cfg.clip_sr), 0.0, 0, 0
    for b_clip, b_align in iter_resampled(src, (cfg.clip_sr, cfg.align_sr), cfg.block_s):
        meter.update(b_clip)
        peak = max(peak, float(np.abs(b_clip).max()), float(np.abs(b_align).max()))
        n_clip, n_align = n_clip + len(b_clip), n_align + len(b_align)
    loudness_in = meter.integrated()
    if peak == 0:
        raise ValueError(f"{src} is silent")
    gain_db, limited = cfg.target_lufs - loudness_in, False
    if peak * 10 ** (gain_db / 20) > cfg.peak_ceiling:
        gain_db, limited = 20 * math.log10(cfg.peak_ceiling / peak), True
    gain = np.float32(10 ** (gain_db / 20))
    paths = {cfg.clip_sr: out_dir / f"{stem}.24k.wav", cfg.align_sr: out_dir / f"{stem}.16k.wav"}
    kw = dict(channels=1, subtype="PCM_16", format="WAV")
    with sf.SoundFile(str(paths[cfg.clip_sr]), "w", samplerate=cfg.clip_sr, **kw) as w_clip, \
            sf.SoundFile(str(paths[cfg.align_sr]), "w", samplerate=cfg.align_sr, **kw) as w_align:
        for b_clip, b_align in iter_resampled(src, (cfg.clip_sr, cfg.align_sr), cfg.block_s):
            w_clip.write(b_clip * gain)
            w_align.write(b_align * gain)
    loudness_out, peak_out = _measure_file(paths[cfg.clip_sr], cfg.block_s)
    info = {"sr_in": meta.samplerate, "channels_in": meta.channels, "duration_s": meta.frames / meta.samplerate,
            "loudness_in": loudness_in, "loudness_out": loudness_out, "gain_db": gain_db, "peak_out": peak_out,
            "peak_in": peak, "peak_limited": limited,
            "clipped_runs_in": count_clipped_runs(src, cfg.block_s, cfg.clip_level),
            "samples_in": meta.frames, "samples_clip": n_clip, "samples_align": n_align}
    (out_dir / f"{stem}.json").write_text(json.dumps(info, sort_keys=True, indent=2))
    return info


def run_audio(raw_dir, work_dir, parts, cfg=AudioConfig()):
    raw_dir, out_dir = Path(raw_dir), Path(work_dir) / "audio"
    report = {"parts": {}}
    for stem in parts:
        src = raw_dir / f"{stem}.wav"
        if not src.is_file():
            raise FileNotFoundError(f"raw part not found: {src}")
        t0 = time.perf_counter()
        info = process_part(src, out_dir, stem, cfg)
        info["seconds"] = time.perf_counter() - t0
        info["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        report["parts"][stem] = info
    (out_dir / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2))
    rows = ["| part | minutes | loudness in | loudness out | gain dB | peak | clipped runs | seconds | RSS MB |",
            "|---|---|---|---|---|---|---|---|---|"]
    for stem, i in report["parts"].items():
        rows.append(f"| {stem} | {i['duration_s'] / 60:.1f} | {i['loudness_in']:.2f} | {i['loudness_out']:.2f} | "
                    f"{i['gain_db']:.2f} | {i['peak_out']:.3f} | {i['clipped_runs_in']} | {i['seconds']:.1f} | "
                    f"{i['peak_rss_mb']:.0f} |")
    (out_dir / "report.md").write_text("\n".join(rows) + "\n")
    return report
