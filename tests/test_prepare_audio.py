"""Audio stage tests: streaming loudness, block resampling, mono mix, peak ceiling, clipped runs, report. Synthetic audio only."""
import json
import math

import numpy as np
import pytest
import pyloudnorm as pyln
import soundfile as sf
import torch
import torchaudio.functional as AF

from src.data.audio import AudioProcessor, MelConfig
from src.data.prepare.audio import (AudioConfig, LoudnessMeter, count_clipped_runs, iter_resampled,
                                    process_part, run_audio)

CONTRACT_KEYS = {"sr_in", "channels_in", "duration_s", "loudness_in", "loudness_out", "gain_db", "peak_out"}
EXTRA_KEYS = {"peak_in", "peak_limited", "clipped_runs_in", "samples_in", "samples_clip", "samples_align"}


def speechlike(seconds=1.0, sr=24000, seed=0):
    """Harmonic 'voice' with a slowly varying pitch plus a little noise, float64 numpy."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(round(seconds * sr))) / sr
    f0 = 140 + 20 * np.sin(2 * np.pi * 1.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    x = sum(np.sin(k * phase) / k for k in range(1, 12)) * 0.2
    return x + 0.003 * rng.standard_normal(len(t))


def at_rms_db(x, db):
    return x * 10 ** (db / 20) / np.sqrt(np.mean(x ** 2))


def write_stereo(path, left, right, sr=44100, subtype="PCM_16"):
    sf.write(str(path), np.stack([left, right], axis=1).astype(np.float32), sr, subtype=subtype)


def voice_part(path, db, seconds=4.0, sr=44100):
    left = at_rms_db(speechlike(seconds, sr, seed=1), db)
    right = 0.9 * left + at_rms_db(speechlike(seconds, sr, seed=2), db - 20)
    write_stereo(path, left, right, sr)


def feed(meter, x, sizes):
    i, k = 0, 0
    while i < len(x):
        n = sizes[k % len(sizes)]
        meter.update(x[i:i + n])
        i, k = i + n, k + 1
    return meter.integrated()


def mixed_signal():
    loud = speechlike(3, seed=3) * 1.5
    quiet = speechlike(3, seed=4) * 0.002
    return np.concatenate([loud, quiet, np.zeros(2 * 24000), loud[:24000]])


class TestLoudnessMeter:
    @pytest.mark.parametrize("name", ["quiet", "loud", "mixed", "ragged"])
    def test_matches_pyloudnorm(self, name):
        x = {"quiet": speechlike(9, seed=5) * 0.05, "loud": speechlike(9, seed=6) * 1.2,
             "mixed": mixed_signal(), "ragged": np.append(speechlike(9, seed=7), np.full(1000, 0.1))}[name]
        ref = pyln.Meter(24000).integrated_loudness(x)
        for sizes in ([1000], [7777], [100000], [len(x)], [1, 5000, 23999]):
            assert feed(LoudnessMeter(24000), x, sizes) == pytest.approx(ref, abs=0.005)

    def test_gates_act_on_mixed_signal(self):
        x = mixed_signal()
        got = feed(LoudnessMeter(24000), x, [7777])
        assert got > pyln.Meter(24000).integrated_loudness(x[:3 * 24000]) - 0.5
        ungated = -0.691 + 10 * np.log10(np.mean(x ** 2))
        assert abs(got - ungated) > 2

    def test_too_short_and_silent_and_bad_rate(self):
        m = LoudnessMeter(24000)
        m.update(speechlike(0.37))
        with pytest.raises(ValueError):
            m.integrated()
        m = LoudnessMeter(24000)
        m.update(np.zeros(48000))
        with pytest.raises(ValueError):
            m.integrated()
        with pytest.raises(ValueError):
            LoudnessMeter(24005)


class TestIterResampled:
    @pytest.mark.parametrize("sr", [44100, 48000, 16000])
    def test_blocks_equal_one_shot(self, tmp_path, sr):
        n = 100003 if sr == 44100 else 70001
        left, right = speechlike(n / sr, sr, 1), speechlike(n / sr, sr, 2)
        left, right = left[:n], right[:n]
        p = tmp_path / "a.wav"
        write_stereo(p, left, right, sr)
        data, _ = sf.read(str(p), dtype="float32", always_2d=True)
        mono = torch.from_numpy(data.mean(axis=1)).unsqueeze(0)
        blocks = list(iter_resampled(p, (24000, 16000), 0.5))
        assert len(blocks) > 1
        for k, rate in enumerate((24000, 16000)):
            got = np.concatenate([b[k] for b in blocks])
            ref = AF.resample(mono, sr, rate)[0].numpy() if rate != sr else mono[0].numpy()
            assert len(got) == math.ceil(rate * n / sr) == len(ref)
            assert np.abs(got - ref).max() < 1e-5
        if sr == 44100:
            ref24 = AudioProcessor().load_audio(str(p))[0].numpy()
            assert np.abs(np.concatenate([b[0] for b in blocks]) - ref24).max() < 1e-5


class TestProcessPart:
    @pytest.mark.parametrize("db", [-45, -30, -12])
    def test_levels(self, tmp_path, db):
        src = tmp_path / "raw.wav"
        voice_part(src, db)
        info = process_part(src, tmp_path / "out", "p", AudioConfig(block_s=1.0))
        assert CONTRACT_KEYS | EXTRA_KEYS <= set(info) == set(json.loads((tmp_path / "out/p.json").read_text()))
        for rate, key in ((24000, "samples_clip"), (16000, "samples_align")):
            f = sf.SoundFile(str(tmp_path / f"out/p.{rate // 1000}k.wav"))
            assert (f.channels, f.subtype, f.samplerate) == (1, "PCM_16", rate)
            assert abs(f.frames / rate - info["samples_in"] / 44100) <= 1 / rate
            assert f.frames == info[key]
        out, _ = sf.read(str(tmp_path / "out/p.24k.wav"), dtype="float64")
        measured = pyln.Meter(24000).integrated_loudness(out)
        assert measured == pytest.approx(-23, abs=0.1)
        assert info["loudness_out"] == pytest.approx(measured, abs=0.02)
        assert info["peak_out"] <= 0.99 and not info["peak_limited"]

    def test_mono_is_channel_mean(self, tmp_path):
        t = np.arange(88200) / 44100
        write_stereo(tmp_path / "a.wav", 0.3 * np.sin(2 * np.pi * 300 * t), 0.3 * np.sin(2 * np.pi * 700 * t))
        process_part(tmp_path / "a.wav", tmp_path / "out", "p", AudioConfig())
        out, _ = sf.read(str(tmp_path / "out/p.24k.wav"))
        spec = np.abs(np.fft.rfft(out)) / len(out) * 2
        a300, a700 = spec[600], spec[1400]
        assert abs(20 * np.log10(a300 / a700)) < 0.5
        assert spec[1000] < 0.01 * a300
        write_stereo(tmp_path / "b.wav", 0.3 * np.sin(2 * np.pi * 300 * t), -0.3 * np.sin(2 * np.pi * 300 * t))
        with pytest.raises(ValueError):
            process_part(tmp_path / "b.wav", tmp_path / "out", "q", AudioConfig())

    def test_short_part_fails(self, tmp_path):
        write_stereo(tmp_path / "a.wav", speechlike(0.3, 44100), speechlike(0.3, 44100))
        with pytest.raises(ValueError):
            process_part(tmp_path / "a.wav", tmp_path / "out", "p", AudioConfig())

    def test_peak_ceiling(self, tmp_path):
        n = 4 * 44100
        rng = np.random.default_rng(0)
        x = 0.01 * rng.standard_normal(n)
        x[n // 2 - 441:n // 2 + 441] += 0.9 * np.hanning(882)
        write_stereo(tmp_path / "a.wav", x, x)
        info = process_part(tmp_path / "a.wav", tmp_path / "out", "p", AudioConfig())
        assert info["peak_limited"] and info["peak_out"] <= 0.99
        ap = AudioProcessor()
        _, ref_gain = ap.normalize_loudness(ap.load_audio(str(tmp_path / "a.wav")))
        assert info["gain_db"] == pytest.approx(ref_gain, abs=0.01)

    def test_clipped_runs_in_json(self, tmp_path):
        x = 0.01 * np.random.default_rng(1).standard_normal(4 * 44100)
        x[22049:22052] = 1.0
        write_stereo(tmp_path / "a.wav", x, 0 * x, subtype="FLOAT")
        info = process_part(tmp_path / "a.wav", tmp_path / "out", "p", AudioConfig(block_s=0.5))
        assert info["clipped_runs_in"] == 1

    def test_deterministic_and_block_size_independent(self, tmp_path):
        voice_part(tmp_path / "raw.wav", -30, seconds=3.0)
        for d, b in (("a", 0.5), ("b", 0.5), ("c", 30.0)):
            process_part(tmp_path / "raw.wav", tmp_path / d, "p", AudioConfig(block_s=b))
        for ext in ("24k.wav", "16k.wav", "json"):
            assert (tmp_path / "a" / f"p.{ext}").read_bytes() == (tmp_path / "b" / f"p.{ext}").read_bytes()
        for ext in ("24k.wav", "16k.wav"):
            a, _ = sf.read(str(tmp_path / "a" / f"p.{ext}"), dtype="int16")
            c, _ = sf.read(str(tmp_path / "c" / f"p.{ext}"), dtype="int16")
            assert len(a) == len(c) and np.abs(a.astype(int) - c).max() <= 1


class TestClippedRuns:
    @pytest.mark.parametrize("start,length,expected", [
        (1000, 3, 1), (1000, 2, 0), (22049, 3, 1), (22048, 3, 1), (22047, 5, 1), (22050, 3, 1), (22049, 2, 0)])
    def test_counts(self, tmp_path, start, length, expected):
        x = np.zeros(44100)
        x[start:start + length] = 1.0
        write_stereo(tmp_path / "a.wav", x, 0 * x, subtype="FLOAT")
        assert count_clipped_runs(tmp_path / "a.wav", 0.5, 0.9999) == expected

    def test_two_runs_and_other_channel(self, tmp_path):
        left, right = np.zeros(44100), np.zeros(44100)
        left[100:104] = -1.0
        right[40000:40003] = 1.0
        write_stereo(tmp_path / "a.wav", left, right, subtype="FLOAT")
        assert count_clipped_runs(tmp_path / "a.wav", 0.5, 0.9999) == 2


class TestRunAudio:
    def test_defaults_match_audio_processor(self):
        ap, cfg = AudioProcessor(), AudioConfig()
        assert (cfg.target_lufs, cfg.peak_ceiling, cfg.clip_sr) == (ap.target_lufs, ap.peak_ceiling, MelConfig().sample_rate)

    def test_report(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        for s in ("p1", "p2"):
            voice_part(raw / f"{s}.wav", -30, seconds=2.0)
        rep = run_audio(raw, tmp_path / "work", ["p1", "p2"])
        on_disk = json.loads((tmp_path / "work/audio/report.json").read_text())
        assert set(on_disk["parts"]) == {"p1", "p2"}
        for info in rep["parts"].values():
            assert CONTRACT_KEYS | EXTRA_KEYS | {"seconds", "peak_rss_mb"} <= set(info)
        md = (tmp_path / "work/audio/report.md").read_text()
        assert "p1" in md and "p2" in md
        for s in ("p1", "p2"):
            for ext in ("24k.wav", "16k.wav", "json"):
                assert (tmp_path / f"work/audio/{s}.{ext}").is_file()
        run_audio(raw, tmp_path / "work", ["p1"])

    def test_missing_part(self, tmp_path):
        (tmp_path / "raw").mkdir()
        with pytest.raises(FileNotFoundError, match="nope"):
            run_audio(tmp_path / "raw", tmp_path / "work", ["nope"])
