"""ASR runner tests: window planning, word grouping, long-form stitching with a stub model, backends, light import."""
from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from src.data.prepare.asr import ASR_BACKENDS, AsrConfig, AsrRunner, AsrWord, plan_windows, words_from_stamps

SR = 16000


def tone(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * 180 * t) + 0.05 * rng.standard_normal(len(t))).astype(np.float32)


def chunk(token: str, start: float, end: float) -> dict:
    return {"token": token, "start": start, "end": end}


class TestPlanWindows:
    def test_short_audio_is_one_window(self):
        assert plan_windows(tone(20.0), SR, 35.0, 1.0, 3.0) == [(0.0, 20.0, 0.0, 20.0)]

    def test_cuts_follow_the_nominal_grid_without_silence(self):
        plan = plan_windows(tone(100.0), SR, 35.0, 1.0, 3.0)
        cuts = [p[3] for p in plan]
        assert len(plan) == 3
        for cut, nominal in zip(cuts[:-1], [35.0, 70.0]):
            assert nominal - 3.0 <= cut <= nominal

    def test_silence_attracts_the_cut(self):
        audio = tone(100.0)
        audio[int(32.0 * SR):int(32.3 * SR)] = 0.0
        cut = plan_windows(audio, SR, 35.0, 1.0, 3.0)[0][3]
        assert 32.0 <= cut <= 32.3

    def test_windows_cover_and_keeps_partition(self):
        audio = tone(100.0)
        audio[int(67.0 * SR):int(67.3 * SR)] = 0.0
        plan = plan_windows(audio, SR, 35.0, 1.0, 3.0)
        assert plan[0][2] == 0.0 and plan[0][0] == 0.0
        assert plan[-1][3] == pytest.approx(100.0) and plan[-1][1] == pytest.approx(100.0)
        for (_, _, _, prev_end), (_, _, next_start, _) in zip(plan, plan[1:]):
            assert prev_end == next_start
        for win_start, win_end, keep_start, keep_end in plan:
            assert win_start <= keep_start < keep_end <= win_end
            assert win_end - win_start <= 35.0 + 1.0 + 1e-6
        assert plan[1][0] == pytest.approx(plan[0][3] - 0.5)
        assert plan[0][1] == pytest.approx(plan[0][3] + 0.5)

    def test_last_window_is_shorter(self):
        plan = plan_windows(tone(80.0), SR, 35.0, 1.0, 3.0)
        lengths = [p[3] - p[2] for p in plan]
        assert lengths[-1] < lengths[0]


class TestWordsFromStamps:
    def test_groups_chunks_into_words(self):
        stamps = [chunk("D", 0.96, 1.12), chunk("obr", 1.12, 1.36), chunk("ý", 1.36, 1.52), chunk(" den", 1.68, 1.92),
                  chunk(" ahoj", 2.08, 2.32), chunk("te", 2.32, 2.56)]
        assert words_from_stamps(stamps) == [("Dobrý", 0.96, 1.52), ("den", 1.68, 1.92), ("ahojte", 2.08, 2.56)]

    def test_punctuation_joins_the_word(self):
        stamps = [chunk(" ahoj", 0.0, 0.4), chunk(",", 0.4, 0.4), chunk(" svět", 0.5, 0.9), chunk(".", 0.9, 0.9)]
        assert words_from_stamps(stamps) == [("ahoj,", 0.0, 0.4), ("svět.", 0.5, 0.9)]

    def test_lone_space_before_digits(self):
        stamps = [chunk(" rok", 0.0, 0.3), chunk(" ", 0.4, 0.5), chunk("7", 0.5, 0.6), chunk("4", 0.6, 0.8)]
        assert words_from_stamps(stamps) == [("rok", 0.0, 0.3), ("74", 0.5, 0.8)]

    def test_empty(self):
        assert words_from_stamps([]) == []
        assert words_from_stamps([chunk(" ", 0.0, 0.1)]) == []


class StubProcessor:
    def __init__(self, stamps_by_window):
        self.stamps_by_window = stamps_by_window
        self.calls = 0

    def __call__(self, clips, sampling_rate, return_tensors):
        import torch
        self.clip_lengths = [len(c) for c in clips]
        return {"input_features": torch.zeros(len(clips), 10, 128), "attention_mask": torch.ones(len(clips), 10, dtype=torch.bool)}

    def decode(self, sequences, durations=None, skip_special_tokens=True):
        n = sequences.shape[0]
        stamps = [self.stamps_by_window[self.calls + i] for i in range(n)]
        self.calls += n
        return [""] * n, stamps

    def batch_decode(self, sequences, skip_special_tokens=True):
        return [f"text{self.clip_lengths[i]}" for i in range(sequences.shape[0])]


class StubModel:
    device = None
    dtype = None

    def __init__(self):
        import torch
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    def generate(self, input_features, attention_mask):
        import torch
        b = input_features.shape[0]
        return SimpleNamespace(sequences=torch.zeros(b, 3, dtype=torch.long), durations=torch.ones(b, 3, dtype=torch.long))


class TestTranscribeWordsStub:
    def runner(self, stamps_by_window):
        pytest.importorskip("torch")
        cfg = AsrConfig(batch_size=2)
        return AsrRunner(cfg, model=StubModel(), processor=StubProcessor(stamps_by_window))

    def test_offsets_overlap_and_order(self):
        audio = tone(60.0)
        plan = plan_windows(audio, SR, 35.0, 1.0, 3.0)
        assert len(plan) == 2
        cut = plan[0][3]
        w1_start, w2_start = plan[0][0], plan[1][0]
        stamps = [
            [chunk(" prvni", 1.0, 1.5), chunk(" okraj", cut - 0.4 - w1_start, cut - 0.2 - w1_start)],
            [chunk(" okraj", cut - 0.4 - w2_start, cut - 0.2 - w2_start), chunk(" treti", cut + 5 - w2_start, cut + 5.4 - w2_start)],
        ]
        words = self.runner(stamps).transcribe_words(audio)
        assert [w.text for w in words] == ["prvni", "okraj", "treti"]
        assert all(isinstance(w, AsrWord) for w in words)
        assert words[0].start == pytest.approx(1.0)
        assert words[2].start == pytest.approx(cut + 5)
        assert [w.start for w in words] == sorted(w.start for w in words)

    def test_word_in_overlap_is_output_once(self):
        audio = tone(60.0)
        plan = plan_windows(audio, SR, 35.0, 1.0, 3.0)
        cut = plan[0][3]
        w1_start, w2_start = plan[0][0], plan[1][0]
        stamps = [
            [chunk(" dvakrat", cut - 0.2 - w1_start, cut + 0.2 - w1_start)],
            [chunk(" dvakrat", cut - 0.2 - w2_start, cut + 0.2 - w2_start)],
        ]
        assert len(self.runner(stamps).transcribe_words(audio)) == 1

    def test_clip_longer_than_window_is_rejected(self):
        runner = self.runner([])
        with pytest.raises(ValueError):
            runner.transcribe([tone(36.0)])

    def test_transcribe_keeps_order_and_empty_clip(self):
        runner = self.runner([])
        texts = runner.transcribe([tone(1.0), np.zeros(0, dtype=np.float32), tone(2.0)])
        assert texts == [f"text{SR}", "", f"text{2 * SR}"]


class TestBackends:
    def test_unknown_backend(self):
        with pytest.raises(ValueError, match="parakeet-tdt-0.6b-v3"):
            AsrRunner(AsrConfig(backend="nope"))

    def test_canary_has_no_words(self):
        with pytest.raises(ValueError):
            AsrRunner(AsrConfig(backend="canary-1b-v2")).transcribe_words(tone(1.0))

    def test_registry_ids(self):
        assert ASR_BACKENDS["parakeet-tdt-0.6b-v3"].model_id == "nvidia/parakeet-tdt-0.6b-v3"
        assert ASR_BACKENDS["canary-1b-v2"].model_id == "nvidia/canary-1b-v2"
        assert AsrRunner(AsrConfig(backend="canary-1b-v2")).model_id == "nvidia/canary-1b-v2"

    def test_import_is_light(self):
        code = ("import sys, src.data.prepare.asr; "
                "bad = [m for m in ('torch', 'transformers', 'uroman') if m in sys.modules]; "
                "sys.exit(1 if bad else 0)")
        env = {**os.environ, "PYTHONPATH": os.getcwd()}
        assert subprocess.run([sys.executable, "-c", code], env=env).returncode == 0


def cached(repo_id: str) -> bool:
    from huggingface_hub import try_to_load_from_cache
    return isinstance(try_to_load_from_cache(repo_id, "config.json"), str)


@pytest.mark.skipif(not cached("nvidia/parakeet-tdt-0.6b-v3"), reason="parakeet not in the HF cache")
class TestRealParakeet:
    def test_transcribe_and_words(self):
        runner = AsrRunner()
        try:
            texts = runner.transcribe([tone(3.0), tone(2.0, seed=1)])
            assert len(texts) == 2 and all(isinstance(t, str) for t in texts)
            assert isinstance(runner.transcribe_words(tone(40.0)), list)
        finally:
            runner.close()
