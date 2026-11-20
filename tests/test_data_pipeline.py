"""Data pipeline: drop-don't-crop filtering, strict symbols, conditioning carried into batches."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from src.data.audio import AudioProcessor, MelConfig
from src.data.dataset import TTSCollator, TTSDataset, corpus_mel_stats, filter_manifest, load_manifest
from src.text.phonemizer import CzechSlovakPhonemizer, UnknownSymbolError, build_vocabulary

SR = 24000
TEXTS = ["Dobrý den, jak se máte?", "Rozhlas vysílá zprávy.", "Kde je ten muž?",
         "Tohle je velmi dlouhá věta, která se nevejde do limitu.", "Ano."]


@pytest.fixture()
def corpus(tmp_path):
    rows = []
    durations = [1.8, 1.4, 1.2, 25.0, 0.3]  # #4 too long, #5 too short
    for i, (text, dur) in enumerate(zip(TEXTS, durations)):
        path = tmp_path / f"u{i}.wav"
        t = np.arange(int(dur * SR)) / SR
        sf.write(path, (0.1 * np.sin(2 * np.pi * 150 * t)).astype(np.float32), SR)
        rows.append({"id": f"u{i}", "audio": str(path), "text": text, "lang": "cs",
                     "speaker": "narrator", "boundary": "para_end" if i == 1 else "mid",
                     "duration_s": dur, "articulation_rate": 12.5, "pause_ratio": 0.1})
    manifest = tmp_path / "train.jsonl"
    manifest.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return manifest


@pytest.fixture()
def g2p():
    return {"cs": CzechSlovakPhonemizer("cs")}


def token_counts(rows, g2p):
    return {r["id"]: len(g2p[r["lang"]].tokenize(r["text"])) for r in rows}


def test_filter_drops_instead_of_cropping(corpus, g2p):
    rows = load_manifest(str(corpus))
    kept, report = filter_manifest(rows, token_counts(rows, g2p), hop_s=256 / SR, max_duration_s=20.0)
    assert [r["id"] for r in kept] == ["u0", "u1", "u2"]
    assert report.too_long == ["u3"] and report.too_short == ["u4"]


def test_filter_drops_impossible_alignments(g2p):
    rows = [{"id": "x", "text": "Rozhlas vysílá zprávy.", "lang": "cs", "duration_s": 0.6}]
    n = token_counts(rows, g2p)
    kept, report = filter_manifest(rows, n, hop_s=0.1, min_duration_s=0.1)  # 6 frames < ~20 tokens
    assert kept == [] and report.impossible_alignment == ["x"]


@pytest.mark.parametrize("sr,n", [(24000, 10240), (22050, 3840), (22050, 7680), (16000, 11008)])
def test_filter_counts_the_frames_the_mel_has(sr, n):
    frames = AudioProcessor(MelConfig(sample_rate=sr)).compute_mel(torch.zeros(1, n)).shape[1]
    rows = [{"id": "x", "duration_s": n / sr}]
    kept, _ = filter_manifest(rows, {"x": frames}, hop_s=256 / sr, min_duration_s=0.0)
    assert kept == rows
    kept, report = filter_manifest(rows, {"x": frames + 1}, hop_s=256 / sr, min_duration_s=0.0)
    assert kept == [] and report.impossible_alignment == ["x"]


def test_items_are_never_truncated(corpus, g2p):
    rows = load_manifest(str(corpus))
    kept, _ = filter_manifest(rows, token_counts(rows, g2p), hop_s=256 / SR)
    vocab = build_vocabulary({"cs": [r["text"] for r in kept]})
    ap = AudioProcessor()
    ds = TTSDataset(kept, ap, vocab, g2p, speaker_ids={"narrator": 0})
    for i, r in enumerate(kept):
        item = ds[i]
        assert item["mel"].shape[1] == ap.n_frames(int(r["duration_s"] * SR))
        assert item["phoneme_ids"].numel() == len(g2p["cs"].tokenize(r["text"]))
        assert item["lang_ids"].numel() == item["phoneme_ids"].numel()


def test_unknown_symbol_is_an_error_not_unk(corpus, g2p):
    rows = load_manifest(str(corpus))[:3]
    vocab = build_vocabulary({"cs": [rows[0]["text"]]})  # deliberately too small
    ds = TTSDataset(rows, AudioProcessor(), vocab, g2p, speaker_ids={"narrator": 0})
    with pytest.raises(UnknownSymbolError):
        for i in range(len(ds)):
            ds[i]


def test_collator_carries_conditioning(corpus, g2p):
    rows = load_manifest(str(corpus))
    kept, _ = filter_manifest(rows, token_counts(rows, g2p), hop_s=256 / SR)
    vocab = build_vocabulary({"cs": [r["text"] for r in kept]})
    ds = TTSDataset(kept, AudioProcessor(), vocab, g2p, speaker_ids={"narrator": 0}, mel_mean=-5.0, mel_std=2.0)
    batch = TTSCollator()([ds[i] for i in range(len(ds))])
    assert batch["mels"].shape[:2] == (3, 100)
    assert batch["boundary_ids"].tolist() == [0, 1, 0]
    assert torch.allclose(batch["rates"][:, 0], torch.tensor(12.5))
    t_len = batch["phoneme_lengths"]
    assert (batch["phoneme_ids"][0, t_len[0]:] == 0).all()
    assert (batch["mels"][2, :, batch["mel_lengths"][2]:] == 0).all()


def _tone(seconds, freq=150.0):
    t = np.arange(int(seconds * SR)) / SR
    return (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _cached_dataset(tmp_path, g2p, ap, wav, seconds):
    sf.write(wav, _tone(seconds), SR)
    row = {"id": "u", "audio": str(wav), "text": "Ano.", "lang": "cs"}
    ds = TTSDataset([row], ap, {}, g2p, speaker_ids={"narrator": 0}, mel_cache_dir=str(tmp_path / "mels"))
    return ds, row


def test_mel_cache_is_reused_while_the_audio_is_unchanged(tmp_path, g2p):
    ap, calls = AudioProcessor(), []
    compute = ap.compute_mel
    ap.compute_mel = lambda wav: (calls.append(1), compute(wav))[1]
    ds, row = _cached_dataset(tmp_path, g2p, ap, tmp_path / "u.wav", 1.0)
    first, second = ds._mel(row), ds._mel(row)
    assert len(calls) == 1 and torch.equal(first, second)


def test_mel_cache_follows_a_recut_under_the_same_id(tmp_path, g2p):
    ap = AudioProcessor()
    ds, row = _cached_dataset(tmp_path, g2p, ap, tmp_path / "u.wav", 1.0)
    assert ds._mel(row).shape[1] == ap.n_frames(SR)
    sf.write(tmp_path / "u.wav", _tone(0.5, 300.0), SR)
    assert ds._mel(row).shape[1] == ap.n_frames(SR // 2)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_mel_cache_does_not_read_the_audio_when_the_row_has_its_sha256(tmp_path, g2p, monkeypatch):
    ds, row = _cached_dataset(tmp_path, g2p, AudioProcessor(), tmp_path / "u.wav", 1.0)
    row["sha256"] = _sha256(tmp_path / "u.wav")
    ds._mel(row)
    reads, read_bytes = [], Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda self: (reads.append(self), read_bytes(self))[1])
    ds._mel(row)
    assert reads == []


def test_mel_cache_is_keyed_by_the_sha256_of_the_row(tmp_path, g2p):
    ap, calls = AudioProcessor(), []
    compute = ap.compute_mel
    ap.compute_mel = lambda wav: (calls.append(1), compute(wav))[1]
    ds, row = _cached_dataset(tmp_path, g2p, ap, tmp_path / "u.wav", 1.0)
    ds._mel({**row, "sha256": "a" * 64}), ds._mel({**row, "sha256": "a" * 64}), ds._mel({**row, "sha256": "b" * 64})
    assert len(calls) == 2


def test_mel_cache_follows_a_recut_that_the_manifest_row_names(tmp_path, g2p):
    ap = AudioProcessor()
    ds, row = _cached_dataset(tmp_path, g2p, ap, tmp_path / "u.wav", 1.0)
    row["sha256"] = _sha256(tmp_path / "u.wav")
    assert ds._mel(row).shape[1] == ap.n_frames(SR)
    sf.write(tmp_path / "u.wav", _tone(0.5, 300.0), SR)
    assert ds._mel({**row, "sha256": _sha256(tmp_path / "u.wav")}).shape[1] == ap.n_frames(SR // 2)


def _stats_dataset(tmp_path, g2p, amplitudes):
    rows = []
    for i, amp in enumerate(amplitudes):
        path = tmp_path / f"s{i}.wav"
        sf.write(path, (amp * np.random.default_rng(i).standard_normal(SR // 2)).astype(np.float32), SR)
        rows.append({"id": f"s{i}", "audio": str(path), "text": "Ano.", "lang": "cs"})
    return TTSDataset(rows, AudioProcessor(), {}, g2p, speaker_ids={"narrator": 0})


def _mean_of(ds, indices):
    mels = [ds._mel(ds.rows[i]) for i in indices]
    return sum(m.sum().item() for m in mels) / sum(m.numel() for m in mels)


def test_mel_stats_sample_evenly_over_the_whole_set(tmp_path, g2p):
    ds = _stats_dataset(tmp_path, g2p, [0.01] * 4 + [0.3] * 4)
    mean, std = corpus_mel_stats(ds, max_items=4)
    assert mean == pytest.approx(_mean_of(ds, [0, 2, 4, 6]))
    assert abs(mean - _mean_of(ds, [0, 1, 2, 3])) > 1.0 and std > 0


def test_mel_stats_use_every_row_when_the_limit_allows(tmp_path, g2p):
    ds = _stats_dataset(tmp_path, g2p, [0.01, 0.05, 0.1, 0.3, 0.02])
    assert corpus_mel_stats(ds, max_items=50)[0] == pytest.approx(_mean_of(ds, range(5)))
    assert corpus_mel_stats(ds, max_items=5)[0] == pytest.approx(_mean_of(ds, range(5)))
