"""Data pipeline: drop-don't-crop filtering, strict symbols, conditioning carried into batches."""
import json

import numpy as np
import pytest
import soundfile as sf
import torch

from src.data.audio import AudioProcessor
from src.data.dataset import TTSCollator, TTSDataset, filter_manifest, load_manifest
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
