"""Bounded real-audio stage2 runner controls; fixtures contain synthetic tones only."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml

from src.text.phonemizer import CzechSlovakPhonemizer, build_vocabulary

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def stage(tmp_path):
    g2p = CzechSlovakPhonemizer(language="sk")
    text = "Mám."
    vocab = build_vocabulary({"sk": [text]}, {"sk": g2p})
    (tmp_path / "vocab.json").write_text(json.dumps(vocab))
    for split, count in [("train", 3), ("val", 1)]:
        rows = []
        for i in range(count):
            clip = f"{split}{i}"
            n = 4800 + i * 256
            signal = .15 * np.sin(np.arange(n) * (0.05 + i * .01))
            if split == "val":
                signal *= .01
            sf.write(tmp_path / f"{clip}.wav", signal, 24000)
            rows.append({"id": clip, "audio": f"{clip}.wav", "text": text, "lang": "sk",
                         "duration_s": n / 24000, "n_tokens": len(g2p.tokenize(text)),
                         "articulation_rate": 99., "pause_ratio": .99})
        (tmp_path / f"{split}.jsonl").write_text("\n".join(map(json.dumps, rows)))
    cfg = {"stage": 2, "output_dir": "run", "data": {
        "train_manifest": "train.jsonl", "val_manifest": "val.jsonl", "vocab": "vocab.json",
        "speaker_ids": {"narrator": 0}, "min_duration_s": .1, "max_duration_s": 1.,
        "max_train_items": 2, "max_val_items": 1},
        "model": {"enc_dim": 8, "enc_layers": 1, "enc_heads": 1, "enc_ff": 16,
                  "spk_dim": 4, "cond_dim": 4, "dur_hidden": 8, "dec_dim": 8,
                  "dec_levels": 1, "dropout": .1},
        "training": {"device": "cpu", "precision": "fp32", "seed": 7,
                     "max_steps": 4, "batch_frames": 30, "bucket_size": 8,
                     "learning_rate": .001, "grad_clip": 1., "validate_every": 1,
                     "checkpoint_every": 1}, "tempo": {"articulation_rate": 12., "pause_ratio": .1}}
    path = tmp_path / "stage.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path, cfg


def rewrite(path, cfg):
    path.write_text(yaml.safe_dump(cfg))


def checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def test_cli_update_checkpoint_and_deterministic_resume(stage):
    path, cfg = stage
    # Compare trajectories under the same runtime even if pytest uses more threads.
    cli_env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

    def run_cli(*args):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/train.py"), "--config", str(path),
                                 *args], cwd=path.parent.parent,
                                capture_output=True, text=True, env=cli_env)
        assert result.returncode == 0, result.stderr
        runtime = json.loads((path.parent / cfg["output_dir"] / "runtime.json").read_text())
        assert runtime["threads"] == 1

    run_cli("--max-steps", "2")
    partial = checkpoint(path.parent / "run/latest.pt")
    assert partial["step"] == 2
    assert all(torch.isfinite(v).all() for v in partial["model"].values())
    assert any("exp_avg" in s for s in partial["optimizer"]["state"].values())
    # Zero-initialized decoder output has received a real optimizer update.
    assert partial["model"]["decoder.estimator.out.2.weight"].abs().sum() > 0
    run_cli("--resume", str(path.parent / "run/latest.pt"), "--max-steps", "4")
    resumed = checkpoint(path.parent / "run/latest.pt")
    cfg["output_dir"] = "whole"
    rewrite(path, cfg)
    run_cli()
    whole = checkpoint(path.parent / "whole/latest.pt")
    assert resumed["step"] == whole["step"] == 4
    for k in whole["model"]:
        torch.testing.assert_close(resumed["model"][k], whole["model"][k], rtol=0, atol=0)
    assert resumed["optimizer"]["param_groups"] == whole["optimizer"]["param_groups"]
    assert resumed["optimizer"]["state"].keys() == whole["optimizer"]["state"].keys()
    for param_id, values in whole["optimizer"]["state"].items():
        assert resumed["optimizer"]["state"][param_id].keys() == values.keys()
        for key, value in values.items():
            torch.testing.assert_close(resumed["optimizer"]["state"][param_id][key], value, rtol=0, atol=0)
    assert resumed["rng"]["python"] == whole["rng"]["python"]
    assert torch.equal(resumed["rng"]["torch"], whole["rng"]["torch"])
    assert not torch.equal(partial["rng"]["torch"], resumed["rng"]["torch"])
    assert resumed["rng"]["cuda"] is whole["rng"]["cuda"] is None
    assert resumed["sampler"] == whole["sampler"]
    assert resumed["mel_stats"] == whole["mel_stats"]
    log = [json.loads(x) for x in (path.parent / "run/metrics.jsonl").read_text().splitlines()]
    assert any("one_frame_fraction" in x for x in log)


def test_padded_frame_budget_and_oversized_sample():
    from src.training.trainer import FrameBudgetBatchSampler
    lengths = [10, 10, 25, 4, 9, 8]
    sampler = FrameBudgetBatchSampler(lengths, 30, seed=9, bucket_size=3)
    batches = sampler.batches(0)
    assert sorted(i for b in batches for i in b) == list(range(len(lengths)))
    assert all(max(lengths[i] for i in b) * len(b) <= 30 for b in batches)
    assert batches == sampler.batches(0)
    with pytest.raises(ValueError, match="frame budget"):
        FrameBudgetBatchSampler([31], 30)


def test_train_stats_constant_bootstrap_no_crop_and_validation(stage):
    from src.training.trainer import prepare_training, validate
    path, cfg = stage
    state = prepare_training(path)
    train = state["train"]
    assert len(train) == 2
    all_mels = torch.cat([train._mel(r).flatten() for r in train.rows])
    assert state["mel_stats"][0] == pytest.approx(all_mels.mean().item(), abs=1e-5)
    assert state["mel_stats"][1] == pytest.approx(all_mels.std(correction=0).item(), abs=1e-5)
    item = train[0]
    assert item["rate"].tolist() == pytest.approx([12., .1])
    assert item["mel"].shape[-1] == 4800 // 256
    assert item["phoneme_ids"].numel() == train.rows[0]["n_tokens"]
    assert "99" in (path.parent / "train.jsonl").read_text()
    before = copy.deepcopy(state["model"].state_dict())
    torch_rng = torch.get_rng_state().clone()
    validate(state)
    assert torch.equal(torch_rng, torch.get_rng_state())
    for k, value in before.items():
        assert torch.equal(value, state["model"].state_dict()[k])


@pytest.mark.parametrize("change", ["vocab", "lexicon", "manifest", "audio", "model"])
def test_resume_rejects_changed_inputs(stage, change):
    from src.training.trainer import run_training
    path, cfg = stage
    lex = path.parent / "lexicon.json"
    lex.write_text("{}")
    cfg["data"]["lexicon"] = "lexicon.json"
    rewrite(path, cfg)
    run_training(path, max_steps=1)
    if change == "vocab":
        p = path.parent / "vocab.json"
        v = json.loads(p.read_text()); v["new"] = len(v); p.write_text(json.dumps(v))
    elif change == "lexicon":
        lex.write_text('{"sk": {}}')
    elif change == "manifest":
        p = path.parent / "train.jsonl"; p.write_text(p.read_text() + "\n")
    elif change == "audio":
        p = path.parent / "train0.wav"; p.write_bytes(p.read_bytes() + b"extra")
    else:
        cfg["model"]["enc_ff"] = 24; rewrite(path, cfg)
    with pytest.raises(ValueError, match="identity"):
        run_training(path, resume=path.parent / "run/latest.pt", max_steps=2)


@pytest.mark.parametrize("change", ["id", "audio", "count", "symbol", "unsupported", "overwrite"])
def test_fail_early(stage, change):
    from src.training.trainer import run_training
    path, cfg = stage
    if change in ("id", "audio"):
        p = path.parent / "val.jsonl"; r = json.loads(p.read_text())
        r["id" if change == "id" else "audio"] = "train0" if change == "id" else "train0.wav"
        p.write_text(json.dumps(r))
    elif change == "count":
        p = path.parent / "train.jsonl"
        p.write_text("\n".join(json.dumps({**json.loads(line), "n_tokens": 0}) for line in p.read_text().splitlines()))
    elif change == "symbol":
        (path.parent / "vocab.json").write_text('{"<pad>": 0}')
    elif change == "unsupported":
        cfg["model"]["discriminators"] = {}; rewrite(path, cfg)
    else:
        (path.parent / "run").mkdir(); (path.parent / "run/precious").write_text("keep")
    with pytest.raises((ValueError, KeyError)):
        run_training(path, max_steps=1)
    assert not (path.parent / "run/latest.pt").exists()


def test_failed_atomic_save_preserves_previous_checkpoint(tmp_path, monkeypatch):
    from src.training.trainer import atomic_checkpoint
    path = tmp_path / "latest.pt"
    atomic_checkpoint({"step": 1}, path)
    def fail(*args, **kwargs):
        raise OSError("injected save failure")
    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError):
        atomic_checkpoint({"step": 2}, path)
    assert checkpoint(path)["step"] == 1


@pytest.mark.parametrize("section,key,value", [
    ("training", "precision", "bf16"), ("training", "max_steps", 0),
    ("training", "learning_rate", float("nan")), ("training", "weight_decay", -1),
    ("training", "batch_frames", True), ("training", "grad_clip", 0),
    ("data", "max_train_items", 0), ("data", "min_duration_s", -1),
    ("data", "g2p", {"keep_stress": {"sk": True}}),
    ("model", "enc_heads", 3), ("model", "dec_dim", 7),
    ("model", "rate_std", 0), ("model", "dropout", 1),
    ("tempo", "pause_ratio", 2), ("audio", "sample_rate", 22050),
])
def test_settings_rejected_before_output(stage, section, key, value):
    from src.training.trainer import run_training
    path, cfg = stage
    cfg.setdefault(section, {})[key] = value
    rewrite(path, cfg)
    with pytest.raises(ValueError):
        run_training(path)
    assert not (path.parent / "run").exists()


def test_leakage_check_includes_unselected_rows(stage):
    from src.training.trainer import run_training
    path, _ = stage
    # train2 is beyond max_train_items=2, but is still checked against validation.
    val = path.parent / "val.jsonl"
    row = json.loads(val.read_text()); row["audio"] = "train2.wav"
    val.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="leakage"):
        run_training(path)


def test_distributed_training_rejected(stage, monkeypatch):
    from src.training.trainer import run_training
    path, _ = stage
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="distributed"):
        run_training(path)


def test_nonempty_output_requires_resume_after_success(stage):
    from src.training.trainer import run_training
    path, _ = stage
    run_training(path, max_steps=1)
    before = (path.parent / "run/latest.pt").read_bytes()
    with pytest.raises(ValueError, match="nonempty"):
        run_training(path, max_steps=2)
    assert (path.parent / "run/latest.pt").read_bytes() == before


@pytest.mark.parametrize("stale", [False, True])
def test_declared_pcm_digest_is_verified(stage, stale):
    import hashlib
    from src.training.trainer import prepare_training
    path, _ = stage
    manifest = path.parent / "train.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    for row in rows:
        pcm, _ = sf.read(path.parent / row["audio"], dtype="int16")
        row["sha256"] = "0" * 64 if stale else hashlib.sha256(pcm.tobytes()).hexdigest()
    manifest.write_text("\n".join(map(json.dumps, rows)))
    if stale:
        with pytest.raises(ValueError, match="PCM digest differs"):
            prepare_training(path)
    else:
        assert len(prepare_training(path)["train"]) == 2


def test_resume_rejects_changed_g2p_implementation(stage, monkeypatch):
    from src.training import trainer
    path, _ = stage
    trainer.run_training(path, max_steps=1)
    identity = trainer._g2p_identity()
    monkeypatch.setattr(trainer, "_g2p_identity", lambda: {**identity, "implementation_sha256": "0" * 64})
    with pytest.raises(ValueError, match="identity"):
        trainer.run_training(path, resume=path.parent / "run/latest.pt", max_steps=2)
