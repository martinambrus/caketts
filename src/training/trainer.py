"""Small, explicit stage2 acoustic runner. Checkpoints are trusted local pickle files.

No corpus edits, text exports, audio crops, distributed training, inference,
external tracking, or later training stages are performed here.
"""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import random
import tempfile

import soundfile as sf
import torch

from ..data.audio import AudioProcessor, MelConfig
from ..data.dataset import BOUNDARY_IDS, TTSCollator, TTSDataset, corpus_mel_stats, load_manifest
from ..model.matcha_tts import MatchaConfig, MatchaTTS
from ..text.phonemizer import CzechSlovakPhonemizer, G2PConfig, tokens_to_ids
from .stages import load_stage


class FrameBudgetBatchSampler:
    """Deterministic shuffled length buckets; budget is max(lengths) * batch size."""
    def __init__(self, lengths, batch_frames, seed=0, bucket_size=64):
        if (type(batch_frames) is not int or batch_frames <= 0 or
                type(bucket_size) is not int or bucket_size <= 0 or not lengths):
            raise ValueError("positive frame budget/bucket size and nonempty lengths required")
        if any(type(n) is not int or n <= 0 or n > batch_frames for n in lengths):
            raise ValueError("sample exceeds frame budget or has invalid length")
        self.lengths, self.budget, self.seed, self.bucket_size = list(lengths), batch_frames, seed, bucket_size

    def batches(self, epoch):
        rng = random.Random(self.seed + epoch)
        order = list(range(len(self.lengths)))
        rng.shuffle(order)
        batches = []
        for start in range(0, len(order), self.bucket_size):
            bucket = sorted(order[start:start + self.bucket_size], key=lambda i: self.lengths[i])
            batch, longest = [], 0
            for i in bucket:
                n = self.lengths[i]
                if batch and max(longest, n) * (len(batch) + 1) > self.budget:
                    batches.append(batch)
                    batch, longest = [], 0
                batch.append(i)
                longest = max(longest, n)
            if batch:
                batches.append(batch)
        rng.shuffle(batches)
        return batches


def _digest_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _g2p_identity():
    """Bind pronunciation behavior even when symbol counts stay unchanged."""
    from phonemizer.backend import EspeakBackend
    from ..text import phonemizer
    phonemizer._ensure_espeak_library()
    return {"implementation_sha256": _digest_file(phonemizer.__file__),
            "backend": "espeak-ng", "espeak_version": list(EspeakBackend.version()),
            "phonemizer_version": version("phonemizer")}


def _read_inputs(cfg):
    """Validate split leakage against all rows, before selecting bounded subsets.

    Manifest sha256 may describe PCM bytes. Compute a separate file digest here;
    never compare the WAV container digest with that preparation metadata.
    """
    ids, paths, digests = set(), set(), set()
    splits, identities = {}, {}
    for split in ("train", "val"):
        manifest = Path(cfg["data"][f"{split}_manifest"])
        rows, audio_ids = load_manifest(str(manifest)), []
        if not rows:
            raise ValueError(f"empty {split} manifest")
        for row in rows:
            clip = row.get("id")
            if not isinstance(clip, str) or not clip or clip in ids:
                raise ValueError("duplicate or invalid clip ID: split leakage")
            ids.add(clip)
            audio_path = (manifest.parent / row["audio"]).resolve()
            digest = _digest_file(audio_path)
            # PCM identity also detects the same signal in different WAV containers.
            try:
                signal, sr = sf.read(audio_path, dtype="float32", always_2d=True)
                info = sf.info(audio_path)
            except Exception:
                raise ValueError(f"{clip}: invalid audio") from None
            pcm_id = hashlib.sha256(str(sr).encode() + signal.tobytes()).hexdigest()
            if "sha256" in row:
                declared = row["sha256"]
                if (not isinstance(declared, str) or len(declared) != 64 or
                        any(c not in "0123456789abcdef" for c in declared)):
                    raise ValueError(f"{clip}: invalid declared PCM digest")
                pcm, _ = sf.read(audio_path, dtype="int16")
                if hashlib.sha256(pcm.tobytes()).hexdigest() != declared:
                    raise ValueError(f"{clip}: declared PCM digest differs from audio")
            if str(audio_path) in paths or pcm_id in digests:
                raise ValueError(f"{clip}: duplicate audio identity: split leakage")
            paths.add(str(audio_path)); digests.add(pcm_id)
            if info.samplerate != cfg["audio"]["sample_rate"] or info.channels != 1:
                raise ValueError(f"{clip}: audio must be mono at the configured sample rate")
            if not torch.isfinite(torch.from_numpy(signal)).all():
                raise ValueError(f"{clip}: nonfinite audio")
            duration = info.frames / info.samplerate
            if not math.isfinite(float(row.get("duration_s", -1))) or abs(row.get("duration_s", -1) - duration) > 1 / info.samplerate:
                raise ValueError(f"{clip}: stored duration differs from audio")
            row = dict(row, audio=str(audio_path))
            row["_frames"] = info.frames // cfg["audio"]["hop_length"]
            row["_duration"] = duration
            audio_ids.append([clip, digest])
            # Remove pre-existing tempo labels in a copy, never alter the manifest.
            row.pop("articulation_rate", None); row.pop("pause_ratio", None)
            splits.setdefault(split, []).append(row)
        identities[split] = {"manifest": _digest_file(manifest), "audio": audio_ids}
    return splits, identities


def _rng_state(device="cpu"):
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if device == "cuda" else None}


def _restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def prepare_training(config_path, saved=None):
    cfg = load_stage(config_path)
    t, d = cfg["training"], cfg["data"]
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("distributed training is unsupported")
    if t["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; no automatic CPU fallback")
    splits, input_ids = _read_inputs(cfg)
    vocab = json.loads(Path(d["vocab"]).read_text(encoding="utf-8"))
    if (not isinstance(vocab, dict) or not vocab or vocab.get("<pad>") != 0 or
            any(not isinstance(k, str) or type(v) is not int for k, v in vocab.items()) or
            sorted(vocab.values()) != list(range(len(vocab)))):
        raise ValueError("vocabulary must map symbols to contiguous IDs with <pad>=0")
    model_cfg = dict(cfg["model"], n_vocab=len(vocab))
    chosen = {}
    for split, rows in splits.items():
        selected = [r for r in rows if d["min_duration_s"] <= r["_duration"] <= d["max_duration_s"]]
        limit = d[f"max_{split}_items"]
        chosen[split] = selected[:limit] if limit is not None else selected
        if not chosen[split]:
            raise ValueError(f"empty selected {split} split")
    stable_cfg = {**cfg, "training": {k: v for k, v in t.items() if k != "max_steps"}}
    stable_cfg.pop("output_dir")
    g2p_identity = _g2p_identity()
    identity = _digest({"config": stable_cfg, "inputs": input_ids, "g2p": g2p_identity,
                        "vocab": _digest_file(d["vocab"]),
                        "lexicon": _digest_file(d["lexicon"]) if d["lexicon"] else None,
                        "selected": {s: [r["id"] for r in rows] for s, rows in chosen.items()},
                        "model": model_cfg})
    if saved is not None and saved.get("identity") != identity:
        raise ValueError("resume identity differs: config, vocabulary, lexicon, manifest or audio changed")
    langs = {r["lang"] for rows in chosen.values() for r in rows}
    g2ps = {lang: CzechSlovakPhonemizer(language=lang, lexicon_path=d["lexicon"],
                                      config=G2PConfig(**d["g2p"])) for lang in langs}
    for rows in chosen.values():
        for row in rows:
            clip = row["id"]
            try:
                tokens = g2ps[row["lang"]].tokenize(row["text"], quote_before=row.get("quote_before", 0), quote_after=row.get("quote_after", 0))
                tokens_to_ids(tokens, vocab, strict=True)
            except Exception:
                raise ValueError(f"{clip}: tokenization or strict vocabulary check failed") from None
            if type(row.get("n_tokens")) is not int or row["n_tokens"] != len(tokens) or not tokens:
                raise ValueError(f"{clip}: stored n_tokens differs from tokenizer")
            if row["_frames"] < len(tokens):
                raise ValueError(f"{clip}: fewer frames than tokens; no cropping permitted")
            if row.get("speaker", "narrator") not in d["speaker_ids"] or row.get("boundary", "mid") not in BOUNDARY_IDS:
                raise ValueError(f"{clip}: unknown speaker or boundary")
    samplers = {s: FrameBudgetBatchSampler([r["_frames"] for r in rows], t["batch_frames"], t["seed"], t["bucket_size"]) for s, rows in chosen.items()}
    audio = AudioProcessor(MelConfig(**cfg["audio"]))
    rates = (cfg["tempo"]["articulation_rate"], cfg["tempo"]["pause_ratio"])
    datasets = {s: TTSDataset(rows, audio, vocab, g2ps, d["speaker_ids"], default_rates=rates) for s, rows in chosen.items()}
    stats = saved["mel_stats"] if saved is not None else corpus_mel_stats(datasets["train"], d["stats_max_items"])
    if len(stats) != 2 or not all(math.isfinite(x) for x in stats) or stats[1] <= 0:
        raise ValueError("invalid train-only mel statistics")
    for dataset in datasets.values():
        dataset.mel_mean, dataset.mel_std = stats
    random.seed(t["seed"]); torch.manual_seed(t["seed"])
    if t["device"] == "cuda":
        torch.cuda.manual_seed_all(t["seed"])
    model = MatchaTTS(MatchaConfig(**model_cfg)).to(device=t["device"], dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    state = {**datasets, "cfg": cfg, "model": model, "optimizer": optimizer, "identity": identity,
             "mel_stats": tuple(stats), "g2p_identity": g2p_identity, "samplers": samplers, "step": 0, "sampler": {"epoch": 0, "position": 0}}
    if saved is not None:
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        state["step"], state["sampler"] = saved["step"], dict(saved["sampler"])
        if state["step"] < 0 or state["sampler"]["epoch"] < 0 or not 0 <= state["sampler"]["position"] < len(samplers["train"].batches(state["sampler"]["epoch"])):
            raise ValueError("invalid checkpoint sampler cursor")
        _restore_rng(saved["rng"])
    return state


def _batch(state, split, indices):
    batch = TTSCollator()([state[split][i] for i in indices])
    budget = state["cfg"]["training"]["batch_frames"]
    if batch["mels"].shape[-1] * len(indices) > budget:
        raise ValueError("actual padded batch exceeds frame budget")
    device = state["cfg"]["training"]["device"]
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _metrics(out, batch):
    if not all(torch.isfinite(out[k]).all() for k in ("loss", "dur_loss", "prior_loss", "cfm_loss", "attn")):
        raise ValueError("nonfinite loss or MAS alignment")
    durations = []
    for i, (tx, ty) in enumerate(zip(batch["phoneme_lengths"], batch["mel_lengths"])):
        tx, ty = int(tx), int(ty)
        a = out["attn"][i]
        valid = a[:tx, :ty]
        if (not ((a == 0) | (a == 1)).all() or valid.sum() != ty or
                not (valid.sum(0) == 1).all() or not (valid.sum(1) >= 1).all() or
                a[tx:, :].any() or a[:, ty:].any()):
            raise ValueError("invalid MAS mask")
        sequence = valid.argmax(0)
        if ((sequence[1:] - sequence[:-1]) < 0).any() or ((sequence[1:] - sequence[:-1]) > 1).any():
            raise ValueError("nonmonotonic MAS mask")
        durations.extend(valid.sum(1).detach().cpu().long().tolist())
    hist = [sum(n == 1 for n in durations), sum(2 <= n <= 4 for n in durations),
            sum(5 <= n <= 9 for n in durations), sum(n >= 10 for n in durations)]
    return {k: float(out[k].detach()) for k in ("loss", "dur_loss", "prior_loss", "cfm_loss")} | {
        "duration_histogram": hist, "one_frame_fraction": hist[0] / len(durations)}


@torch.no_grad()
def validate(state):
    """Evaluation restores training mode and RNG; validation never changes the trajectory."""
    rng, model = _rng_state(state["cfg"]["training"]["device"]), state["model"]
    was_training = model.training
    model.eval()
    total, count = 0., 0
    try:
        for indices in state["samplers"]["val"].batches(0):
            batch = _batch(state, "val", indices)
            metrics = _metrics(model(batch), batch)
            total += metrics["loss"] * len(indices); count += len(indices)
        return total / count
    finally:
        model.train(was_training)
        _restore_rng(rng)


def atomic_checkpoint(value, path):
    """Save in the destination filesystem; failed writes leave the previous file intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".checkpoint-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(value, f); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _save(state, path):
    atomic_checkpoint({"format": 1, "model": state["model"].state_dict(),
                       "optimizer": state["optimizer"].state_dict(), "step": state["step"],
                       "sampler": state["sampler"], "rng": _rng_state(state["cfg"]["training"]["device"]),
                       "config": state["cfg"], "identity": state["identity"],
                       "mel_stats": state["mel_stats"], "g2p_identity": state["g2p_identity"]}, path)


def run_training(config_path, resume=None, max_steps=None):
    """Run a finite stage2 job; --max-steps is an absolute target, including on resume.

    Resume accepts only a trusted local checkpoint inside this run's output directory.
    Input/config identities are checked before model or optimizer state is loaded.
    """
    cfg = load_stage(config_path)
    target = cfg["training"]["max_steps"] if max_steps is None else max_steps
    if type(target) is not int or target <= 0:
        raise ValueError("max_steps must be a positive integer")
    output = Path(cfg["output_dir"])
    saved = None
    if resume is not None:
        resume = Path(resume).resolve()
        if resume.parent != output or not resume.is_file():
            raise ValueError("resume checkpoint must exist inside output_dir")
        # Explicitly trusted local semantics: RNG Python tuples require pickle.
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        if saved.get("format") != 1:
            raise ValueError("unsupported checkpoint format")
        if target < saved["step"]:
            raise ValueError("max_steps precedes checkpoint step")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("output_dir is nonempty; use an explicit valid resume")
    state = prepare_training(config_path, saved)
    output.mkdir(parents=True, exist_ok=True)
    provenance = {"python": platform.python_version(), "torch": str(torch.__version__),
                  "cuda": torch.version.cuda, "device": cfg["training"]["device"],
                  "threads": torch.get_num_threads(), "identity": state["identity"],
                  "g2p": state["g2p_identity"]}
    (output / "resolved_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (output / "runtime.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (output / "normalization.json").write_text(json.dumps({"identity": state["identity"], "mel_stats": state["mel_stats"], "tempo": cfg["tempo"]}), encoding="utf-8")
    model, optimizer = state["model"], state["optimizer"]
    model.train()
    with (output / "metrics.jsonl").open("a", encoding="utf-8") as log:
        while state["step"] < target:
            cursor = state["sampler"]
            batches = state["samplers"]["train"].batches(cursor["epoch"])
            batch = _batch(state, "train", batches[cursor["position"]])
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            metrics = _metrics(out, batch)
            out["loss"].backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError("nonfinite model parameters after update")
            state["step"] += 1
            cursor["position"] += 1
            if cursor["position"] == len(batches):
                cursor["epoch"] += 1; cursor["position"] = 0
            metrics.update({"step": state["step"], "grad_norm": float(grad)})
            if state["step"] % cfg["training"]["validate_every"] == 0:
                metrics["val_loss"] = validate(state)
            line = json.dumps(metrics, allow_nan=False)
            log.write(line + "\n"); log.flush()
            print(line, flush=True)
            if state["step"] % cfg["training"]["checkpoint_every"] == 0 or state["step"] == target:
                _save(state, output / "latest.pt")
    return state
