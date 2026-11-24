"""The implemented training stage: bounded acoustic Matcha-TTS optimization."""
from dataclasses import asdict, fields
import math
from pathlib import Path

import yaml

from ..data.audio import MelConfig
from ..model.matcha_tts import MatchaConfig
from ..text.phonemizer import G2PConfig


def _known(value, allowed, name):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError(f"unsupported {name} settings")
    return value


def load_stage(path):
    """Resolve paths relative to this YAML, never the caller's working directory.

    Model overrides use supported flat MatchaConfig fields. The historical nested
    base template is deliberately not accepted: its extra options are not wired.
    """
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    _known(raw, {"stage", "output_dir", "data", "model", "audio", "training", "tempo"}, "stage")
    if type(raw.get("stage")) is not int or raw["stage"] != 2:
        raise ValueError("only acoustic stage 2 is implemented")
    data = {"speaker_ids": {"narrator": 0}, "min_duration_s": .5, "max_duration_s": 20.,
            "max_train_items": None, "max_val_items": None, "stats_max_items": 2000,
            "lexicon": None, "g2p": {}}
    data.update(_known(raw.get("data", {}), set(data) | {"train_manifest", "val_manifest", "vocab"}, "data"))
    def resolve(p):
        return str((path.parent / p).resolve())
    for key in ("train_manifest", "val_manifest", "vocab"):
        if not isinstance(data.get(key), str):
            raise ValueError(f"data.{key} is required")
        data[key] = resolve(data[key])
    if data["lexicon"] is not None:
        data["lexicon"] = resolve(data["lexicon"])
    _known(data["g2p"], {f.name for f in fields(G2PConfig)}, "g2p")
    g2p = G2PConfig(**data["g2p"])
    data["g2p"] = asdict(g2p)
    if (not isinstance(g2p.keep_stress, dict) or set(g2p.keep_stress) != {"cs", "sk", "en"} or
            any(type(v) is not bool for v in g2p.keep_stress.values()) or
            type(g2p.apply_voicing_rules) is not bool or type(g2p.apply_word_rules) is not bool):
        raise ValueError("g2p flags must be booleans with cs/sk/en stress settings")
    speakers = data["speaker_ids"]
    if (not isinstance(speakers, dict) or not speakers or
            any(not isinstance(k, str) or type(v) is not int for k, v in speakers.items()) or
            sorted(speakers.values()) != list(range(len(speakers)))):
        raise ValueError("speaker_ids must be a contiguous name to integer mapping")
    for key in ("max_train_items", "max_val_items", "stats_max_items"):
        n = data[key]
        if n is not None and (type(n) is not int or n <= 0):
            raise ValueError(f"data.{key} must be positive")
    if data["stats_max_items"] is None:
        raise ValueError("stats_max_items is required")
    for key in ("min_duration_s", "max_duration_s"):
        _positive(data[key], key)
    if data["min_duration_s"] > data["max_duration_s"]:
        raise ValueError("duration bounds are reversed")
    audio = asdict(MelConfig())
    audio.update(_known(raw.get("audio", {}), audio, "audio"))
    if audio != asdict(MelConfig()):
        raise ValueError("stage2 requires the BigVGAN 24kHz/100-mel feature settings")
    model = {k: v for k, v in asdict(MatchaConfig(n_vocab=1)).items() if k != "n_vocab"}
    model.update(_known(raw.get("model", {}), model, "model"))
    if type(model["n_mels"]) is not int or type(model["n_speakers"]) is not int or model["n_mels"] != 100 or model["n_speakers"] != len(speakers):
        raise ValueError("model mel/speaker counts must match audio and speaker_ids")
    if type(model["n_langs"]) is not int or type(model["n_boundaries"]) is not int or model["n_langs"] != 3 or model["n_boundaries"] != 3:
        raise ValueError("stage2 requires the existing 3-language/3-boundary mapping")
    for k in ("enc_dim", "enc_layers", "enc_heads", "enc_ff", "spk_dim", "cond_dim", "dur_hidden", "dec_dim", "dec_levels"):
        if type(model[k]) is not int or model[k] <= 0:
            raise ValueError(f"model.{k} must be a positive integer")
    if model["enc_dim"] % model["enc_heads"] or model["dec_dim"] % 8:
        raise ValueError("encoder heads must divide width; decoder width must be divisible by 8")
    for k in ("rate_mean", "rate_std"):
        _positive(model[k], k)
    for k in ("dropout", "p_uncond", "sigma_min"):
        if not isinstance(model[k], (float, int)) or not math.isfinite(model[k]) or not 0 <= model[k] < 1:
            raise ValueError(f"invalid model.{k}")
    train = {"device": "cpu", "precision": "fp32", "seed": 0, "max_steps": 100,
             "batch_frames": 4000, "bucket_size": 64, "learning_rate": 1e-4,
             "weight_decay": 0., "grad_clip": 1., "validate_every": 10, "checkpoint_every": 10}
    train.update(_known(raw.get("training", {}), train, "training"))
    if train["device"] not in ("cpu", "cuda") or train["precision"] != "fp32":
        raise ValueError("stage2 supports explicit cpu/cuda devices and fp32 precision only")
    for k in ("max_steps", "batch_frames", "bucket_size", "validate_every", "checkpoint_every"):
        if type(train[k]) is not int or train[k] <= 0:
            raise ValueError(f"training.{k} must be a positive integer")
    if type(train["seed"]) is not int or not 0 <= train["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    for k in ("learning_rate", "grad_clip"):
        _positive(train[k], k)
    if not isinstance(train["weight_decay"], (int, float)) or not math.isfinite(train["weight_decay"]) or train["weight_decay"] < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    tempo = {"articulation_rate": 12., "pause_ratio": .1}
    tempo.update(_known(raw.get("tempo", {}), tempo, "tempo"))
    _positive(tempo["articulation_rate"], "articulation_rate")
    if not isinstance(tempo["pause_ratio"], (int, float)) or not math.isfinite(tempo["pause_ratio"]) or not 0 <= tempo["pause_ratio"] <= 1:
        raise ValueError("pause_ratio must be in [0, 1]")
    if not isinstance(raw.get("output_dir"), str) or not raw["output_dir"]:
        raise ValueError("output_dir is required")
    return {"stage": 2, "output_dir": resolve(raw["output_dir"]), "data": data,
            "audio": audio, "model": model, "training": train, "tempo": tempo}


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
