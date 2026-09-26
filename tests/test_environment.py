import json
import os
from pathlib import Path

import pytest
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]
BIGVGAN_DIR = Path(os.environ.get("BIGVGAN_DIR", ROOT / "third_party" / "BigVGAN"))


def test_versions():
    import torch, torchaudio
    assert Version(torch.__version__.split("+")[0]) >= Version("2.14")   # not a string comparison
    assert Version(torchaudio.__version__.split("+")[0]) >= Version("2.11")


def test_espeak_languages():
    from src.text.phonemizer import CzechSlovakPhonemizer
    for lang in ("cs", "sk", "en"):   # 'en' is mapped to espeak-ng's 'en-us'
        assert CzechSlovakPhonemizer(lang).tokenize("test")


def test_audio_config_matches_vocoder():
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "model" / "matcha_base.yaml").read_text())["audio"]
    voc = json.loads((BIGVGAN_DIR / "configs" / "bigvgan_v2_24khz_100band_256x.json").read_text())
    assert (cfg["n_mels"], cfg["hop_length"], cfg["n_fft"], cfg["sample_rate"]) == \
        (voc["num_mels"], voc["hop_size"], voc["n_fft"], voc["sampling_rate"])


def test_cuda_available():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available: training will be slow")
