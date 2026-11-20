"""CTC forced-alignment backends of the aligner (design D3): emissions of a wav2vec2-style model and the word spans.

`forced_spans` is the pure core (emissions and token ids in, word frames out); `CtcAligner` wraps a model around it.

Usage:
    aligner = CtcAligner(BACKENDS["mms-300m"], language="cs")
    aligner.align_block(block_audio_16k, t0, unit_ids, words_per_unit)    # [[WordSpan, ...] per unit]
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.data.prepare.align.types import WordSpan

SAMPLE_RATE = 16000
FRAME_SAMPLES = 320
LANG_CODES = {"cs": "ces", "sk": "slk", "en": "eng"}
UROMAN_FIRST_ID = 4


@dataclass(frozen=True)
class BackendSpec:
    model_id: str
    tokenizer: str
    licence: str


BACKENDS = {
    "mms-300m": BackendSpec("MahmoudAshraf/mms-300m-1130-forced-aligner", "uroman", "CC-BY-NC-4.0"),
    "xls-r-cs": BackendSpec("comodoro/wav2vec2-xls-r-300m-cs-250", "native", "Apache-2.0"),
    "mms-1b-all": BackendSpec("facebook/mms-1b-all", "native", "CC-BY-NC-4.0"),
}


class NoRoomError(ValueError):
    pass


def forced_spans(emissions, words: Sequence[Sequence[int]], blank: int,
                 delimiter: Optional[int] = None) -> List[Tuple[int, int, float]]:
    import torch
    import torchaudio.functional as F

    frames, vocab = emissions.shape
    star = vocab
    targets = [star]
    kept = [i for i, tokens in enumerate(words) if tokens]
    for n, i in enumerate(kept):
        if (n > 0) and (delimiter is not None):
            targets.append(delimiter)
        targets.extend(words[i])
    targets.append(star)
    equal_pairs = sum(1 for a, b in zip(targets, targets[1:]) if a == b)
    if frames < len(targets) + equal_pairs:
        raise NoRoomError(f"{frames} frames for {len(targets)} targets and {equal_pairs} repeats")

    padded = torch.cat([emissions, torch.zeros(frames, 1, dtype=emissions.dtype)], 1)[None]
    alignment, scores = F.forced_align(padded, torch.tensor([targets], dtype=torch.int32), blank=blank)
    merged = F.merge_tokens(alignment[0], scores[0].exp(), blank=blank)[1:-1]

    spans: List[Tuple[int, int, float]] = []
    k = 0
    prev_end = 0
    for n, i in enumerate(kept):
        if (n > 0) and (delimiter is not None):
            k += 1
        tokens = merged[k:k + len(words[i])]
        k += len(words[i])
        while len(spans) < i:
            spans.append((prev_end, prev_end, 0.0))
        total = sum(t.end - t.start for t in tokens)
        score = sum(t.score * (t.end - t.start) for t in tokens) / total
        prev_end = tokens[-1].end
        spans.append((tokens[0].start, prev_end, float(score)))
    while len(spans) < len(words):
        spans.append((prev_end, prev_end, 0.0))
    return spans


class CtcAligner:
    stride_s = FRAME_SAMPLES / SAMPLE_RATE

    def __init__(self, spec: BackendSpec, language: str = "cs", device: str = "auto",
                 window_s: float = 30.0, context_s: float = 2.0):
        self.spec = spec
        self.language = language
        self._device = device
        self.window = int(window_s * SAMPLE_RATE) // FRAME_SAMPLES * FRAME_SAMPLES
        self.context = int(context_s * SAMPLE_RATE) // FRAME_SAMPLES * FRAME_SAMPLES
        self._model: Any = None
        self._vocab: Optional[Dict[str, int]] = None
        self._blank = 0
        self._delimiter: Optional[int] = None
        self._target_lang: Optional[str] = None
        self._uroman: Any = None

    @property
    def model_id(self) -> str:
        return self.spec.model_id

    @property
    def device(self) -> str:
        if self._device != "auto":
            return self._device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def blank(self) -> int:
        self._load_vocab()
        return self._blank

    @property
    def delimiter(self) -> Optional[int]:
        self._load_vocab()
        return self._delimiter

    def _lang_code(self) -> str:
        if self.language not in LANG_CODES:
            raise ValueError(f"unknown language {self.language!r}; valid: {sorted(LANG_CODES)}")
        return LANG_CODES[self.language]

    def _load_vocab(self) -> None:
        if self._vocab is not None:
            return
        from huggingface_hub import hf_hub_download
        from transformers import AutoConfig

        with open(hf_hub_download(self.spec.model_id, "vocab.json"), encoding="utf-8") as f:
            vocab = json.load(f)
        if all(isinstance(v, dict) for v in vocab.values()):
            self._target_lang = self._lang_code()
            vocab = vocab[self._target_lang]
        self._blank = AutoConfig.from_pretrained(self.spec.model_id).pad_token_id
        if self.spec.tokenizer == "native":
            self._delimiter = vocab.get("|")
        self._vocab = vocab

    def tokens(self, word: str) -> List[int]:
        self._load_vocab()
        if self.spec.tokenizer == "uroman":
            if self._uroman is None:
                import uroman

                self._uroman = uroman.Uroman()
            text = self._uroman.romanize_string(word, lcode=self._lang_code()).lower()
            return [self._vocab[c] for c in text if (c in self._vocab) and (self._vocab[c] >= UROMAN_FIRST_ID)]
        return [self._vocab[c] for c in word.lower()
                if (c in self._vocab) and (self._vocab[c] not in (self._blank, self._delimiter))]

    def _load_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCTC

        self._load_vocab()
        device = self.device
        extra = {"target_lang": self._target_lang, "ignore_mismatched_sizes": True} if self._target_lang else {}
        dtype = torch.float16 if device == "cuda" else torch.float32
        self._model = AutoModelForCTC.from_pretrained(self.spec.model_id, dtype=dtype, **extra).to(device).eval()

    def emissions(self, audio: np.ndarray):
        import torch

        self._load_model()
        x = np.asarray(audio, dtype=np.float32)
        x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
        model = self._model
        parts = []
        with torch.inference_mode():
            for core in range(0, len(x), self.window):
                lo = max(0, core - self.context)
                hi = min(len(x), core + self.window + self.context)
                segment = torch.from_numpy(x[lo:hi]).to(model.device, model.dtype)[None]
                logp = model(segment).logits[0].float().log_softmax(-1).cpu()
                first = (core - lo) // FRAME_SAMPLES
                parts.append(logp[first:first + self.window // FRAME_SAMPLES])
        return torch.cat(parts)

    def align_block(self, audio: np.ndarray, t0: float, ids: Sequence[str],
                    words_per_unit: Sequence[Sequence[str]]) -> List[List[WordSpan]]:
        words = [w for unit in words_per_unit for w in unit]
        frames = forced_spans(self.emissions(audio), [self.tokens(w) for w in words], self.blank, self.delimiter)
        out: List[List[WordSpan]] = []
        k = 0
        for unit in words_per_unit:
            out.append([WordSpan(w, t0 + s * self.stride_s, t0 + e * self.stride_s, sc)
                        for w, (s, e, sc) in zip(unit, frames[k:k + len(unit)])])
            k += len(unit)
        return out
