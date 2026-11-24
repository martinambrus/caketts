"""ASR runner of the corpus preparation stages: the long-form anchors of the aligner and the audit (Step 4.2) use it.

Usage:
    runner = AsrRunner(AsrConfig())
    runner.transcribe([clip_16k_mono, ...])      # one text per clip
    runner.transcribe_words(part_16k_mono)       # [AsrWord(text, start_s, end_s)] of a whole part
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SAMPLE_RATE = 16000
CUT_FRAME_S = 0.1


@dataclass(frozen=True)
class AsrWord:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class AsrSpec:
    model_id: str
    kind: str
    licence: str
    timestamps: bool


ASR_BACKENDS = {
    "parakeet-tdt-0.6b-v3": AsrSpec("nvidia/parakeet-tdt-0.6b-v3", "tdt", "CC-BY-4.0", True),
    "canary-1b-v2": AsrSpec("nvidia/canary-1b-v2", "canary", "CC-BY-4.0", False),
}


@dataclass
class AsrConfig:
    backend: str = "parakeet-tdt-0.6b-v3"
    device: str = "auto"
    language: str = "cs"
    window_s: float = 35.0
    overlap_s: float = 1.0
    batch_size: Optional[int] = None
    cut_search_s: float = 3.0


def transcription_identity(backend: str, language: str, *, legacy: bool = False) -> dict:
    """Book language is authoritative; TDT cannot be prompted to enforce it.

    Legacy Canary requests supplied source only; the installed processor defaulted
    target to source. Keep that history distinct when validated caches are reused.
    """
    canary = ASR_BACKENDS[backend].kind == "canary"
    return {"version": 1, "book_language": language,
            "language_control": ("source_default_target" if legacy else "source_target") if canary else "automatic",
            "source_language": language if canary else None,
            "target_language": language if canary else None}


def compatible_legacy_report(report: dict, config: dict, models: dict, parts: Sequence[str]) -> bool:
    """Only an old, matching producer report can vouch for missing cache language.

    Callers must additionally check each cache's current audio/model identity.
    Rescoring thresholds, chunk size and terminology do not change recognition.
    """
    if not isinstance(report, dict) or "asr_provenance" in report:
        return False
    saved = report.get("config")
    if not isinstance(saved, dict):
        return False
    return (saved.get("language") == config["language"] and report.get("judges") == models
            and report.get("asr_model") == " + ".join(models.values())
            and isinstance(report.get("parts"), dict) and set(parts) <= report["parts"].keys())


def validate_runner_language(runner, backend, language):
    """Injected real runners must not contradict the provenance we are recording."""
    if isinstance(runner, AsrRunner) and (runner.cfg.language != language or runner.cfg.backend != backend):
        raise ValueError("ASR runner differs from the book language or configured backend")


def plan_windows(audio: np.ndarray, sr: int, window_s: float, overlap_s: float,
                 cut_search_s: float) -> List[Tuple[float, float, float, float]]:
    """(window start, window end, keep start, keep end) in seconds; the keep ranges partition the audio."""
    duration = len(audio) / sr
    frame = int(CUT_FRAME_S * sr)
    cuts = [0.0]
    nominal = window_s
    while nominal < duration:
        lo = max(int(round((nominal - cut_search_s) * sr)), int(round(cuts[-1] * sr)))
        hi = int(round(nominal * sr))
        starts = list(range(lo, hi - frame + 1, frame))
        if starts:
            rms = [float(np.sqrt(np.mean(np.square(audio[s:s + frame])))) for s in starts]
            cut = (starts[int(np.argmin(rms))] + frame / 2) / sr
        else:
            cut = nominal
        cuts.append(cut)
        nominal += window_s
    cuts.append(duration)
    return [(max(0.0, a - overlap_s / 2), min(duration, b + overlap_s / 2), a, b) for a, b in zip(cuts, cuts[1:])]


def words_from_stamps(stamps: Sequence[Dict]) -> List[Tuple[str, float, float]]:
    words: List[List] = []
    for st in stamps:
        token = st["token"]
        if token.startswith(" ") or not words:
            words.append(["", None, st["end"]])
        word = words[-1]
        word[0] += token
        if (word[1] is None) and token.strip():
            word[1] = st["start"]
        word[2] = st["end"]
    return [(w[0].strip(), w[1], w[2]) for w in words if w[0].strip()]


class AsrRunner:
    def __init__(self, cfg: AsrConfig = AsrConfig(), *, model=None, processor=None):
        if cfg.backend not in ASR_BACKENDS:
            raise ValueError(f"unknown ASR backend {cfg.backend!r}; valid: {sorted(ASR_BACKENDS)}")
        self.cfg = cfg
        self.spec = ASR_BACKENDS[cfg.backend]
        self._model = model
        self._processor = processor

    @property
    def model_id(self) -> str:
        return self.spec.model_id

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoModelForTDT, AutoProcessor

        device = self.cfg.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        auto_model = AutoModelForTDT if self.spec.kind == "tdt" else AutoModelForSpeechSeq2Seq
        self._processor = AutoProcessor.from_pretrained(self.spec.model_id)
        self._model = auto_model.from_pretrained(self.spec.model_id, dtype=dtype).to(device).eval()

    @property
    def batch_size(self) -> int:
        if self.cfg.batch_size:
            return self.cfg.batch_size
        return 4 if str(self._model.device).startswith("cuda") else 1

    def close(self) -> None:
        self._model = None
        self._processor = None
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _generate(self, clips: Sequence[np.ndarray]):
        import torch

        if self.spec.kind == "canary":
            inputs = self._processor.apply_transcription_request(
                list(clips), source_language=self.cfg.language, target_language=self.cfg.language)
        else:
            inputs = self._processor(list(clips), sampling_rate=SAMPLE_RATE, return_tensors="pt")
        model = self._model
        inputs = {k: (v.to(model.device, model.dtype) if v.is_floating_point() else v.to(model.device))
                  for k, v in inputs.items()}
        extra = {"max_new_tokens": int(self.cfg.window_s * 10)} if self.spec.kind == "canary" else {}
        with torch.inference_mode():
            return model.generate(**inputs, **extra)

    def transcribe(self, clips: Sequence[np.ndarray]) -> List[str]:
        limit = int(self.cfg.window_s * SAMPLE_RATE)
        if any(len(c) > limit for c in clips):
            raise ValueError(f"clip longer than window_s={self.cfg.window_s}")
        self._load()
        texts = [""] * len(clips)
        todo = [i for i, c in enumerate(clips) if len(c) > 0]
        for b in range(0, len(todo), self.batch_size):
            idx = todo[b:b + self.batch_size]
            out = self._generate([clips[i] for i in idx])
            sequences = out if self.spec.kind == "canary" else out.sequences
            for i, text in zip(idx, self._processor.batch_decode(sequences, skip_special_tokens=True)):
                texts[i] = text.strip()
        return texts

    def transcribe_words(self, audio: np.ndarray) -> List[AsrWord]:
        if not self.spec.timestamps:
            raise ValueError(f"backend {self.cfg.backend!r} has no word timestamps")
        if len(audio) == 0:
            return []
        self._load()
        cfg = self.cfg
        plan = plan_windows(audio, SAMPLE_RATE, cfg.window_s, cfg.overlap_s, cfg.cut_search_s)
        words: List[AsrWord] = []
        for b in range(0, len(plan), self.batch_size):
            batch = plan[b:b + self.batch_size]
            clips = [audio[int(round(w[0] * SAMPLE_RATE)):int(round(w[1] * SAMPLE_RATE))] for w in batch]
            out = self._generate(clips)
            _, stamps = self._processor.decode(out.sequences.cpu(), durations=out.durations.cpu(),
                                               skip_special_tokens=True)
            for (win_start, _, keep_start, keep_end), win_stamps in zip(batch, stamps):
                for text, start, end in words_from_stamps(win_stamps):
                    if keep_start <= win_start + (start + end) / 2 < keep_end:
                        words.append(AsrWord(text, win_start + start, win_start + end))
        return sorted(words, key=lambda w: w.start)
