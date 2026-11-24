"""Entry point of the aligner (design D3): audio of a part plus its text units in, a span for every unit out.

Usage:
    result = align(audio_16k_mono, [{"id": "u1", "words": ["Dobrý", "den"]}, ...], AlignConfig())
    result.units[0].start_s, result.gaps, result.meta
"""
from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Set

import numpy as np

from src.data.prepare.align.ctc import BACKENDS, CtcAligner
from src.data.prepare.align.longform import (Block, assign_status, fold, match_words, plan_blocks,
                                             repair_first_words, resolve_monotonic, skipped_gaps, word_times)
from src.data.prepare.align.types import AlignConfig, AlignResult, UnitSpan, WordSpan
from src.data.prepare.asr import ASR_BACKENDS, SAMPLE_RATE, AsrRunner, AsrWord

LONGFORM_MODES = ("auto", "always")


def check_distinct_models(aligner_model_id: str, judge_model_id: str) -> None:
    if aligner_model_id == judge_model_id:
        raise ValueError(f"the aligner and the judge must be different models, both are {aligner_model_id!r}")


def validate_config(cfg: AlignConfig) -> None:
    if not np.isfinite(cfg.edge_repair_max_s) or cfg.edge_repair_max_s < 0:
        raise ValueError("edge_repair_max_s must be finite and nonnegative")
    if cfg.backend not in BACKENDS:
        raise ValueError(f"unknown aligner backend {cfg.backend!r}; valid: {sorted(BACKENDS)}")
    if cfg.longform not in LONGFORM_MODES:
        raise ValueError(f"unknown longform mode {cfg.longform!r}; valid: {list(LONGFORM_MODES)}")
    if cfg.asr.backend not in ASR_BACKENDS:
        raise ValueError(f"unknown ASR backend {cfg.asr.backend!r}; valid: {sorted(ASR_BACKENDS)}")
    check_distinct_models(BACKENDS[cfg.backend].model_id, ASR_BACKENDS[cfg.asr.backend].model_id)


def _to_numpy(audio: Any) -> np.ndarray:
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def _unit_span(uid: str, spans: List[WordSpan], flags: Set[str]) -> UnitSpan:
    weights = [len(w.text) for w in spans]
    mean = sum(w * s.score for w, s in zip(weights, spans)) / max(sum(weights), 1)
    if any((s.score == 0.0) and (s.end == s.start) for s in spans):
        flags = flags | {"no_tokens"}
    return UnitSpan(uid, spans[0].start, spans[-1].end, spans, mean, min(s.score for s in spans), "ok",
                    sorted(flags))


def align(audio, units: Sequence[Mapping[str, Any]], cfg: AlignConfig = AlignConfig(), *,
          asr_words: Optional[Sequence[AsrWord]] = None, backend=None, asr=None) -> AlignResult:
    validate_config(cfg)
    audio = _to_numpy(audio)
    duration = len(audio) / SAMPLE_RATE
    unit_words = [list(u["words"]) for u in units]
    counts = [len(w) for w in unit_words]
    result = [UnitSpan(u["id"], None, None, [], 0.0, 0.0, "unmatched", ["empty"] if not unit_words[i] else [])
              for i, u in enumerate(units)]
    backend = backend or CtcAligner(BACKENDS[cfg.backend], cfg.language, cfg.device, cfg.window_s, cfg.context_s)

    asr_model_id = None
    if (asr_words is not None) or (cfg.longform == "always") or (duration > cfg.block_max_s):
        asr_model_id = asr.model_id if asr is not None else ASR_BACKENDS[cfg.asr.backend].model_id
        check_distinct_models(backend.model_id, asr_model_id)
        if asr_words is None:
            runner = asr or AsrRunner(cfg.asr)
            try:
                asr_words = runner.transcribe_words(audio)
            finally:
                if asr is None:
                    runner.close()

    rate = None
    pairs = []
    if asr_words is not None:
        keys = [fold(w) for words in unit_words for w in words]
        pairs = match_words(keys, [fold(w.text) for w in asr_words])
        plan = plan_blocks(counts, pairs, asr_words, duration, cfg)
        rate = word_times(sum(counts), pairs, asr_words, duration)[2]
        blocks, gaps, plan_flags = plan.blocks, list(plan.gaps), plan.flags
    else:
        nonempty = [i for i, c in enumerate(counts) if c]
        blocks = [Block(nonempty[0], nonempty[-1], 0.0, duration, False)] if nonempty else []
        gaps, plan_flags = [], {}

    suspect_ids: Set[str] = set()
    for block in blocks:
        members = [i for i in range(block.first, block.last + 1) if counts[i]]
        chunk = audio[int(round(block.t0 * SAMPLE_RATE)):int(round(block.t1 * SAMPLE_RATE))]
        try:
            spans = backend.align_block(chunk, block.t0, [units[i]["id"] for i in members],
                                        [unit_words[i] for i in members])
        except ValueError:
            for i in members:
                result[i].flags.append("no_room")
            continue
        for i, unit_spans in zip(members, spans):
            flags = set(plan_flags.get(i, ()))
            if block.suspect:
                flags.add("suspect")
                suspect_ids.add(units[i]["id"])
            result[i] = _unit_span(units[i]["id"], unit_spans, flags)

    resolve_monotonic(result, suspect_ids)
    thresholds = assign_status(result, cfg)
    repaired = repair_first_words(result, counts, pairs, asr_words or [], cfg.edge_repair_max_s, audio)
    gaps = sorted(gaps + skipped_gaps(result, duration), key=lambda g: g.start)
    meta = {"backend": cfg.backend, "model_id": backend.model_id, "stride_ms": backend.stride_s * 1000.0,
            "first_words_repaired": repaired, "confidence_kind": "mean_token_posterior", "asr_model_id": asr_model_id, "n_blocks": len(blocks),
            "n_suspect_blocks": sum(1 for b in blocks if b.suspect), "thresholds": thresholds,
            "device": getattr(backend, "device", cfg.device), "duration_s": duration, "rate_s_per_word": rate}
    return AlignResult(result, gaps, meta)
