"""
ASR verification for every rendered sentence (Step 12/14).

Why relative thresholds: the best open Czech/Slovak ASR still errs on 5.5-11% of words of
HUMAN speech (FLEURS: Parakeet-TDT-0.6B-v3 cs 11.01%, sk 8.82%), so absolute WER targets like
"2-4%" are unmeasurable. Compare against the SAME ASR on the narrator's real recordings, and
look for failure SHAPES that ASR noise does not produce: long runs of deleted words (skips),
long runs of inserted/substituted words (hallucinations), repeated n-grams, truncated endings.
Run-length definitions follow Argmax, "Taming Long-form TTS" (Sep 2026): skip = >= 10
contiguous deleted words, hallucination = >= 20 contiguous inserted/substituted words; for
sentence-level audiobook QA we use much stricter defaults (see CheckConfig).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional

import jiwer


def normalize_for_asr(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"<[^>]+>", " ", text)              # drop <en> span tags
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class CheckConfig:
    max_cer_over_baseline: float = 0.05   # absolute CER margin over the narrator's own median
    max_deleted_run: int = 2              # words; a 3-word hole in one sentence is a skip
    max_inserted_run: int = 3
    tail_words: int = 2                   # the last words must be present (cut-off check)


@dataclass
class CheckResult:
    ok: bool
    cer: float
    wer: float
    reasons: List[str] = field(default_factory=list)
    suspect_word_indices: List[int] = field(default_factory=list)  # indices into the reference words


def check_sentence(reference: str, hypothesis: str, narrator_baseline_cer: float,
                   cfg: CheckConfig = CheckConfig()) -> CheckResult:
    ref, hyp = normalize_for_asr(reference), normalize_for_asr(hypothesis)
    cer = jiwer.cer(ref, hyp) if ref else 0.0
    out = jiwer.process_words(ref, hyp)
    reasons, suspects = [], []
    del_run = ins_run = 0
    max_del = max_ins = 0
    for chunk in out.alignments[0]:
        n_ref = chunk.ref_end_idx - chunk.ref_start_idx
        n_hyp = chunk.hyp_end_idx - chunk.hyp_start_idx
        if chunk.type == "delete":
            del_run += n_ref; max_del = max(max_del, del_run)
            suspects.extend(range(chunk.ref_start_idx, chunk.ref_end_idx))
        else:
            del_run = 0
        if chunk.type in ("insert", "substitute"):
            ins_run += n_hyp; max_ins = max(max_ins, ins_run)
            if chunk.type == "substitute":
                suspects.extend(range(chunk.ref_start_idx, chunk.ref_end_idx))
        else:
            ins_run = 0
    if cer > narrator_baseline_cer + cfg.max_cer_over_baseline:
        reasons.append(f"CER {cer:.3f} > baseline {narrator_baseline_cer:.3f} + {cfg.max_cer_over_baseline}")
    if max_del > cfg.max_deleted_run:
        reasons.append(f"skip: {max_del} contiguous words missing")
    if max_ins > cfg.max_inserted_run:
        reasons.append(f"hallucination: {max_ins} contiguous inserted/substituted words")
    ref_words = ref.split()
    hyp_words = hyp.split()
    if ref_words and ref_words[-cfg.tail_words:] != hyp_words[-cfg.tail_words:]:
        tail_ok = all(w in hyp_words[-(cfg.tail_words + 2):] for w in ref_words[-cfg.tail_words:])
        if not tail_ok:
            reasons.append("ending missing or cut off")
            suspects.extend(range(max(0, len(ref_words) - cfg.tail_words), len(ref_words)))
    rep = _repeated_ngram(hyp_words)
    if rep and rep not in " ".join(ref_words):
        reasons.append(f"repeated phrase not in text: '{rep}'")
    return CheckResult(not reasons, cer, out.wer, reasons, sorted(set(suspects)))


def _repeated_ngram(words: List[str], n: int = 3) -> Optional[str]:
    for i in range(len(words) - 2 * n + 1):
        if words[i:i + n] == words[i + n:i + 2 * n]:
            return " ".join(words[i:i + n])
    return None


def retry_actions(result: CheckResult) -> List[str]:
    """
    Durations are deterministic, so re-seeding the decoder cannot fix a duration-caused skip.
    Retries must change something that matters, in this order.
    """
    actions = []
    if any(r.startswith("skip") or r.startswith("ending") for r in result.reasons):
        actions.append("raise local length_scale (x1.15) on the suspect words")
    if result.suspect_word_indices:
        actions.append("check G2P of suspect words; add a lexicon entry if wrong")
    actions.append("re-split the sentence at a comma / render clauses separately")
    actions.append("flag for human listening if still failing")
    return actions
