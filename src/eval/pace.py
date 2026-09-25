"""
Pace metrics M1-M6 (Step 14). Thresholds are PROPOSALS anchored to the ~5% just-noticeable
difference for tempo; no industry standard exists. Compute rates with an aligner that is
independent of the model (e.g. Parakeet-TDT-0.6B-v3 word/char timestamps), never from the
model's own durations (circular).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class Sentence:
    n_phones: int
    speech_s: float   # duration without pauses >= 100-150 ms
    position: int     # index within its paragraph (0-based)
    paragraph_len: int


def articulation_rate(s: Sentence) -> float:
    return s.n_phones / max(s.speech_s, 1e-6)


def m1_sentence_outliers(sentences: Sequence[Sentence], target_rate: float, tol: float = 0.12) -> List[int]:
    """M1: sentences more than tol (10-12%) away from the target articulation rate."""
    return [i for i, s in enumerate(sentences) if abs(articulation_rate(s) / target_rate - 1) > tol]


def m2_end_slowdown(sentences: Sequence[Sentence]) -> Dict[str, float]:
    """
    M2: median rate of the last two sentences of each text / median of the preceding ones,
    plus the least-squares slope of rate against relative position. Pass: 0.95 <= ratio <= 1.05.
    This is the metric for the ElevenLabs "slows down over the last sentences" symptom.
    """
    rates = np.array([articulation_rate(s) for s in sentences])
    if len(rates) < 4:
        return {"ratio": float("nan"), "slope": float("nan")}
    ratio = float(np.median(rates[-2:]) / np.median(rates[:-2]))
    pos = np.linspace(0, 1, len(rates))
    slope = float(np.polyfit(pos, rates / np.median(rates), 1)[0])
    return {"ratio": ratio, "slope": slope}


def m3_position_ab(rate_as_final: float, rate_with_continuation: float) -> float:
    """M3: same paragraph rendered as the final one vs followed by more text. Pass: <= 0.03-0.05."""
    return abs(rate_as_final / rate_with_continuation - 1)


def m4_cross_text_cv(chapter_medians: Sequence[float]) -> float:
    """M4: coefficient of variation of chapter-median rates. Pass: <= the narrator's own CV."""
    x = np.asarray(chapter_medians, dtype=float)
    return float(x.std() / x.mean())


def m5_probe_spread(probe_rates: Sequence[float]) -> float:
    """M5: one fixed probe sentence embedded in many texts; max relative spread of its rate."""
    x = np.asarray(probe_rates, dtype=float)
    return float((x.max() - x.min()) / np.median(x))


def m6_distribution_distance(synthetic_rates: Sequence[float], human_rates: Sequence[float]) -> float:
    """M6: 1-D Wasserstein distance between synthetic and human sentence-rate distributions."""
    a, b = np.sort(np.asarray(synthetic_rates, float)), np.sort(np.asarray(human_rates, float))
    q = np.linspace(0, 1, 200)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))
