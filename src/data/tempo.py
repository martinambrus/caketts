"""
Tempo labels for the narrator data (Step 4.4) and the pace-conditioning inputs.

articulation rate = phones per second of SPEECH (pauses >= min_pause_s excluded)
pause ratio       = share of the utterance spent in pauses >= min_pause_s
Computed from token durations of an alignment (MAS durations from a first training pass, or a
forced aligner). The phones/s measure follows Data-Speech's nb_phonemes / utterance_length with
long silences removed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

PAUSE_KINDS = {"punct", "boundary"}


@dataclass
class TempoLabel:
    articulation_rate: float  # phones / s of speech
    pause_ratio: float        # 0..1


def tempo_from_alignment(kinds: Sequence[str], durations_frames: Sequence[int], hop_s: float,
                         min_pause_s: float = 0.12) -> TempoLabel:
    """
    kinds: token kinds from the phonemizer ('phone' | 'punct' | 'boundary' | 'stress')
    durations_frames: frames aligned to each token (MAS / forced aligner)
    A punctuation/boundary token longer than min_pause_s counts as a pause; shorter ones
    (and all phones) count as speech time.
    """
    n_phones, speech_s, pause_s = 0, 0.0, 0.0
    for k, d in zip(kinds, durations_frames):
        dur = d * hop_s
        if k in PAUSE_KINDS and dur >= min_pause_s:
            pause_s += dur
        else:
            speech_s += dur
            n_phones += k == "phone"
    total = speech_s + pause_s
    return TempoLabel(n_phones / max(speech_s, 1e-6), pause_s / max(total, 1e-6))


def flag_rate_outliers(rates: Dict[str, float], rel_tol: float = 0.12, z: float = 2.0) -> List[str]:
    """Utterances whose articulation rate is beyond +-rel_tol of the median or +-z SD (proposals)."""
    ids = list(rates)
    x = np.array([rates[i] for i in ids])
    med, sd = np.median(x), x.std() + 1e-9
    return [i for i, r in zip(ids, x) if abs(r - med) / med > rel_tol or abs(r - med) / sd > z]
