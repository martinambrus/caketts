"""
Long-form logic of the aligner (design D3): what is decided before and after the CTC model runs, as pure functions.

`match_words` pairs the text words with the ASR words, `plan_blocks` turns the pairs into blocks of at most
`block_max_s` (units the narrator did not read are isolated, speech that is not in the text breaks a block), and after
the model `resolve_monotonic` and `assign_status` make the spans monotone and grade them by the part's own scores.

Usage:
    pairs = match_words([fold(w) for w in text_words], [fold(w.text) for w in asr])
    plan = plan_blocks(words_per_unit, pairs, asr, duration_s, cfg)    # Plan(blocks, gaps, flags)
"""
from __future__ import annotations

import bisect
import unicodedata
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from src.data.prepare.align.types import AlignConfig, Gap, UnitSpan
from src.data.prepare.asr import AsrWord

SUSPECT_MARGIN_S = 0.2
OVERLAP_MAX_S = 0.3
_MAX_CELLS = 150_000_000


@dataclass(frozen=True)
class Block:
    first: int
    last: int
    t0: float
    t1: float
    suspect: bool


@dataclass
class Plan:
    blocks: List[Block]
    gaps: List[Gap]
    flags: Dict[int, Set[str]]


def fold(word: str) -> str:
    decomposed = unicodedata.normalize("NFD", word.lower())
    return "".join(c for c in decomposed if c.isalpha())


def match_words(text_keys: Sequence[str], asr_keys: Sequence[str]) -> List[Tuple[int, int]]:
    n, m = len(text_keys), len(asr_keys)
    if n == 0 or m == 0:
        return []
    if n * m > _MAX_CELLS:
        raise ValueError("split the part")
    ids: Dict[str, int] = {}
    text = np.array([ids.setdefault(k, len(ids)) if k else -1 for k in text_keys])
    asr = np.array([ids.setdefault(k, len(ids)) if k else -2 for k in asr_keys])
    dtype = np.int32 if n + m >= 32000 else np.int16
    ar = np.arange(m + 1, dtype=dtype)
    d = np.empty((n + 1, m + 1), dtype)
    d[0] = ar
    tmp = np.empty(m + 1, dtype)
    for i in range(1, n + 1):
        prev = d[i - 1]
        tmp[0] = i
        tmp[1:] = np.minimum(prev[:-1] + (asr != text[i - 1]), prev[1:] + 1)
        d[i] = np.minimum.accumulate(tmp - ar) + ar
    pairs = []
    i, j = n, m
    while i > 0 and j > 0:
        equal = text[i - 1] == asr[j - 1]
        if d[i, j] == d[i - 1, j - 1] + (not equal):
            if equal:
                pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif d[i, j] == d[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def word_times(n_words: int, pairs: Sequence[Tuple[int, int]], asr: Sequence[AsrWord],
               duration_s: float) -> Tuple[np.ndarray, np.ndarray, float]:
    idx = np.arange(n_words, dtype=float)
    if not pairs:
        rate = duration_s / max(n_words, 1)
        return idx * rate, (idx + 1) * rate, rate
    mi = np.array([i for i, _ in pairs], dtype=float)
    ms = np.array([asr[j].start for _, j in pairs])
    me = np.array([asr[j].end for _, j in pairs])
    rate = (ms[-1] - ms[0]) / (mi[-1] - mi[0]) if mi[-1] != mi[0] else duration_s / max(n_words, 1)

    def spread(values: np.ndarray) -> np.ndarray:
        out = np.interp(idx, mi, values)
        out = np.where(idx < mi[0], values[0] - (mi[0] - idx) * rate, out)
        return np.where(idx > mi[-1], values[-1] + (idx - mi[-1]) * rate, out)

    start = np.clip(spread(ms), 0.0, duration_s)
    end = np.maximum(np.clip(spread(me), 0.0, duration_s), start)
    return start, end, float(rate)


def plan_blocks(unit_words: Sequence[int], pairs: Sequence[Tuple[int, int]], asr: Sequence[AsrWord],
                duration_s: float, cfg: AlignConfig) -> Plan:
    n_units = len(unit_words)
    off = np.concatenate([[0], np.cumsum(unit_words)]).astype(int)
    n_words = int(off[-1])
    unit_of = np.repeat(np.arange(n_units), unit_words)
    est_start, est_end, rate = word_times(n_words, pairs, asr, duration_s)
    matched = np.zeros(n_units, int)
    for i, _ in pairs:
        matched[unit_of[i]] += 1
    suspect = (np.asarray(unit_words) > 0) & (matched == 0) if pairs else np.zeros(n_units, bool)

    gaps: List[Gap] = []
    flags: Dict[int, Set[str]] = {}
    breaks: Set[int] = set()
    if pairs:
        chain = [(-1, -1)] + list(pairs) + [(n_words, len(asr))]
        for (p0, a0), (p1, a1) in zip(chain, chain[1:]):
            t_words, a_words = p1 - p0 - 1, a1 - a0 - 1
            if not ((a_words >= cfg.gap_min_words) and (a_words - t_words >= cfg.gap_min_words)):
                continue
            span = asr[a1 - 1].end - asr[a0 + 1].start
            if span - t_words * rate < cfg.gap_min_s:
                continue
            gaps.append(Gap(asr[a0 + 1].start, asr[a1 - 1].end, "extra_speech"))
            u0 = int(unit_of[p0]) if p0 >= 0 else -1
            u1 = int(unit_of[p1]) if p1 < n_words else n_units
            if u1 == u0:
                flags.setdefault(u0, set()).add("gap_inside")
            else:
                breaks.add(u0)

    pair_words = [i for i, _ in pairs]

    def suspect_window(first: int, last: int) -> Tuple[float, float]:
        k = bisect.bisect_left(pair_words, int(off[first]))
        before = asr[pairs[k - 1][1]].end - SUSPECT_MARGIN_S if k > 0 else 0.0
        k = bisect.bisect_left(pair_words, int(off[last + 1]))
        after = asr[pairs[k][1]].start + SUSPECT_MARGIN_S if k < len(pairs) else duration_s
        t0, t1 = max(0.0, before), min(duration_s, after)
        if t1 - t0 < 2 * SUSPECT_MARGIN_S:
            t1 = min(duration_s, t0 + 2 * SUSPECT_MARGIN_S)
            t0 = max(0.0, t1 - 2 * SUSPECT_MARGIN_S)
        return t0, t1

    def normal_window(first: int, last: int) -> Tuple[float, float]:
        t0 = max(0.0, float(est_start[off[first]]) - cfg.margin_s)
        t1 = min(duration_s, float(est_end[off[last + 1] - 1]) + cfg.margin_s)
        return t0, t1

    runs: List[List[int]] = []
    for u in range(n_units):
        if unit_words[u] == 0:
            continue
        if runs:
            prev = runs[-1][-1]
            same = bool(suspect[u]) == bool(suspect[prev])
            if same and not any(prev <= b < u for b in breaks):
                runs[-1].append(u)
                continue
        runs.append([u])

    blocks: List[Block] = []
    for run in runs:
        if suspect[run[0]]:
            blocks.append(Block(run[0], run[-1], *suspect_window(run[0], run[-1]), True))
            continue
        first = run[0]
        for prev, u in zip(run, run[1:]):
            if est_end[off[u + 1] - 1] - est_start[off[first]] + 2 * cfg.margin_s > cfg.block_max_s:
                blocks.append(Block(first, prev, *normal_window(first, prev), False))
                first = u
        blocks.append(Block(first, run[-1], *normal_window(first, run[-1]), False))
    return Plan(blocks, gaps, flags)


def _clear(unit: UnitSpan, flag: Optional[str] = None) -> None:
    unit.start_s = unit.end_s = None
    unit.words = []
    unit.score_mean = unit.score_min = 0.0
    unit.status = "unmatched"
    if flag and flag not in unit.flags:
        unit.flags.append(flag)


def _add_flag(unit: UnitSpan, flag: str) -> None:
    if flag not in unit.flags:
        unit.flags.append(flag)


def _clamp(unit: UnitSpan, lo: Optional[float], hi: Optional[float]) -> None:
    words = unit.words
    if lo is not None:
        unit.start_s = lo
        words = [replace(w, start=max(w.start, lo), end=max(w.end, lo)) for w in words]
    if hi is not None:
        unit.end_s = hi
        words = [replace(w, start=min(w.start, hi), end=min(w.end, hi)) for w in words]
    unit.words = words


def resolve_monotonic(units: List[UnitSpan], suspect_ids: Set[str]) -> None:
    kept: List[UnitSpan] = []
    for cur in units:
        if cur.start_s is None:
            continue
        if cur.start_s >= cur.end_s:
            _clear(cur, "overlap")
            continue
        while kept and cur.start_s is not None and cur.start_s < kept[-1].end_s:
            prev = kept[-1]
            overlap = prev.end_s - cur.start_s
            prev_suspect, cur_suspect = prev.id in suspect_ids, cur.id in suspect_ids
            if prev_suspect != cur_suspect:
                loser = prev if prev_suspect else cur
                _clear(loser, "overlap")
                if loser is prev:
                    kept.pop()
                continue
            mid = (prev.end_s + cur.start_s) / 2
            _clamp(prev, None, mid)
            _clamp(cur, mid, None)
            for unit in (prev, cur):
                _add_flag(unit, "overlap_fixed")
                if overlap > OVERLAP_MAX_S:
                    _add_flag(unit, "overlap")
            if prev.start_s >= prev.end_s:
                _clear(prev, "overlap")
                kept.pop()
            if cur.start_s >= cur.end_s:
                _clear(cur, "overlap")
        if cur.start_s is not None:
            kept.append(cur)


def _median_sigma(values: np.ndarray, floor: float) -> Tuple[float, float]:
    med = float(np.median(values))
    return med, max(1.4826 * float(np.median(np.abs(values - med))), floor)


def assign_status(units: List[UnitSpan], cfg: AlignConfig) -> Dict[str, float]:
    placed = [u for u in units if u.start_s is not None]
    if not placed:
        return {k: 0.0 for k in ("score_median", "score_sigma", "ok_min", "unmatched_max", "edge_median",
                                 "edge_sigma", "edge_ok_min", "n_placed")}
    edge = {u.id: min(u.words[0].score, u.words[-1].score) if u.words else u.score_mean for u in placed}
    m, sigma = _median_sigma(np.array([u.score_mean for u in placed]), cfg.sigma_floor)
    m_e, sigma_e = _median_sigma(np.array(list(edge.values())), cfg.sigma_floor)
    ok_min, unmatched_max = m - cfg.ok_sigmas * sigma, cfg.unmatched_ratio * m
    edge_ok_min = m_e - cfg.edge_sigmas * sigma_e
    for u in placed:
        if u.score_mean < unmatched_max:
            _clear(u)
        elif ((u.score_mean >= ok_min) and (edge[u.id] >= edge_ok_min)
              and not ({"overlap", "no_tokens"} & set(u.flags))):
            u.status = "ok"
        else:
            u.status = "low_conf"
    return {"score_median": m, "score_sigma": sigma, "ok_min": ok_min, "unmatched_max": unmatched_max,
            "edge_median": m_e, "edge_sigma": sigma_e, "edge_ok_min": edge_ok_min, "n_placed": len(placed)}


def skipped_gaps(units: List[UnitSpan], duration_s: float) -> List[Gap]:
    gaps: List[Gap] = []
    last_end = 0.0
    run = False
    for k, u in enumerate(units):
        if u.start_s is not None:
            last_end, run = u.end_s, False
        elif ("empty" not in u.flags) and not run:
            run = True
            nxt = next((v.start_s for v in units[k + 1:] if v.start_s is not None), duration_s)
            gaps.append(Gap(last_end, nxt, "skipped_text"))
    return gaps
