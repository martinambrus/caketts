"""
Cut stage of the corpus pipeline (design D4): units + alignment spans + the -23 LUFS 24 kHz part -> clips
that can never share audio, with pads of about pad_s (shorter in a short gap, longer to reach a quiet edge frame), measured pauses and
the narrator's pause medians.

Inputs are text/<part>.units.jsonl, align/<part>.align.jsonl and audio/<part>.24k.wav of the work directory;
outputs are cut/<part>.cuts.jsonl, cut/<part>.rejects.jsonl, cut/clips/<id>.wav, cut/pauses.json and a report.
Positions are integer samples at the part's sample rate; clips are half-open [start, end). run_cut takes every part of the
run in one call: the report, the pause medians and the audit's noise floors describe exactly the parts of the last call.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import soundfile as sf

from src.data.prepare.gold import _load_word_mids
from src.text.phonemizer import _SPAN_RE, _WORD_RE

GAP_TYPES = {"mid": "sentence", "para_end": "paragraph", "chapter_end": "chapter"}


class CutError(ValueError):
    """A hard assertion of the stage failed; the message names the unit or clip ids."""


@dataclass
class CutConfig:
    pad_s: float = 0.15
    frame_s: float = 0.010
    search_s: float = 0.08
    min_quiet_s: float = 0.04
    quiet_db: float = 15.0
    speech_db: float = 15.0
    floor_percentile: float = 5.0
    max_est_s: float = 17.0
    max_clip_s: float = 20.0
    min_split_words: int = 3
    close_s: float = 0.03
    bridge_s: float = 0.35


@dataclass
class SeqItem:
    unit: dict
    span: dict
    pos: int
    s: int
    e: int
    ok: bool
    s_ref: int = 0
    e_ref: int = 0


@dataclass
class Clip:
    id: str
    units: list[str]
    text: str
    boundary: str
    part: str
    start_sample: int
    end_sample: int
    speech_start: int
    speech_end: int
    al_start: int
    al_end: int
    pad_start_s: float
    pad_end_s: float
    gap_before_s: Optional[float]
    gap_after_s: Optional[float]
    flags: list[str]
    sha256: str

    def record(self) -> dict:
        r = lambda x: None if x is None else round(x, 6)
        return {"id": self.id, "units": self.units, "text": self.text, "boundary": self.boundary, "part": self.part,
                "start_sample": self.start_sample, "end_sample": self.end_sample,
                "pad_start_s": r(self.pad_start_s), "pad_end_s": r(self.pad_end_s),
                "gap_before_s": r(self.gap_before_s), "gap_after_s": r(self.gap_after_s),
                "flags": self.flags, "sha256": self.sha256}


@dataclass
class PartPlan:
    clips: list[Clip]
    rejects: list[dict]
    gaps: list[tuple[str, float]]
    commas: list[float]
    stats: dict = field(default_factory=dict)


def to_sample(t_s: float, sr: int) -> int:
    return int(round(t_s * sr))


def clip_bounds(s_ref: int, e_ref: int, c_prev: int, c_next: int, pad: int) -> tuple[int, int]:
    return max(s_ref - pad, c_prev), min(e_ref + pad, c_next)


def _loud_window(pcm: np.ndarray, a: int, frame: int, floor_db: float, quiet_db: float) -> bool:
    return len(pcm) >= a + frame and a >= 0 and float(frame_db(pcm[a:a + frame], frame)[0] - floor_db) > quiet_db


def quiet_start(pcm: np.ndarray, a: int, c_prev: int, prev_quiet: bool, floor_db: float, quiet_db: float, frame: int,
                room: int) -> int:
    if a <= c_prev or not _loud_window(pcm, a, frame, floor_db, quiet_db):
        return a
    x = a
    while (x - frame > c_prev) and (a - (x - frame) <= room):
        x -= frame
        if not _loud_window(pcm, x, frame, floor_db, quiet_db):
            return x
    if x - frame > c_prev:
        return x
    return c_prev if (prev_quiet and (a - c_prev <= room)) else a


def quiet_end(pcm: np.ndarray, b: int, c_next: int, next_quiet: bool, floor_db: float, quiet_db: float, frame: int,
              room: int) -> int:
    if b >= c_next or not _loud_window(pcm, b - frame, frame, floor_db, quiet_db):
        return b
    x = b
    while (x + frame < c_next) and ((x + frame) - b <= room):
        x += frame
        if not _loud_window(pcm, x - frame, frame, floor_db, quiet_db):
            return x
    if x + frame < c_next:
        return x
    return c_next if (next_quiet and (c_next - b <= room)) else b


def frame_db(pcm: np.ndarray, frame: int) -> np.ndarray:
    n = len(pcm) // frame
    out = np.empty(n)
    step = 4096
    for a in range(0, n, step):
        b = min(a + step, n)
        x = pcm[a * frame:b * frame].reshape(b - a, frame).astype(np.float64) / 32768
        out[a:b] = 10 * np.log10(np.mean(x ** 2, axis=1) + 1e-10)
    return out


def noise_floor_db(db: np.ndarray, percentile: float) -> float:
    return float(np.percentile(db, percentile))


def validate_spans(part: str, units: Sequence[dict], spans: Sequence[dict], sr: int) -> tuple[list[SeqItem], list[dict]]:
    ids = {u["id"] for u in units}
    by_id: dict = {}
    for sp in spans:
        if sp["id"] not in ids:
            raise CutError(f"{part}: span id {sp['id']} is not a unit id")
        if sp["id"] in by_id:
            raise CutError(f"{part}: span id {sp['id']} appears twice")
        by_id[sp["id"]] = sp
    seq: list[SeqItem] = []
    rejects: list[dict] = []
    for pos, u in enumerate(units):
        sp = by_id.get(u["id"])
        if sp is None or sp.get("start_s") is None or sp.get("end_s") is None:
            rejects.append({"id": u["id"], "reason": "no_span"})
            continue
        status = sp.get("status") or "ok"
        s, e = to_sample(sp["start_s"], sr), to_sample(sp["end_s"], sr)
        if status == "ok":
            if e <= s:
                raise CutError(f"{part}: unit {u['id']} has an empty or reversed span ({s}, {e})")
        else:
            rejects.append({"id": u["id"], "reason": status})
            if e <= s:
                continue
        seq.append(SeqItem(unit=u, span=sp, pos=pos, s=s, e=e, ok=status == "ok"))
    for a, b in zip(seq, seq[1:]):
        if a.e > b.s:
            raise CutError(f"{part}: spans of units {a.unit['id']} and {b.unit['id']} overlap or are out of order")
    return seq, rejects


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _anchor(db: np.ndarray, thr: float, lo: int, hi: int, ref: int, late: bool) -> Optional[int]:
    lo, hi = max(lo, 0), min(hi, len(db))
    if lo >= hi:
        return None
    idx = np.flatnonzero(db[lo:hi] > thr) + lo
    if not idx.size:
        return None
    dist = np.abs(idx - ref)
    near = idx[dist == dist.min()]
    return int(near.max() if late else near.min())


def _extend(db: np.ndarray, thr: float, f: int, look: int, limit: int, forward: bool, bridge: int = 0) -> int:
    while True:
        if forward:
            a, b = f + 1, min(f + 1 + look, limit, len(db))
        else:
            a, b = max(f - look, limit, 0), f
        hit = np.flatnonzero(db[a:b] > thr) if a < b else np.empty(0, dtype=int)
        if hit.size:
            f = a + int(hit[-1] if forward else hit[0])
            continue
        if bridge <= look:
            return f
        if forward:
            a, b = f + 1, min(f + 1 + bridge, limit, len(db))
        else:
            a, b = max(f - bridge, limit, 0), f
        cand = (a + np.flatnonzero(db[a:b] > thr)) if a < b else np.empty(0, dtype=int)
        for k in (cand if forward else cand[::-1]):
            end = _extend(db, thr, int(k), look, limit, forward)
            lo, hi = (int(k), end) if forward else (end, int(k))
            if int((db[lo:hi + 1] > thr).sum()) >= max(1, look):
                f = int(k)
                break
        else:
            return f


def _edges(db, thr, e, s, cfg, sr, bridge):
    F, m = round(cfg.frame_s * sr), round(cfg.search_s * sr)
    look = round(cfg.close_s * sr / F)
    fm = ((e + s) // 2) // F
    fe = _anchor(db, thr, (e - m) // F, min(fm, _ceil_div(e + m, F)), max(e - 1, 0) // F, True)
    fs = _anchor(db, thr, max(fm, (s - m) // F), _ceil_div(s + m, F), s // F, False)
    return ((e if fe is None else (_extend(db, thr, fe, look, fm, True, bridge) + 1) * F),
            (s if fs is None else _extend(db, thr, fs, look, fm, False, bridge) * F))


def refine_gap(db: np.ndarray, floor_db: float, e: int, s: int, cfg: CutConfig, sr: int) -> tuple[int, int]:
    F, thr = round(cfg.frame_s * sr), floor_db + cfg.speech_db
    bridge = round(cfg.bridge_s * sr / F)
    e1, s1 = _edges(db, thr, e, s, cfg, sr, bridge)
    if not bridge:
        return e1, s1
    e0, s0 = _edges(db, thr, e, s, cfg, sr, 0)
    for ce, cs in ((e1, s1), (e0, s1), (e1, s0)):
        if ce < cs and choose_cut(db, floor_db, ce, cs, cfg, sr) is not None:
            return ce, cs
    return e0, s0


def refine_start(db: np.ndarray, floor_db: float, s: int, cfg: CutConfig, sr: int) -> int:
    F, m = round(cfg.frame_s * sr), round(cfg.search_s * sr)
    thr, look = floor_db + cfg.speech_db, round(cfg.close_s * sr / F)
    f = _anchor(db, thr, (s - m) // F, _ceil_div(s + m, F), s // F, False)
    return s if f is None else _extend(db, thr, f, look, 0, False, round(cfg.bridge_s * sr / F)) * F


def refine_end(db: np.ndarray, floor_db: float, e: int, cfg: CutConfig, sr: int) -> int:
    F, m = round(cfg.frame_s * sr), round(cfg.search_s * sr)
    thr, look = floor_db + cfg.speech_db, round(cfg.close_s * sr / F)
    f = _anchor(db, thr, (e - m) // F, _ceil_div(e + m, F), max(e - 1, 0) // F, True)
    return e if f is None else (_extend(db, thr, f, look, len(db), True, round(cfg.bridge_s * sr / F)) + 1) * F


def can_merge(a: SeqItem, b: SeqItem, gap: int, group_est_s: float, cfg: CutConfig, sr: int) -> bool:
    ua, ub = a.unit, b.unit
    pad, window = to_sample(cfg.pad_s, sr), to_sample(cfg.search_s, sr)
    return bool(a.ok and b.ok and (b.pos - a.pos == 1) and gap < 2 * pad and (b.s - a.e) < 2 * (pad + window)
                and ua.get("boundary") == "mid"
                and not (ua.get("split_of") and ua.get("split_of") == ub.get("split_of"))
                and ua.get("paragraph") == ub.get("paragraph") and ua.get("chapter") == ub.get("chapter")
                and not ua.get("heading") and not ub.get("heading") and group_est_s <= cfg.max_est_s)


def choose_cut(db: np.ndarray, floor_db: float, lo: int, hi: int, cfg: CutConfig, sr: int) -> Optional[int]:
    F = round(cfg.frame_s * sr)
    fa, fb = max(_ceil_div(lo, F), 0), min(hi // F, len(db))
    if fa >= fb:
        return None
    q = np.concatenate(([False], db[fa:fb] <= floor_db + cfg.quiet_db, [False])).astype(np.int8)
    d = np.diff(q)
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    lengths = ends - starts
    ok = lengths * F >= cfg.min_quiet_s * sr
    if not ok.any():
        return None
    k = int(np.argmax(np.where(ok, lengths, -1)))
    return ((fa + int(starts[k])) * F + (fa + int(ends[k])) * F) // 2


def check_clips(clips: Sequence[Clip], n_samples: int, tol: int) -> None:
    for c in clips:
        a, b = c.start_sample, c.end_sample
        if a >= b:
            raise CutError(f"clip {c.id}: empty or reversed [{a}, {b})")
        if a < 0 or b > n_samples:
            raise CutError(f"clip {c.id}: [{a}, {b}) outside the part (0, {n_samples})")
        if a > c.speech_start or b < c.speech_end:
            raise CutError(f"clip {c.id}: [{a}, {b}) cuts into its speech [{c.speech_start}, {c.speech_end})")
        if a > c.al_start + tol or b < c.al_end - tol:
            raise CutError(f"clip {c.id}: [{a}, {b}) misses the aligner span [{c.al_start}, {c.al_end}) by more than {tol}")
    for c, d in zip(clips, clips[1:]):
        if c.end_sample > d.start_sample:
            raise CutError(f"clips {c.id} and {d.id} overlap or are out of order")
    if sum(c.end_sample - c.start_sample for c in clips) > n_samples:
        raise CutError(f"clips {[c.id for c in clips]} are longer in total than the part ({n_samples} samples)")


def check_unique_hashes(items: Iterable[tuple[str, str]]) -> None:
    seen: dict = {}
    for cid, h in items:
        if h in seen:
            raise CutError(f"clips {seen[h]} and {cid} hold identical audio")
        seen[h] = cid


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _word_spans(text: str) -> list[tuple[int, int]]:
    out, pos = [], 0
    for s in _SPAN_RE.finditer(text):
        out += [m.span() for m in _WORD_RE.finditer(text, pos, s.start())]
        out += [m.span() for m in _WORD_RE.finditer(text, s.start(2), s.end(2))]
        pos = s.end()
    return out + [m.span() for m in _WORD_RE.finditer(text, pos, len(text))]


def _cap_ranges(words: list, pos: list, tags: list, lo: int, hi: int, cfg: CutConfig) -> list[tuple[int, int]]:
    if words[hi - 1]["end"] - words[lo]["start"] + 2 * cfg.pad_s + 2 * (cfg.search_s + cfg.bridge_s) <= cfg.max_clip_s:
        return [(lo, hi)]
    n, m = hi - lo, cfg.min_split_words
    ks = [k for k in range(lo + m, hi - m + 1) if not any(a < pos[k][0] < b for a, b in tags)]
    if not ks:
        return [(lo, hi)]
    k = max(ks, key=lambda k: (words[k]["start"] - words[k - 1]["end"], -abs(2 * (k - lo) - n), -k))
    return _cap_ranges(words, pos, tags, lo, k, cfg) + _cap_ranges(words, pos, tags, k, hi, cfg)


def split_over_cap(units: Sequence[dict], spans: Sequence[dict], cfg: CutConfig, sr: int) -> tuple[list[dict], list[dict]]:
    ids = Counter(s.get("id") for s in spans)
    by_id = {s.get("id"): s for s in spans}
    new_units: list[dict] = []
    new_spans: dict = {}  # parent id -> its piece spans
    for u in units:
        span, text = by_id.get(u.get("id")), u.get("text") or ""
        words = (span or {}).get("words")
        pos = _word_spans(text)
        if (ids[u.get("id")] != 1 or span.get("status", "ok") != "ok" or not _num(span.get("start_s"))
                or not _num(span.get("end_s")) or not span["start_s"] < span["end_s"]
                or (span["end_s"] - span["start_s"]) + 2 * cfg.pad_s + 2 * (cfg.search_s + cfg.bridge_s) <= cfg.max_clip_s
                or not isinstance(words, list) or len(words) != len(pos)
                or not all(isinstance(w, dict) and _num(w.get("start")) and _num(w.get("end")) for w in words)):
            new_units.append(u)
            continue
        tags = [m.span() for m in _SPAN_RE.finditer(text)]
        ranges = _cap_ranges(words, pos, tags, 0, len(words), cfg)
        if len(ranges) == 1:
            new_units.append(u)
            continue
        whole = words[-1]["end"] - words[0]["start"]
        for j, (lo, hi) in enumerate(ranges):
            last = j == len(ranges) - 1
            ptext = text[0 if j == 0 else pos[lo][0]:len(text) if last else pos[hi][0]].strip()
            pid = f"{u['id']}.{j + 1}"
            piece = words[lo:hi]
            new_units.append({**u, "id": pid, "text": ptext, "words": [ptext[x:y] for x, y in _word_spans(ptext)],
                              "boundary": u.get("boundary") if last else "mid", "split_of": u["id"],
                              "est_s": u.get("est_s", 0.0) * (piece[-1]["end"] - piece[0]["start"]) / whole if whole > 0 else u.get("est_s", 0.0)})
            new_spans.setdefault(u["id"], []).append({"id": pid, "start_s": piece[0]["start"], "end_s": piece[-1]["end"], "words": piece, "status": "ok",
                              **{k: span[k] for k in ("score_mean", "score_min") if k in span}})
    return new_units, [x for s in spans for x in new_spans.get(s.get("id"), [s])]


def comma_gaps(unit: dict, span: dict) -> list[float]:
    words = span.get("words") or []
    pos = _word_spans(unit.get("text") or "")
    if not words or len(pos) != len(words):
        return []
    text = unit["text"]
    return [words[k + 1]["start"] - words[k]["end"] for k in range(len(words) - 1)
            if "," in text[pos[k][1]:pos[k + 1][0]]]


def pause_stats(gaps: Sequence[tuple[str, float]], commas: Sequence[float]) -> dict:
    groups = {"comma": list(commas), "sentence": [], "paragraph": [], "chapter": []}
    for kind, sec in gaps:
        groups[kind].append(sec)
    out = {}
    for k, v in groups.items():
        if not v:
            out[k] = {"n": 0, "median_s": None, "p10_s": None, "p90_s": None}
        else:
            p10, p50, p90 = (float(x) for x in np.percentile(v, [10, 50, 90]))
            out[k] = {"n": len(v), "median_s": p50, "p10_s": p10, "p90_s": p90}
    return out


def _holds_word_mid(mids: Optional[np.ndarray], lo: int, hi: int, sr: int) -> bool:
    return (mids is not None) and bool(((mids >= lo / sr) & (mids < hi / sr)).any())


def plan_part(part: str, pcm: np.ndarray, sr: int, units: Sequence[dict], spans: Sequence[dict],
              cfg: CutConfig = CutConfig(), word_mids: Optional[np.ndarray] = None) -> PartPlan:
    units, spans = split_over_cap(units, spans, cfg, sr)
    seq, rejects = validate_spans(part, units, spans, sr)
    F, pad = round(cfg.frame_s * sr), to_sample(cfg.pad_s, sr)
    reason_counts = Counter(r["reason"] for r in rejects)
    stats = {"units": len(units), "ok": sum(1 for it in seq if it.ok), "rejects": dict(reason_counts), "clips": 0,
             "merged_clips": 0, "flags": {}, "floor_db": None, "quiet_thr_db": None, "speech_thr_db": None,
             "edges_unconfirmed": 0}
    if not any(it.ok for it in seq):
        return PartPlan([], rejects, [], [], stats)
    db = frame_db(pcm, F)
    floor = noise_floor_db(db, cfg.floor_percentile)
    stats.update(floor_db=floor, quiet_thr_db=floor + cfg.quiet_db, speech_thr_db=floor + cfg.speech_db)

    n = len(seq)
    starts, ends = [0] * n, [0] * n
    starts[0] = refine_start(db, floor, seq[0].s, cfg, sr)
    ends[-1] = refine_end(db, floor, seq[-1].e, cfg, sr)
    for i in range(n - 1):
        ends[i], starts[i + 1] = refine_gap(db, floor, seq[i].e, seq[i + 1].s, cfg, sr)
    for it, s_ref, e_ref in zip(seq, starts, ends):
        if s_ref >= e_ref:
            s_ref, e_ref = it.s, it.e
        if (s_ref == it.s) and (e_ref == it.e):
            stats["edges_unconfirmed"] += 1
        it.s_ref, it.e_ref = s_ref, e_ref
    for x, y in zip(seq, seq[1:]):
        if y.s_ref < x.e_ref:
            x.e_ref = max(y.s_ref, x.s_ref + 1)

    groups: list[list[SeqItem]] = []
    for it in seq:
        if groups and it.ok and groups[-1][-1].ok:
            last = groups[-1][-1]
            est = sum(x.unit.get("est_s") or 0.0 for x in groups[-1]) + (it.unit.get("est_s") or 0.0)
            fits = (it.e_ref - groups[-1][0].s_ref) / sr + 2 * cfg.pad_s <= cfg.max_clip_s  # measured, not estimated, length
            if fits and can_merge(last, it, it.s_ref - last.e_ref, est, cfg, sr):
                groups[-1].append(it)
                continue
        groups.append([it])

    cuts, quiet = [0], [True]
    for g, h in zip(groups, groups[1:]):
        lo, hi = g[-1].e_ref, h[0].s_ref
        c = choose_cut(db, floor, lo, hi, cfg, sr)
        cuts.append((lo + hi) // 2 if c is None else c)
        quiet.append(c is not None)
    cuts.append(len(pcm))

    clips: list[Clip] = []
    gap_flags = 2 * pad
    for k, g in enumerate(groups):
        if not g[0].ok:
            continue
        first, last = g[0], g[-1]
        a, b = clip_bounds(first.s_ref, last.e_ref, cuts[k], cuts[k + 1], pad)
        room = max(0, round(cfg.max_clip_s * sr) - (b - a))
        a0 = a
        a = quiet_start(pcm, a, cuts[k], quiet[k], floor, cfg.quiet_db, F, room)
        if _holds_word_mid(word_mids, a, a0, sr):
            a = a0
        b0 = b
        b = quiet_end(pcm, b, cuts[k + 1], quiet[k + 1] if k + 1 < len(quiet) else True, floor, cfg.quiet_db, F,
                      room - (a0 - a))
        if _holds_word_mid(word_mids, b0, b, sr):
            b = b0
        flags = set()
        gb = ga = None
        if k > 0:
            prev = groups[k - 1][-1]
            gap = first.s_ref - prev.e_ref
            if gap < gap_flags:
                flags.add("short_pause")
            if not quiet[k]:
                flags.add("no_quiet_cut")
            elif prev.ok and first.pos - prev.pos == 1:
                gb = gap / sr
        if k + 1 < len(groups):
            nxt = groups[k + 1][0]
            gap = nxt.s_ref - last.e_ref
            if gap < gap_flags:
                flags.add("short_pause")
            if not quiet[k + 1]:
                flags.add("no_quiet_cut")
            elif nxt.ok and nxt.pos - last.pos == 1:
                ga = gap / sr
        if (b - a) / sr > cfg.max_clip_s:
            flags.add("over_cap")
        clips.append(Clip(
            id=first.unit["id"], units=[x.unit["id"] for x in g], text=" ".join(x.unit.get("text") or "" for x in g),
            boundary=last.unit.get("boundary"), part=part, start_sample=a, end_sample=b,
            speech_start=first.s_ref, speech_end=last.e_ref, al_start=first.s, al_end=last.e,
            pad_start_s=(first.s_ref - a) / sr, pad_end_s=(b - last.e_ref) / sr, gap_before_s=gb, gap_after_s=ga,
            flags=sorted(flags), sha256=hashlib.sha256(pcm[a:b].tobytes()).hexdigest()))

    gaps: list[tuple[str, float]] = []
    group_of = {id(it): k for k, g in enumerate(groups) for it in g}
    for x, y in zip(seq, seq[1:]):
        if not (x.ok and y.ok and y.pos - x.pos == 1):
            continue
        kx, ky = group_of[id(x)], group_of[id(y)]
        if (kx == ky) or quiet[ky]:
            gaps.append((GAP_TYPES.get(x.unit.get("boundary") or "mid", "sentence"), (y.s_ref - x.e_ref) / sr))
    commas = [c for it in seq if it.ok for c in comma_gaps(it.unit, it.span)]

    check_clips(clips, len(pcm), to_sample(cfg.search_s, sr) + F)
    check_unique_hashes([(c.id, c.sha256) for c in clips])
    stats["clips"] = len(clips)
    stats["merged_clips"] = sum(1 for c in clips if len(c.units) > 1)
    stats["flags"] = dict(Counter(f for c in clips for f in c.flags))
    return PartPlan(clips, rejects, gaps, commas, stats)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def _dump_jsonl(rows: Iterable[dict]) -> str:
    return "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows)


def run_cut(work_dir: Path, parts: Sequence[str], cfg: CutConfig = CutConfig()) -> dict:
    work_dir = Path(work_dir)
    out = work_dir / "cut"
    clips_dir = out / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    stale = sorted({p.name[:-len(".cuts.jsonl")] for p in out.glob("*.cuts.jsonl")} - set(parts))
    if stale:
        raise CutError(f"cut/ holds the clips of {stale}, which are not in this call: report.json and pauses.json describe "
                       f"the whole run, so pass every part of the run in one call or delete the files of the others")
    all_gaps: list = []
    all_commas: list = []
    report: dict = {"parts": {}}
    for part in parts:
        t0 = time.perf_counter()
        wav = work_dir / "audio" / f"{part}.24k.wav"
        info = sf.info(str(wav))
        if info.channels != 1 or info.samplerate != 24000:
            raise CutError(f"{part}: {wav.name} must be mono 24000 Hz, got {info.channels} channel(s) at {info.samplerate} Hz")
        pcm, sr = sf.read(str(wav), dtype="int16")
        units = _read_jsonl(work_dir / "text" / f"{part}.units.jsonl")
        spans = _read_jsonl(work_dir / "align" / f"{part}.align.jsonl")
        plan = plan_part(part, pcm, sr, units, spans, cfg, _load_word_mids(work_dir / "align" / f"{part}.asr.json"))
        cuts_path = out / f"{part}.cuts.jsonl"
        if cuts_path.exists():
            for r in _read_jsonl(cuts_path):
                (clips_dir / f"{r['id']}.wav").unlink(missing_ok=True)
        for c in plan.clips:
            sf.write(str(clips_dir / f"{c.id}.wav"), pcm[c.start_sample:c.end_sample], sr, subtype="PCM_16")
        cuts_path.write_text(_dump_jsonl(c.record() for c in sorted(plan.clips, key=lambda c: c.start_sample)))
        (out / f"{part}.rejects.jsonl").write_text(_dump_jsonl(plan.rejects))
        all_gaps += plan.gaps
        all_commas += plan.commas
        report["parts"][part] = {**plan.stats, "seconds": round(time.perf_counter() - t0, 3)}
    check_unique_hashes((r["id"], r["sha256"]) for p in sorted(out.glob("*.cuts.jsonl")) for r in _read_jsonl(p))
    pauses = {**pause_stats(all_gaps, all_commas), "frame_s": cfg.frame_s, "pad_s": cfg.pad_s}
    (out / "pauses.json").write_text(json.dumps(pauses, sort_keys=True, indent=2) + "\n")
    (out / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    lines = ["# Cut report", "", "| part | units | ok | clips | merged | rejects | flags | floor_db | edges_unconfirmed | seconds |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for part, s in report["parts"].items():
        lines.append(f"| {part} | {s['units']} | {s['ok']} | {s['clips']} | {s['merged_clips']} | {s['rejects']} | {s['flags']} "
                     f"| {s['floor_db']} | {s['edges_unconfirmed']} | {s['seconds']} |")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return report
