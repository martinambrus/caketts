"""Gold-set scorer of the corpus pipeline (design D6).

Compares a run's cuts (cut/<part>.cuts.jsonl) and aligner spans (align/<part>.align.jsonl) with hand-verified speech spans.
Reads files only. Times in seconds, errors in ms (signed e = ours - gold).
MATCH: IoU = overlap / union of two intervals; all gold/candidate pairs with IoU >= match_iou are taken in descending IoU
(ties: lower gold index, then lower candidate index), each gold span and each candidate used once. Candidates are the clips'
refined speech intervals, and separately the aligner units with status "ok" and numeric times.
UNSAFE CUT: a clip edge t is unsafe for a gold span (gs, ge) iff gs < t < ge (strict); the rate is unsafe spans / gold spans. A gold
span that two or more of our clips overlap may hold a pause of its own (an old clip of several sentences): the rate over the spans
that exactly one clip overlaps (unsafe_cut_rate_single_clip) leaves those out.
EDGE ERROR: of matched pairs; the end error only for spans with end_reliable.
PAUSE CENTRE: our pauses are [refined end of clip k, refined start of clip k+1] where clip k has a measured gap_after_s; a gold pause
goes to the pause with the largest positive overlap (ties: earlier); error = centre(ours) - centre(gold).
LOST SPEECH: speech-level audio of a gold span that no clip covers. Frame levels follow the cut stage's speech test (10 ms frames,
dB of the mean square, speech-level iff strictly above the part's speech threshold from cut/report.json, or its noise floor plus
speech_db when given). A frame is judged for a span iff it lies wholly inside the span (a partly inside edge frame never is); it is
lost iff speech-level and not wholly inside the union of the clips (clips that touch count as one). A span that no clip overlaps is not
judged (its audio belongs to rejected units). Lost frames are split into head (before the first overlapping clip), tail (after the
last) and inner (between). A span with end_reliable false ends at the old cut: verified speech is judged only up to that end, never
beyond. None without the part's audio, floor or threshold. Energy alone counts breaths and other non-lexical sounds in pauses, so
it never reaches 0 on real audio: when align/<part>.asr.json exists, a lost run is confirmed by words. LOST WORD: an ASR word (midpoint
= (start + end) / 2, part timeline) whose midpoint lies in the span [lo, hi) and within WORD_TOL_S (50 ms, rounded to whole frames) of
a lost frame of that span; a word counts once. A lost word whose midpoint lies in a rejected region (speech of units with status other
than ok, which the pipeline never cuts: from REJECTED_MARGIN_S after the end of the previous ok unit to REJECTED_MARGIN_S before the
start of the next, the part start and end open) is a rejected-unit word, reported apart; without the align file there are no regions.
word_spans = judged spans losing at least one word by the cut, rejected_word_spans likewise for rejected-unit words. PASS CRITERION "no speech cut off":
criterion_rate == 0, with criterion "words" (matched_word_rate, words lost by the cut only, the part has the ASR cache; in the total only when every part with
lost-speech data has it) else "energy" (matched_lost_speech_rate, the rate over IoU-matched judged spans); lost_speech_rate over all
judged spans is the stricter figure. The unsafe-cut rates stay as information.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import soundfile as sf

CUT_SR = 24000
WORD_TOL_S = 0.05
REJECTED_MARGIN_S = 0.08


@dataclass
class GoldConfig:
    file: Optional[str] = None
    audacity: Dict[str, str] = field(default_factory=dict)
    match_iou: float = 0.5


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def load_gold(path) -> dict:
    path = Path(path)
    g = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(g, dict):
        raise ValueError(f"{path}: top level must be an object")
    if not _is_int(g.get("sample_rate")) or g["sample_rate"] <= 0:
        raise ValueError(f"{path}: sample_rate must be an int > 0")
    if not isinstance(g.get("parts"), dict):
        raise ValueError(f"{path}: parts must be an object of lists")
    for part, spans in g["parts"].items():
        if not isinstance(spans, list):
            raise ValueError(f"{path}: parts[{part}] must be a list")
        for i, s in enumerate(spans):
            where = f"{path}: parts[{part}][{i}]"
            if not isinstance(s, dict):
                raise ValueError(f"{where} must be an object")
            if not (_is_int(s.get("speech_start")) and _is_int(s.get("speech_end")) and s["speech_start"] < s["speech_end"]):
                raise ValueError(f"{where}: speech_start < speech_end must be ints")
            s.setdefault("end_reliable", True)
            if not isinstance(s["end_reliable"], bool):
                raise ValueError(f"{where}: end_reliable must be a bool")
    g.setdefault("pauses", [])
    if not isinstance(g["pauses"], list):
        raise ValueError(f"{path}: pauses must be a list")
    for i, p in enumerate(g["pauses"]):
        if not (isinstance(p, dict) and isinstance(p.get("part"), str) and _is_int(p.get("start")) and _is_int(p.get("end"))
                and p["start"] < p["end"]):
            raise ValueError(f"{path}: pauses[{i}] needs a part and int start < end")
    return g


def gold_from_audacity(labels: Mapping[str, "str | Path"], sample_rate: int = 44100) -> dict:
    pauses = []
    for part, path in labels.items():
        rows = []
        for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip() or line.startswith("\\"):
                continue
            f = line.split("\t", 2)
            try:
                if len(f) < 2:
                    raise ValueError("needs start and end")
                a, b = float(f[0]), float(f[1])
                if not (math.isfinite(a) and math.isfinite(b)):
                    raise ValueError("time is not finite")
            except ValueError as e:
                raise ValueError(f"{path}:{n}: bad label line ({e})") from None
            if a < 0 or b <= a:
                raise ValueError(f"{path}:{n}: needs 0 <= start < end, got {a} {b}")
            rows.append((a, b, f[2].strip() if len(f) > 2 else ""))
        for a, b, label in sorted(rows, key=lambda r: r[0]):
            pauses.append({"part": part, "start": round(a * sample_rate), "end": round(b * sample_rate),
                           "start_s": a, "end_s": b, "duration_s": round(b - a, 6), "label": label})
    return {"sample_rate": sample_rate, "parts": {part: [] for part in labels}, "pauses": pauses}


def _read_jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _iou(a, b) -> float:
    inter = min(a[1], b[1]) - max(a[0], b[0])
    if inter <= 0:
        return 0.0
    return inter / ((a[1] - a[0]) + (b[1] - b[0]) - inter)


def _match(gold: Sequence, cands: Sequence, match_iou: float) -> list:
    pairs = []
    for gi, g in enumerate(gold):
        for ci, c in enumerate(cands):
            v = _iou(g, c)
            if v >= match_iou and v > 0:
                pairs.append((-v, gi, ci))
    used_g, used_c, out = set(), set(), []
    for _, gi, ci in sorted(pairs):
        if gi in used_g or ci in used_c:
            continue
        used_g.add(gi)
        used_c.add(ci)
        out.append((gi, ci))
    return out


def _stats(errors: Sequence[float]) -> dict:
    keys = ("median_ms", "p90_ms", "bias_ms", "within_20ms", "within_50ms", "within_100ms")
    if not errors:
        return {"n": 0, **{k: None for k in keys}}
    signed = np.round(np.asarray(errors, dtype=float) * 1000, 6)
    a = np.abs(signed)
    share = lambda x: round(float(np.mean(a <= x)), 6)
    return {"n": len(errors), "median_ms": round(float(np.median(a)), 3), "p90_ms": round(float(np.percentile(a, 90)), 3),
            "bias_ms": round(float(np.median(signed)), 3), "within_20ms": share(20), "within_50ms": share(50),
            "within_100ms": share(100)}


def _edge_errors(spans, cands, pairs) -> tuple:
    starts = [cands[ci][0] - spans[gi]["s"] for gi, ci in pairs]
    ends = [cands[ci][1] - spans[gi]["e"] for gi, ci in pairs if spans[gi]["reliable"]]
    return starts, ends


def _frame_db(pcm: np.ndarray, frame: int) -> np.ndarray:
    n = len(pcm) // frame
    out = np.empty(n)
    step = 4096
    for a in range(0, n, step):
        b = min(a + step, n)
        x = pcm[a * frame:b * frame].reshape(b - a, frame).astype(np.float64) / 32768
        out[a:b] = 10 * np.log10(np.mean(x ** 2, axis=1) + 1e-10)
    return out


def _rejected_regions(units) -> list:
    regions, prev_end, pending = [], None, False
    for u in units or []:
        s, e = u.get("start_s"), u.get("end_s")
        if not (u.get("status") == "ok" and isinstance(s, (int, float)) and isinstance(e, (int, float))):
            pending = True
            continue
        lo = 0.0 if prev_end is None else prev_end + REJECTED_MARGIN_S
        if pending and lo < s - REJECTED_MARGIN_S:
            regions.append((lo, s - REJECTED_MARGIN_S))
        prev_end, pending = e, False
    if pending:
        regions.append((0.0 if prev_end is None else prev_end + REJECTED_MARGIN_S, math.inf))
    return regions


def _words_near(word_mids: np.ndarray, lo: int, hi: int, frame: int, lost_frames: list, rejected=()) -> tuple:
    if not lost_frames:
        return 0, 0
    tol = round(WORD_TOL_S * CUT_SR / frame)
    lost = np.asarray(lost_frames)
    a, b = np.searchsorted(word_mids, [lo / CUT_SR, hi / CUT_SR], side="left")
    cut_n = rej_n = 0
    for m in word_mids[a:b]:
        fm = int(m * CUT_SR // frame)
        k = int(np.searchsorted(lost, fm - tol, side="left"))
        if k < len(lost) and lost[k] <= fm + tol:
            if any(r0 <= m < r1 for r0, r1 in rejected):
                rej_n += 1
            else:
                cut_n += 1
    return cut_n, rej_n


def _lost_speech(spans: Sequence, clips: Sequence, level_db: np.ndarray, frame: int, thr: float, floor_db: float = 0.0,
                 word_mids: Optional[np.ndarray] = None, rejected=()) -> list:
    merged = []
    for a, b in sorted(clips):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    out = []
    for lo, hi in spans:
        over = [c for c in merged if c[0] < hi and c[1] > lo]
        if not over:
            out.append(None)
            continue
        f0, f1 = -(-lo // frame), min(hi // frame, len(level_db))
        head = inner = tail = 0
        loudest = None
        lost_frames = []
        for f in range(f0, f1):
            if not level_db[f] > thr or any(c[0] <= f * frame and (f + 1) * frame <= c[1] for c in over):
                continue
            mid = (f + 0.5) * frame
            if mid < over[0][0]:
                head += 1
            elif mid >= over[-1][1]:
                tail += 1
            else:
                inner += 1
            loudest = level_db[f] if loudest is None else max(loudest, level_db[f])
            lost_frames.append(f)
        fs = frame / CUT_SR
        by_cut, in_rejected = (0, 0) if word_mids is None else _words_near(word_mids, lo, hi, frame, lost_frames, rejected)
        out.append({"head_s": round(head * fs, 6), "inner_s": round(inner * fs, 6), "tail_s": round(tail * fs, 6),
                    "lost_s": round((head + inner + tail) * fs, 6),
                    "max_over_floor_db": None if loudest is None else round(float(loudest) - floor_db, 1),
                    **({} if word_mids is None else {"words": by_cut, "rejected_words": in_rejected})})
    return out


def _load_word_mids(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    try:
        words = json.loads(path.read_text(encoding="utf-8"))["words"]
        if not isinstance(words, list):
            raise TypeError("words is not a list")
        mids = []
        for w in words:
            if not (isinstance(w, list) and len(w) == 3 and isinstance(w[0], str)):
                raise TypeError("a word is not [text, start, end]")
            a, b = w[1], w[2]
            if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in (a, b)):
                raise TypeError("a word time is not a finite number")
            mids.append((a + b) / 2)
    except (ValueError, TypeError, KeyError) as e:
        raise ValueError(f"{path}: not a long-form ASR cache ({e.__class__.__name__}: {e})") from None
    return np.sort(np.asarray(mids, dtype=float))


def _load_levels(work_dir: Path, part: str, speech_db: Optional[float]):
    wav, rep, pauses = work_dir / "audio" / f"{part}.24k.wav", work_dir / "cut" / "report.json", work_dir / "cut" / "pauses.json"
    if not (wav.exists() and rep.exists()):
        return None
    stats = (json.loads(rep.read_text(encoding="utf-8")).get("parts") or {}).get(part) or {}
    floor = stats.get("floor_db")
    thr = stats.get("speech_thr_db") if speech_db is None else (None if floor is None else floor + speech_db)
    if floor is None or thr is None:
        return None
    frame_s = json.loads(pauses.read_text(encoding="utf-8")).get("frame_s", 0.01) if pauses.exists() else 0.01
    frame = round(frame_s * CUT_SR)
    pcm, _ = sf.read(str(wav), dtype="int16")
    return _frame_db(pcm, frame), frame, floor, thr, _load_word_mids(work_dir / "align" / f"{part}.asr.json")


def _part_lost(g_spans: list, clips: list, pairs: list, levels, rejected=()) -> tuple:
    if levels is None:
        return None, []
    level_db, frame, floor, thr, word_mids = levels
    res = _lost_speech([(round(s["s"] * CUT_SR), round(s["e"] * CUT_SR)) for s in g_spans],
                       [(c["start_sample"], c["end_sample"]) for c in clips], level_db, frame, thr, floor, word_mids, rejected)
    matched = {gi for gi, _ in pairs}
    judged = [(gi, r) for gi, r in enumerate(res) if r is not None]
    bad = [(gi, r) for gi, r in judged if r["lost_s"] > 0]
    wordy = [(gi, r) for gi, r in judged if r.get("words", 0) > 0]
    rejy = [(gi, r) for gi, r in judged if r.get("rejected_words", 0) > 0]
    has = word_mids is not None
    raw = {"floor_db": floor, "speech_thr_db": thr, "covered_spans": len(judged), "lost_spans": len(bad),
           "lost_s": sum(r["lost_s"] for _, r in judged), "matched_covered": sum(gi in matched for gi, _ in judged),
           "matched_lost_spans": sum(gi in matched for gi, _ in bad),
           "word_spans": len(wordy) if has else None, "lost_words": sum(r["words"] for _, r in wordy) if has else None,
           "matched_word_spans": sum(gi in matched for gi, _ in wordy) if has else None,
           "rejected_word_spans": len(rejy) if has else None, "rejected_words": sum(r["rejected_words"] for _, r in rejy) if has else None,
           "matched_rejected_word_spans": sum(gi in matched for gi, _ in rejy) if has else None}
    return raw, [(gi, gi in matched, r) for gi, r in bad]


def _score_part(part: str, g_spans: list, g_pauses: list, clips: list, units, match_iou: float, levels=None) -> tuple:
    clips = sorted(clips, key=lambda c: c["start_sample"])
    refined = [(c["start_sample"] / CUT_SR + c["pad_start_s"], c["end_sample"] / CUT_SR - c["pad_end_s"]) for c in clips]
    pairs = _match([(s["s"], s["e"]) for s in g_spans], refined, match_iou)
    start_e, end_e = _edge_errors(g_spans, refined, pairs)
    unsafe, single = [], 0
    for s in g_spans:
        overlapping = sum(1 for c in clips if c["start_sample"] / CUT_SR < s["e"] and c["end_sample"] / CUT_SR > s["s"])
        single += overlapping == 1
        for c in clips:
            for edge, t in (("start", c["start_sample"] / CUT_SR), ("end", c["end_sample"] / CUT_SR)):
                if s["s"] < t < s["e"]:
                    unsafe.append({"part": part, "gold_clip": s["clip"], "edge": edge, "t_s": round(t, 6),
                                   "gold_start_s": round(s["s"], 6), "gold_end_s": round(s["e"], 6),
                                   "clips_overlapping": overlapping})
    aligner = None
    if units is not None:
        ok = [(u["start_s"], u["end_s"]) for u in units
              if u.get("status") == "ok" and isinstance(u.get("start_s"), (int, float)) and isinstance(u.get("end_s"), (int, float))]
        ap = _match([(s["s"], s["e"]) for s in g_spans], ok, match_iou)
        a_start, a_end = _edge_errors(g_spans, ok, ap)
        aligner = {"matched": len(ap), "start": a_start, "end": a_end}
    ours = [(refined[k][1], refined[k + 1][0]) for k in range(len(clips) - 1) if clips[k].get("gap_after_s") is not None]
    centre = []
    for p in g_pauses:
        best, best_ov = None, 0.0
        for o in ours:
            ov = min(o[1], p[1]) - max(o[0], p[0])
            if ov > best_ov:
                best, best_ov = o, ov
        if best is not None:
            centre.append((best[0] + best[1]) / 2 - (p[0] + p[1]) / 2)
    lost_raw, lost_rows = _part_lost(g_spans, clips, pairs, levels, _rejected_regions(units))
    return {"lost": lost_raw, "gold_spans": len(g_spans), "matched": len(pairs), "unsafe_edges": len(unsafe),
            "unsafe_spans": len({(u["gold_clip"], u["gold_start_s"]) for u in unsafe}), "single_clip_spans": single,
            "unsafe_spans_single_clip": len({(u["gold_clip"], u["gold_start_s"]) for u in unsafe if u["clips_overlapping"] == 1}),
            "start": start_e, "end": end_e,
            "aligner": aligner, "pauses": {"gold": len(g_pauses), "matched": len(centre), "centre": centre}}, unsafe, lost_rows


def _rate(n: int, d: int) -> Optional[float]:
    return round(n / d, 6) if d else None


def _finish_lost(x: Optional[dict]) -> Optional[dict]:
    if x is None:
        return None
    words = x["word_spans"] is not None
    energy = _rate(x["matched_lost_spans"], x["matched_covered"])
    matched_words = _rate(x["matched_word_spans"], x["matched_covered"]) if words else None
    return {**x, "lost_s": round(x["lost_s"], 3), "lost_speech_rate": _rate(x["lost_spans"], x["covered_spans"]),
            "matched_lost_speech_rate": energy,
            "word_rate": _rate(x["word_spans"], x["covered_spans"]) if words else None, "matched_word_rate": matched_words,
            "criterion": "words" if words else "energy", "criterion_rate": matched_words if words else energy}


def _finish(m: dict) -> dict:
    a = m["aligner"]
    return {"lost_speech": _finish_lost(m["lost"]), "gold_spans": m["gold_spans"], "matched": m["matched"], "unsafe_edges": m["unsafe_edges"],
            "unsafe_spans": m["unsafe_spans"],
            "unsafe_cut_rate": round(m["unsafe_spans"] / m["gold_spans"], 6) if m["gold_spans"] else None,
            "single_clip_spans": m["single_clip_spans"], "unsafe_spans_single_clip": m["unsafe_spans_single_clip"],
            "unsafe_cut_rate_single_clip": (round(m["unsafe_spans_single_clip"] / m["single_clip_spans"], 6)
                                            if m["single_clip_spans"] else None),
            "start_error": _stats(m["start"]), "end_error": _stats(m["end"]),
            "aligner": None if a is None else {"matched": a["matched"], "start_error": _stats(a["start"]),
                                               "end_error": _stats(a["end"])},
            "pauses": {"gold": m["pauses"]["gold"], "matched": m["pauses"]["matched"],
                       "centre_error": _stats(m["pauses"]["centre"])}}


def _pool_lost(ms: list) -> Optional[dict]:
    xs = [m["lost"] for m in ms if m["lost"] is not None]
    if not xs:
        return None
    keys = ("covered_spans", "lost_spans", "lost_s", "matched_covered", "matched_lost_spans")
    wkeys = ("word_spans", "lost_words", "matched_word_spans", "rejected_word_spans", "rejected_words", "matched_rejected_word_spans")
    words = all(x["word_spans"] is not None for x in xs)
    return {"floor_db": None, "speech_thr_db": None, **{k: sum(x[k] for x in xs) for k in keys},
            **{k: sum(x[k] for x in xs) if words else None for k in wkeys}}


def _pool(ms: list) -> dict:
    aligned = [m["aligner"] for m in ms if m["aligner"] is not None]
    return {"lost": _pool_lost(ms), "gold_spans": sum(m["gold_spans"] for m in ms), "matched": sum(m["matched"] for m in ms),
            "unsafe_edges": sum(m["unsafe_edges"] for m in ms), "unsafe_spans": sum(m["unsafe_spans"] for m in ms),
            "single_clip_spans": sum(m["single_clip_spans"] for m in ms),
            "unsafe_spans_single_clip": sum(m["unsafe_spans_single_clip"] for m in ms),
            "start": [e for m in ms for e in m["start"]], "end": [e for m in ms for e in m["end"]],
            "aligner": {"matched": sum(a["matched"] for a in aligned), "start": [e for a in aligned for e in a["start"]],
                        "end": [e for a in aligned for e in a["end"]]} if aligned else None,
            "pauses": {"gold": sum(m["pauses"]["gold"] for m in ms), "matched": sum(m["pauses"]["matched"] for m in ms),
                       "centre": [e for m in ms for e in m["pauses"]["centre"]]}}


def score_run(gold: dict, work_dir, parts: Sequence[str], match_iou: float = 0.5, speech_db: Optional[float] = None) -> dict:
    work_dir = Path(work_dir)
    sr = gold["sample_rate"]
    mentioned = set(gold["parts"]) | {p["part"] for p in gold.get("pauses", [])}
    scored = [p for p in parts if p in mentioned]
    if not scored:
        raise ValueError(f"the gold set mentions none of the parts {list(parts)}")
    raw, unsafe, lost = {}, [], []
    for part in scored:
        cuts = work_dir / "cut" / f"{part}.cuts.jsonl"
        if not cuts.exists():
            raise FileNotFoundError(str(cuts))
        al = work_dir / "align" / f"{part}.align.jsonl"
        g_spans = [{"s": s["speech_start"] / sr, "e": s["speech_end"] / sr, "clip": s.get("clip"),
                    "reliable": s.get("end_reliable", True)} for s in gold["parts"].get(part, [])]
        g_pauses = [(p["start"] / sr, p["end"] / sr) for p in gold.get("pauses", []) if p["part"] == part]
        raw[part], u, rows = _score_part(part, g_spans, g_pauses, _read_jsonl(cuts), _read_jsonl(al) if al.exists() else None,
                                         match_iou, _load_levels(work_dir, part, speech_db))
        unsafe += u
        lost += [{"part": part, "gold_clip": g_spans[gi]["clip"], "matched": matched, "end_reliable": g_spans[gi]["reliable"],
                  "lost_s": r["lost_s"], "head_s": r["head_s"], "inner_s": r["inner_s"], "tail_s": r["tail_s"],
                  "max_over_floor_db": r["max_over_floor_db"], "words": r.get("words"),
                  "rejected_words": r.get("rejected_words")} for gi, matched, r in rows]
    return {"sample_rate": sr, "match_iou": match_iou, "parts": {p: _finish(raw[p]) for p in scored},
            "total": _finish(_pool(list(raw.values()))), "parts_without_gold": [p for p in parts if p not in mentioned],
            "unsafe": unsafe, "lost": lost}


def _fmt(x) -> str:
    return "-" if x is None else str(x)


def _markdown(rep: dict) -> str:
    lines = ["# Gold-set report", "",
             f"Match rule: IoU = overlap / union; pairs with IoU >= {rep['match_iou']} taken greedily by descending IoU, one-to-one.",
             "Unsafe cut: a clip edge strictly inside a gold speech span. Errors are ours - gold, in ms.",
             "Lost speech: speech-level audio of a gold span (whole frames, up to its end) that no clip covers; pass = criterion rate 0 (words: matched spans losing an ASR word; "
             "energy: matched spans losing speech-level audio when no ASR cache exists). "
             "Lost words exclude the words of rejected units (never cut by design), which are listed apart.",
             "",
             "| part | gold | matched | unsafe spans | unsafe rate | unsafe rate single-clip | start med | start p90 | end med | end p90 "
             "| aligner matched | aligner start med | pauses gold | pauses matched | pause centre med "
             "| lost spans | lost rate | matched lost rate | lost s | lost words | rejected words | word rate | criterion rate |",
             "|" + "---|" * 23]
    for name, m in [*rep["parts"].items(), ("total", rep["total"])]:
        a, ls = m["aligner"], m["lost_speech"]
        row = [name, m["gold_spans"], m["matched"], m["unsafe_spans"], m["unsafe_cut_rate"], m["unsafe_cut_rate_single_clip"],
               m["start_error"]["median_ms"],
               m["start_error"]["p90_ms"], m["end_error"]["median_ms"], m["end_error"]["p90_ms"],
               None if a is None else a["matched"], None if a is None else a["start_error"]["median_ms"],
               m["pauses"]["gold"], m["pauses"]["matched"], m["pauses"]["centre_error"]["median_ms"],
               *([None] * 8 if ls is None else [ls["lost_spans"], ls["lost_speech_rate"], ls["matched_lost_speech_rate"], ls["lost_s"],
                                                ls["lost_words"], ls["rejected_words"], ls["word_rate"], ls["criterion_rate"]])]
        lines.append("| " + " | ".join(_fmt(x) for x in row) + " |")
    lines += ["", "## Unsafe cuts", ""]
    lines += ([f"- {u['part']} {u['gold_clip']} {u['edge']} edge at {u['t_s']} s inside gold {u['gold_start_s']}-{u['gold_end_s']} s"
               for u in rep["unsafe"]] or ["none"])
    lines += ["", "## Lost speech", ""]
    lines += ([f"- {x['part']} {x['gold_clip']} lost {x['lost_s']} s (head {x['head_s']}, inner {x['inner_s']}, tail {x['tail_s']}), "
               f"max {x['max_over_floor_db']} dB over floor, words {_fmt(x.get('words'))}, rejected-unit words {_fmt(x.get('rejected_words'))}, matched {x['matched']}, "
               f"end_reliable {x['end_reliable']}"
               for x in rep["lost"]] or ["none"])
    if rep["parts_without_gold"]:
        lines += ["", "Parts without gold: " + ", ".join(rep["parts_without_gold"])]
    return "\n".join(lines) + "\n"


def run_gold(work_dir, parts: Sequence[str], cfg: GoldConfig) -> dict:
    if (cfg.file is None) == (not cfg.audacity):
        raise ValueError("gold needs exactly one of file and audacity")
    work_dir = Path(work_dir)
    gold = load_gold(cfg.file) if cfg.file is not None else gold_from_audacity(cfg.audacity)
    rep = score_run(gold, work_dir, parts, cfg.match_iou)
    out = work_dir / "gold"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "report.md").write_text(_markdown(rep), encoding="utf-8")
    return rep
