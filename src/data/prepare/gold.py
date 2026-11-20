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
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

CUT_SR = 24000


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


def _score_part(part: str, g_spans: list, g_pauses: list, clips: list, units, match_iou: float) -> tuple:
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
    return {"gold_spans": len(g_spans), "matched": len(pairs), "unsafe_edges": len(unsafe),
            "unsafe_spans": len({(u["gold_clip"], u["gold_start_s"]) for u in unsafe}), "single_clip_spans": single,
            "unsafe_spans_single_clip": len({(u["gold_clip"], u["gold_start_s"]) for u in unsafe if u["clips_overlapping"] == 1}),
            "start": start_e, "end": end_e,
            "aligner": aligner, "pauses": {"gold": len(g_pauses), "matched": len(centre), "centre": centre}}, unsafe


def _finish(m: dict) -> dict:
    a = m["aligner"]
    return {"gold_spans": m["gold_spans"], "matched": m["matched"], "unsafe_edges": m["unsafe_edges"],
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


def _pool(ms: list) -> dict:
    aligned = [m["aligner"] for m in ms if m["aligner"] is not None]
    return {"gold_spans": sum(m["gold_spans"] for m in ms), "matched": sum(m["matched"] for m in ms),
            "unsafe_edges": sum(m["unsafe_edges"] for m in ms), "unsafe_spans": sum(m["unsafe_spans"] for m in ms),
            "single_clip_spans": sum(m["single_clip_spans"] for m in ms),
            "unsafe_spans_single_clip": sum(m["unsafe_spans_single_clip"] for m in ms),
            "start": [e for m in ms for e in m["start"]], "end": [e for m in ms for e in m["end"]],
            "aligner": {"matched": sum(a["matched"] for a in aligned), "start": [e for a in aligned for e in a["start"]],
                        "end": [e for a in aligned for e in a["end"]]} if aligned else None,
            "pauses": {"gold": sum(m["pauses"]["gold"] for m in ms), "matched": sum(m["pauses"]["matched"] for m in ms),
                       "centre": [e for m in ms for e in m["pauses"]["centre"]]}}


def score_run(gold: dict, work_dir, parts: Sequence[str], match_iou: float = 0.5) -> dict:
    work_dir = Path(work_dir)
    sr = gold["sample_rate"]
    mentioned = set(gold["parts"]) | {p["part"] for p in gold.get("pauses", [])}
    scored = [p for p in parts if p in mentioned]
    if not scored:
        raise ValueError(f"the gold set mentions none of the parts {list(parts)}")
    raw, unsafe = {}, []
    for part in scored:
        cuts = work_dir / "cut" / f"{part}.cuts.jsonl"
        if not cuts.exists():
            raise FileNotFoundError(str(cuts))
        al = work_dir / "align" / f"{part}.align.jsonl"
        g_spans = [{"s": s["speech_start"] / sr, "e": s["speech_end"] / sr, "clip": s.get("clip"),
                    "reliable": s.get("end_reliable", True)} for s in gold["parts"].get(part, [])]
        g_pauses = [(p["start"] / sr, p["end"] / sr) for p in gold.get("pauses", []) if p["part"] == part]
        raw[part], u = _score_part(part, g_spans, g_pauses, _read_jsonl(cuts), _read_jsonl(al) if al.exists() else None, match_iou)
        unsafe += u
    return {"sample_rate": sr, "match_iou": match_iou, "parts": {p: _finish(raw[p]) for p in scored},
            "total": _finish(_pool(list(raw.values()))), "parts_without_gold": [p for p in parts if p not in mentioned],
            "unsafe": unsafe}


def _fmt(x) -> str:
    return "-" if x is None else str(x)


def _markdown(rep: dict) -> str:
    lines = ["# Gold-set report", "",
             f"Match rule: IoU = overlap / union; pairs with IoU >= {rep['match_iou']} taken greedily by descending IoU, one-to-one.",
             "Unsafe cut: a clip edge strictly inside a gold speech span. Errors are ours - gold, in ms.", "",
             "| part | gold | matched | unsafe spans | unsafe rate | unsafe rate single-clip | start med | start p90 | end med | end p90 "
             "| aligner matched | aligner start med | pauses gold | pauses matched | pause centre med |",
             "|" + "---|" * 15]
    for name, m in [*rep["parts"].items(), ("total", rep["total"])]:
        a = m["aligner"]
        row = [name, m["gold_spans"], m["matched"], m["unsafe_spans"], m["unsafe_cut_rate"], m["unsafe_cut_rate_single_clip"],
               m["start_error"]["median_ms"],
               m["start_error"]["p90_ms"], m["end_error"]["median_ms"], m["end_error"]["p90_ms"],
               None if a is None else a["matched"], None if a is None else a["start_error"]["median_ms"],
               m["pauses"]["gold"], m["pauses"]["matched"], m["pauses"]["centre_error"]["median_ms"]]
        lines.append("| " + " | ".join(_fmt(x) for x in row) + " |")
    lines += ["", "## Unsafe cuts", ""]
    lines += ([f"- {u['part']} {u['gold_clip']} {u['edge']} edge at {u['t_s']} s inside gold {u['gold_start_s']}-{u['gold_end_s']} s"
               for u in rep["unsafe"]] or ["none"])
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
