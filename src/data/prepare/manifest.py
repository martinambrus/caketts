"""
Manifest stage of the corpus pipeline (design D5): cut clips + audit verdicts + aligner word times -> train, val, test and
held-out manifests of the clips a from-scratch model can trust, plus the list of every clip left out and why.

Inputs are text/, cut/, align/ and audit/ of the work directory; outputs are manifest/{train,val,test,heldout,dropped}.jsonl,
report.json, report.md, coverage.json and coverage.md. Reports hold counts, ids and numbers only, never clip text.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import soundfile as sf

from src.data.prepare.coverage import SPLITS, SplitConfig, check_split, clip_tags, coverage_markdown, coverage_table, make_split
from src.data.tempo import TempoLabel, tempo_from_alignment
from src.text.phonemizer import _SPAN_RE, _WORD_RE, CzechSlovakPhonemizer


@dataclass
class ManifestConfig:
    language: str = "cs"
    speaker: str = "narrator"
    sample_rate: int = 24000
    hop_s: float = 256 / 24000
    max_duration_s: float = 20.0
    min_duration_s: float = 0.5
    min_frames_per_token: float = 1.0
    min_pause_s: float = 0.12
    split: SplitConfig = field(default_factory=SplitConfig)
    exclude_flags: tuple = ("short_pause", "no_quiet_cut", "over_cap")
    exclude_tags: tuple = ("part_fragment", "over_cap")
    use_part_fragments: bool = False


def _word_spans(text: str) -> list:
    out, pos = [], 0
    for s in _SPAN_RE.finditer(text):
        out += [m.span() for m in _WORD_RE.finditer(text, pos, s.start())]
        out += [m.span() for m in _WORD_RE.finditer(text, s.start(2), s.end(2))]
        pos = s.end()
    return out + [m.span() for m in _WORD_RE.finditer(text, pos, len(text))]


def tempo_from_words(words: Sequence[tuple], n_phones: int, sr: int, min_pause_s: float) -> TempoLabel:
    if not words or n_phones <= 0:
        raise ValueError("tempo needs at least one word and one phone")
    spoken = sum(e - s for s, e in words)
    kinds = ["phone"] * n_phones + ["boundary"] * (len(words) - 1)
    durations = [spoken / n_phones] * n_phones + [max(0, b[0] - a[1]) for a, b in zip(words, words[1:])]
    return tempo_from_alignment(kinds, durations, 1 / sr, min_pause_s)


def clip_word_times(clip: dict, spans_by_id: dict, sr: int) -> Optional[list]:
    words, seen = [], set()
    for uid in clip["units"]:
        parent = uid.split(".")[0]
        if parent in seen:
            continue
        seen.add(parent)
        got = (spans_by_id.get(parent) or {}).get("words") or []
        if not all(isinstance(w.get("start"), (int, float)) and isinstance(w.get("end"), (int, float)) for w in got):
            return None
        if "." in uid:
            got = [w for w in got if clip["start_sample"] <= (w["start"] + w["end"]) / 2 * sr < clip["end_sample"]]
        words += [(round(w["start"] * sr), round(w["end"] * sr)) for w in got]
    return words if len(words) == len(_word_spans(clip["text"])) else None


def _read_jsonl(path: Path) -> list:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _dump_jsonl(rows) -> str:
    return "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows)


def _median(values: list) -> Optional[float]:
    return statistics.median(values) if values else None


def _markdown(report: dict) -> str:
    keys = ("units", "clips", "audit_dropped", "part_fragments", "g2p_errors", "text_overlap", *SPLITS)
    lines = ["# Manifest report", "", "## Waterfall", "", "| part | " + " | ".join(keys) + " | filtered |",
             "|---|" + "---|" * (len(keys) + 1)]
    for name, w in [*report["parts"].items(), ("total", report["total"])]:
        filtered = ", ".join(f"{k}: {v}" for k, v in w["filtered"].items())
        lines.append(f"| {name} | " + " | ".join(str(w[k]) for k in keys) + f" | {filtered} |")
    lines += ["", "## Hours and rate medians", "", "| split | hours | articulation_rate | pause_ratio |", "|---|---|---|---|"]
    for s in SPLITS:
        m = report["rate_medians"][s]
        lines.append(f"| {s} | {report['hours'][s]} | {m['articulation_rate']} | {m['pause_ratio']} |")
    lines += ["", f"Rows without word times: {report['rows_without_word_times']}", "",
              "Split checks: " + ", ".join(f"{k}: {v}" for k, v in report["split_checks"].items()), "",
              "Unmet quotas: " + (json.dumps(report["unmet"], sort_keys=True) if any(report["unmet"].values()) else "none"),
              "", "Held-out block: " + (json.dumps(report["heldout"], sort_keys=True) if report["heldout"] else "none"),
              "Held-out holes (clips of its span that are not in it): "
              + (json.dumps(report["heldout_holes"], sort_keys=True) if report["heldout_holes"] else "none"), ""]
    return "\n".join(lines)


def run_manifest(work_dir: Path, parts: Sequence[str], cfg: ManifestConfig = ManifestConfig(), g2p=None) -> dict:
    # Pass the same phonetic lexicon/configured phonemizer to vocabulary building and
    # TTSDataset. text.lexicon is a separate orthographic TSV, already applied to text.
    from src.data.dataset import filter_manifest

    t0 = time.perf_counter()
    work_dir = Path(work_dir).resolve()
    out = work_dir / "manifest"
    out.mkdir(parents=True, exist_ok=True)
    g2p = g2p if g2p is not None else CzechSlovakPhonemizer(cfg.language)
    sr = cfg.sample_rate

    rows, tokens_n, dropped, order, part_of = [], {}, {}, [], {}
    units_by_id, counts, rejected, reject_rows = {}, {}, {}, {}
    for part in parts:
        units = _read_jsonl(work_dir / "text" / f"{part}.units.jsonl")
        cuts = _read_jsonl(work_dir / "cut" / f"{part}.cuts.jsonl")
        spans = {a["id"]: a for a in _read_jsonl(work_dir / "align" / f"{part}.align.jsonl")}
        audit = {a["id"]: a for a in _read_jsonl(work_dir / "audit" / f"{part}.audit.jsonl")}
        reject_rows[part] = _read_jsonl(work_dir / "cut" / f"{part}.rejects.jsonl")
        rejected[part] = Counter(r["reason"] for r in reject_rows[part])
        counts[part] = {"units": len(units), "clips": len(cuts)}
        units_by_id.update({u["id"]: u for u in units})
        for c in cuts:
            a = audit.get(c["id"])
            if a is None:
                raise ValueError(f"{c['id']}: no audit row")
            if a["sha256"] != c["sha256"]:
                raise ValueError(f"{c['id']}: audit row is stale, its sha256 differs from the cut record")
            order.append(c["id"])
            part_of[c["id"]] = part
            if a["verdict"] == "drop":
                dropped[c["id"]] = {"id": c["id"], "stage": "audit", "reasons": a["reasons"]}
                continue
            tags = clip_tags(c, units_by_id)
            if ("part_fragment" in tags) and not cfg.use_part_fragments:
                dropped[c["id"]] = {"id": c["id"], "stage": "fragment", "reasons": ["part_fragment"]}
                continue
            wav = work_dir / "cut" / "clips" / f"{c['id']}.wav"
            pcm, rate = sf.read(str(wav), dtype="int16")
            if hashlib.sha256(pcm.tobytes()).hexdigest() != c["sha256"]:
                raise ValueError(f"{c['id']}: clip file differs from the cut record sha256")
            try:
                tokens = g2p.tokenize(c["text"], quote_before=c.get("quote_before", 0),
                                      quote_after=c.get("quote_after", 0))
            except ValueError:
                dropped[c["id"]] = {"id": c["id"], "stage": "g2p", "reasons": ["g2p_error"]}
                continue
            row = {"id": c["id"], "audio": str(wav), "text": c["text"], "lang": cfg.language, "speaker": cfg.speaker,
                   "session": part, "boundary": c["boundary"], "duration_s": len(pcm) / rate,
                   "gap_before_s": c["gap_before_s"], "gap_after_s": c["gap_after_s"],
                   "tags": tags, "audit": a["verdict"], "audit_reasons": a["reasons"],
                   "cer": a["metrics"].get("cer"), "units": c["units"], "flags": c["flags"],
                   "n_tokens": len(tokens), "sha256": c["sha256"],
                   **{key: c.get(key, 0) for key in ("open_before", "open_after", "quote_before", "quote_after")}}
            first = units_by_id[c["units"][0].split(".")[0]]
            row["paragraph"], row["chapter"] = first["paragraph"], first["chapter"]
            n_phones = sum(t.kind == "phone" for t in tokens)
            words = clip_word_times(c, spans, sr)
            if words and n_phones:
                label = tempo_from_words(words, n_phones, sr, cfg.min_pause_s)
                row["articulation_rate"], row["pause_ratio"] = label.articulation_rate, label.pause_ratio
            rows.append(row)
            tokens_n[c["id"]] = len(tokens)

    kept, fr = filter_manifest(rows, tokens_n, cfg.hop_s, cfg.max_duration_s, cfg.min_duration_s,
                               cfg.min_frames_per_token)
    for reason, ids in (("too_long", fr.too_long), ("too_short", fr.too_short),
                        ("impossible_alignment", fr.impossible_alignment)):
        for cid in ids:
            dropped[cid] = {"id": cid, "stage": "filter", "reasons": [reason]}

    items, last_unit = [], {}
    for r in kept:
        index = [units_by_id[u.split(".")[0]]["index"] for u in r["units"]]
        holes = max(0, min(index) - last_unit[r["session"]] - 1) if r["session"] in last_unit else 0
        last_unit[r["session"]] = max(index)
        items.append({"id": r["id"], "part": r["session"], "chapter": r["chapter"], "paragraph": r["paragraph"],
                      "tags": r["tags"], "duration_s": r["duration_s"], "sha256": r["sha256"], "text": r["text"],
                      "eligible": not ((set(r["flags"]) & set(cfg.exclude_flags)) or (set(r["tags"]) & set(cfg.exclude_tags))),
                      "holes": holes})
    res = make_split(items, cfg.split)
    check_split(items, res.assignment, {r["id"]: r["sha256"] for r in kept}, excluded=res.excluded)
    for e in res.excluded:
        dropped[e["id"]] = {"id": e["id"], "stage": "split", "reasons": [e["reason"]]}

    by_split = {s: [] for s in SPLITS}
    for r in kept:
        if r["id"] in res.assignment:
            r["split"] = res.assignment[r["id"]]
            by_split[r["split"]].append(r)
    for s in SPLITS:
        (out / f"{s}.jsonl").write_text(_dump_jsonl(by_split[s]), encoding="utf-8")
    (out / "dropped.jsonl").write_text(_dump_jsonl(dropped[i] for i in order if i in dropped), encoding="utf-8")

    held = by_split["heldout"]
    holes = None
    if held:
        at = [k for k, i in enumerate(order) if i in {r["id"] for r in held}]
        span = order[at[0]:at[-1] + 1]
        index = [units_by_id[u.split(".")[0]]["index"] for r in held for u in r["units"]]
        holes = {"span_clips": len(span), "dropped_clips": sum(1 for i in span if i in dropped),
                 "cut_rejected_units": sum(1 for r in reject_rows[held[0]["session"]]
                                           if min(index) < units_by_id.get(r["id"], {}).get("index", -1) < max(index))}

    def waterfall(ids: set, base: dict, rej: Counter) -> dict:
        stage = Counter((d["stage"], d["reasons"][0]) for i, d in dropped.items() if i in ids)
        return {**base, "cut_rejected_units": dict(sorted(rej.items())),
                "audit_dropped": sum(n for (s, _), n in stage.items() if s == "audit"),
                "g2p_errors": sum(n for (s, _), n in stage.items() if s == "g2p"),
                "part_fragments": sum(n for (s, _), n in stage.items() if s == "fragment"),
                "filtered": {k: stage[("filter", k)] for k in ("too_long", "too_short", "impossible_alignment")},
                "text_overlap": stage[("split", "text_overlap")],
                **{s: sum(1 for r in by_split[s] if r["id"] in ids) for s in SPLITS}}

    per_part = {p: waterfall({i for i, q in part_of.items() if q == p}, counts[p], rejected[p]) for p in parts}
    total = waterfall(set(order), {k: sum(c[k] for c in counts.values()) for k in ("units", "clips")},
                      sum(rejected.values(), Counter()))
    report = {
        "config": dataclasses.asdict(cfg), "parts": per_part, "total": total,
        "hours": {s: round(sum(r["duration_s"] for r in by_split[s]) / 3600, 6) for s in SPLITS},
        "rate_medians": {s: {k: _median([r[k] for r in by_split[s] if k in r]) for k in ("articulation_rate", "pause_ratio")}
                         for s in SPLITS},
        "rows_without_word_times": sum(1 for r in kept if "articulation_rate" not in r and r["id"] in res.assignment),
        "split_checks": {"no_group_in_two_splits": True, "no_shared_audio": True, "heldout_contiguous": True},
        "unmet": res.unmet, "skipped_tags": res.skipped_tags, "quotas": res.quotas, "heldout": res.heldout,
        "heldout_holes": holes,
        "seconds": round(time.perf_counter() - t0, 3)}
    (out / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (out / "report.md").write_text(_markdown(report), encoding="utf-8")
    table = coverage_table(items, res.assignment, cfg.split)
    (out / "coverage.json").write_text(json.dumps(table, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (out / "coverage.md").write_text(coverage_markdown(table), encoding="utf-8")
    return report
