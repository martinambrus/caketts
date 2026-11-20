"""
Audit stage of the corpus pipeline (design D5): every cut clip is judged against its text by an ASR transcript and by
audio metrics, so the manifest stage can drop clips whose text the narrator did not read and every other risk is listed.

Inputs are cut/<part>.cuts.jsonl, cut/clips/<id>.wav and cut/report.json (floor_db) of the work directory; outputs are
audit/<part>.asr.jsonl (transcript cache), audit/<part>.audit.jsonl, review.csv, listening.csv, report.json and report.md.
"""
from __future__ import annotations

import csv
import dataclasses
import io
import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

import jiwer

from src.data.prepare.asr import ASR_BACKENDS, AsrConfig, AsrRunner
from src.eval.asr_check import normalize_for_asr

CUT_FLAG_REASONS = ("no_quiet_cut", "over_cap")
DIGIT_RUN = re.compile(r"[0-9]+(?: [0-9]{3}(?![0-9]))*")
MAX_SPELLED_DIGITS = 15
METRIC_KEYS = ("duration_s", "ref_chars", "cer", "cer_raw", "edits", "rate", "voiced_s", "edge_start_db", "edge_end_db",
               "clipped_runs", "lufs")
QUANTILES = (("p10", 10), ("p25", 25), ("p50", 50), ("p75", 75), ("p90", 90), ("p95", 95))


@dataclass
class AuditConfig:
    language: str = "cs"
    asr_chunk: int = 32
    cer_margin: float = 0.05
    short_chars: int = 30
    min_edits: int = 3
    median_min_chars: int = 30
    rate_frame_s: float = 0.02
    rate_voiced_db: float = 35.0
    rate_min_voiced_s: float = 1.0
    rate_sigma_floor: float = 0.05
    level_min_s: float = 1.0
    level_sigma_floor: float = 0.5
    sigmas: float = 3.0
    min_clips: int = 20
    edge_frame_s: float = 0.01
    edge_db: float = 12.0
    clip_level: float = 0.98
    clip_run: int = 3
    listen_n: int = 50
    drop_reasons: tuple = ("cer",)


@dataclass
class JudgeResult:
    verdicts: dict
    reasons: dict
    review_rows: list
    listening: list
    narrator: dict


def spell_digits(text: str, language: str) -> str:
    if language == "cs":
        from src.text.num2words_cs import num2words
    elif language == "sk":
        from src.text.num2words_sk import num2words
    else:
        return text

    def spell(m: re.Match) -> str:
        digits = m.group(0).replace(" ", "")
        return m.group(0) if len(digits) > MAX_SPELLED_DIGITS else num2words(int(digits))

    return DIGIT_RUN.sub(spell, text)


def cer_stats(reference: str, hypothesis: str, language: str) -> dict:
    ref = normalize_for_asr(reference)
    hyp = normalize_for_asr(spell_digits(hypothesis, language))
    hyp_raw = normalize_for_asr(hypothesis)
    if not ref:
        return {"ref_chars": 0, "cer": 0.0, "cer_raw": 0.0, "edits": 0}
    out = jiwer.process_characters(ref, hyp) if hyp else None
    edits = len(ref) if out is None else out.substitutions + out.deletions + out.insertions
    return {"ref_chars": len(ref), "cer": float(jiwer.cer(ref, hyp)), "cer_raw": float(jiwer.cer(ref, hyp_raw)),
            "edits": int(edits)}


def frame_db(pcm: np.ndarray, frame: int) -> np.ndarray:
    n = len(pcm) // frame
    out = np.empty(n)
    step = 4096
    for a in range(0, n, step):
        b = min(a + step, n)
        x = pcm[a * frame:b * frame].reshape(b - a, frame).astype(np.float64) / 32768
        out[a:b] = 10 * np.log10(np.mean(x ** 2, axis=1) + 1e-10)
    return out


def _edge_db(frame_pcm: np.ndarray, pad_s: Optional[float], frame: int, floor_db: float) -> Optional[float]:
    if pad_s is None or pad_s < 0.01 or len(frame_pcm) < frame:
        return None
    return float(frame_db(frame_pcm, frame)[0] - floor_db)


def _clipped_runs(pcm: np.ndarray, level: float, run: int) -> int:
    hot = np.abs(pcm.astype(np.int32)) >= round(level * 32768)
    edges = np.diff(np.concatenate(([0], hot.astype(np.int8), [0])))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return int(np.sum((ends - starts) >= run))


def audio_metrics(pcm: np.ndarray, sr: int, floor_db: float, pad_start_s: Optional[float], pad_end_s: Optional[float],
                  text: str, cfg: AuditConfig) -> dict:
    duration = len(pcm) / sr
    letters = sum(c.isalpha() for c in normalize_for_asr(text))
    db = frame_db(pcm, max(1, round(cfg.rate_frame_s * sr)))
    voiced_s = rate = None
    if len(db):
        voiced_s = float(np.sum(db >= np.percentile(db, 99) - cfg.rate_voiced_db) * cfg.rate_frame_s)
        rate = letters / voiced_s if voiced_s >= cfg.rate_min_voiced_s else None
    edge = max(1, round(cfg.edge_frame_s * sr))
    lufs = None
    if duration >= cfg.level_min_s:
        import pyloudnorm

        value = pyloudnorm.Meter(sr).integrated_loudness(pcm.astype(np.float64) / 32768)
        lufs = float(value) if math.isfinite(value) else None
    return {"duration_s": duration, "rate": rate, "voiced_s": voiced_s,
            "edge_start_db": _edge_db(pcm[:edge], pad_start_s, edge, floor_db),
            "edge_end_db": _edge_db(pcm[-edge:], pad_end_s, edge, floor_db),
            "clipped_runs": _clipped_runs(pcm, cfg.clip_level, cfg.clip_run), "lufs": lufs}


def _band(values: list, floor: float, cfg: AuditConfig, log: bool) -> Optional[tuple]:
    if len(values) < cfg.min_clips:
        return None
    x = np.log(values) if log else np.asarray(values, dtype=np.float64)
    centre = float(np.median(x))
    sigma = max(1.4826 * float(np.median(np.abs(x - centre))), floor)
    return centre, sigma, centre - cfg.sigmas * sigma, centre + cfg.sigmas * sigma


def _band_summary(band: Optional[tuple], n: int, log: bool) -> dict:
    if band is None:
        return {"median": None, "sigma": None, "low": None, "high": None, "n": n}
    back = math.exp if log else float
    return {"median": back(band[0]), "sigma": band[1], "low": back(band[2]), "high": back(band[3]), "n": n}


def judge(rows: Sequence[dict], cfg: AuditConfig) -> JudgeResult:
    long_cer = [r["metrics"]["cer"] for r in rows
                if r["metrics"]["cer"] is not None and r["metrics"]["ref_chars"] >= cfg.median_min_chars]
    all_cer = [r["metrics"]["cer"] for r in rows if r["metrics"]["cer"] is not None]
    pool = long_cer or all_cer
    median_cer = float(np.median(pool)) if pool else 0.0
    cer_threshold = median_cer + cfg.cer_margin
    rates = [r["metrics"]["rate"] for r in rows if r["metrics"]["rate"] is not None]
    levels = [r["metrics"]["lufs"] for r in rows if r["metrics"]["lufs"] is not None]
    rate_band = _band(rates, cfg.rate_sigma_floor, cfg, True)
    level_band = _band(levels, cfg.level_sigma_floor, cfg, False)
    guard_saved = 0
    verdicts, reasons_by, review_rows, risks = {}, {}, [], {}
    for r in rows:
        m = r["metrics"]
        found = []
        if m["cer"] is not None and m["cer"] > cer_threshold:
            if (m["ref_chars"] > cfg.short_chars) or (m["edits"] >= cfg.min_edits):
                found.append(("cer", "cer", m["cer"], cer_threshold))
            else:
                guard_saved += 1
        if rate_band is not None and m["rate"] is not None:
            if math.log(m["rate"]) < rate_band[2]:
                found.append(("rate_low", "rate", m["rate"], math.exp(rate_band[2])))
            elif math.log(m["rate"]) > rate_band[3]:
                found.append(("rate_high", "rate", m["rate"], math.exp(rate_band[3])))
        if level_band is not None and m["lufs"] is not None:
            if m["lufs"] < level_band[2]:
                found.append(("level_low", "lufs", m["lufs"], level_band[2]))
            elif m["lufs"] > level_band[3]:
                found.append(("level_high", "lufs", m["lufs"], level_band[3]))
        for code, key in (("edge_noise_start", "edge_start_db"), ("edge_noise_end", "edge_end_db")):
            if m[key] is not None and m[key] > cfg.edge_db:
                found.append((code, key, m[key], cfg.edge_db))
        if m["clipped_runs"] and m["clipped_runs"] > 0:
            found.append(("clipping", "clipped_runs", m["clipped_runs"], 0))
        flags = set(r["flags"])
        for code in CUT_FLAG_REASONS:
            if code in flags:
                found.append((code, "cut_flag", 1, 0))
        found.sort(key=lambda f: f[0])
        reasons_by[r["id"]] = [f[0] for f in found]
        review_rows += [(r["id"], key, value, threshold, code) for code, key, value, threshold in found]
        risks[r["id"]] = round(sum(1.0 if key == "cut_flag"
                                   else 1 + min(2.0, abs(value - threshold) / max(abs(threshold), 1e-9))
                                   for _, key, value, threshold in found), 6)
        if any(c in cfg.drop_reasons for c in reasons_by[r["id"]]):
            verdicts[r["id"]] = "drop"
        else:
            verdicts[r["id"]] = "review" if found else "ok"
    flagged = [(i, r["id"]) for i, r in enumerate(rows) if reasons_by[r["id"]]]
    flagged.sort(key=lambda t: (-risks[t[1]], t[0]))
    listening = [(rank, cid, risks[cid], reasons_by[cid]) for rank, (_, cid) in enumerate(flagged[:cfg.listen_n], 1)]
    narrator = {"median_cer": median_cer, "cer_threshold": cer_threshold, "n_median": len(pool),
                "guard_saved": guard_saved, "rate": _band_summary(rate_band, len(rates), True),
                "level": _band_summary(level_band, len(levels), False)}
    return JudgeResult(verdicts, reasons_by, review_rows, listening, narrator)


def _read_jsonl(path: Path) -> list:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def _dump_jsonl(rows) -> str:
    return "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows)


def _num(v):
    return round(v, 6) if isinstance(v, float) else v


def _csv_text(header: Sequence[str], rows) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(header)
    writer.writerows([[_num(v) for v in r] for r in rows])
    return buf.getvalue()


def _floors(work_dir: Path, parts: Sequence[str]) -> dict:
    report = json.loads((work_dir / "cut" / "report.json").read_text())
    floors = {}
    for part in parts:
        floor = report.get("parts", {}).get(part, {}).get("floor_db")
        if floor is None:
            raise ValueError(f"{part}: cut/report.json has no floor_db for this part")
        floors[part] = float(floor)
    return floors


def _to_16k(pcm: np.ndarray) -> np.ndarray:
    return resample_poly((pcm / 32768).astype(np.float32), 2, 3).astype(np.float32)


def _stats(part_ids: list, res: JudgeResult) -> dict:
    n = len(part_ids)
    reason_counts = Counter(code for cid in part_ids for code in res.reasons[cid])
    return {"verdicts": dict(sorted(Counter(res.verdicts[cid] for cid in part_ids).items())),
            "reasons": {code: {"clips": k, "share": round(k / n, 6)} for code, k in sorted(reason_counts.items())}}


def _distribution(values: list) -> Optional[dict]:
    if not values:
        return None
    out = {name: round(float(np.percentile(values, q)), 6) for name, q in QUANTILES}
    out["mean"] = round(float(np.mean(values)), 6)
    return out


def _share_over(values: list, threshold: float) -> Optional[float]:
    return round(sum(v > threshold for v in values) / len(values), 6) if values else None


def _markdown(report: dict) -> str:
    lines = ["# Audit report", "", f"ASR model: {report['asr_model']}", "",
             "| part | clips | transcripts cached | transcripts new | ok | review | drop | source clipped runs |",
             "|---|---|---|---|---|---|---|---|"]
    for name, s in [*report["parts"].items(), ("total", report["total"])]:
        v = s["verdicts"]
        lines.append(f"| {name} | {s['clips']} | {s['transcripts_cached']} | {s['transcripts_new']} | {v.get('ok', 0)} "
                     f"| {v.get('review', 0)} | {v.get('drop', 0)} | {s['source_clipped_runs']} |")
    lines += ["", "| reason | clips | share |", "|---|---|---|"]
    for code, s in report["total"]["reasons"].items():
        lines.append(f"| {code} | {s['clips']} | {s['share']} |")
    dist = report["cer_distribution"]
    lines += ["", "| CER | " + " | ".join(n for n, _ in QUANTILES) + " | mean |", "|---|" + "---|" * 7]
    for kind in ("raw", "spelled"):
        d = dist[kind] or {}
        lines.append(f"| {kind} | " + " | ".join(str(d.get(n)) for n in [*(n for n, _ in QUANTILES), "mean"]) + " |")
    share = dist["share_over_threshold"]
    nar = report["narrator"]
    lines += ["", "| quantity | value |", "|---|---|",
              f"| share over threshold, digits spelled | {share['with_spelling']} |",
              f"| share over threshold, digits not spelled | {share['without_spelling']} |",
              f"| median CER | {nar['median_cer']} |", f"| CER threshold | {nar['cer_threshold']} |",
              f"| clips in the median | {nar['n_median']} |", f"| guard saved | {report['guard_saved']} |",
              f"| rate median / sigma / low / high / n | {' / '.join(str(nar['rate'][k]) for k in ('median', 'sigma', 'low', 'high', 'n'))} |",
              f"| level median / sigma / low / high / n | {' / '.join(str(nar['level'][k]) for k in ('median', 'sigma', 'low', 'high', 'n'))} |",
              f"| seconds | {report['seconds']} |"]
    return "\n".join(lines) + "\n"


def run_audit(work_dir: Path, parts: Sequence[str], cfg: AuditConfig = AuditConfig(), runner=None) -> dict:
    t0 = time.perf_counter()
    work_dir = Path(work_dir)
    out = work_dir / "audit"
    out.mkdir(parents=True, exist_ok=True)
    floors = _floors(work_dir, parts)
    model_id = runner.model_id if runner is not None else ASR_BACKENDS[AsrConfig().backend].model_id
    owned = None
    rows, cuts_by_id, hyps, sha_by_id, counts = [], {}, {}, {}, {}
    try:
        for part in parts:
            cuts = _read_jsonl(work_dir / "cut" / f"{part}.cuts.jsonl")
            cache_path = out / f"{part}.asr.jsonl"
            cache = {r["id"]: r for r in _read_jsonl(cache_path)} if cache_path.exists() else {}
            pcms, missing = {}, []
            for c in cuts:
                pcm, sr = sf.read(str(work_dir / "cut" / "clips" / f"{c['id']}.wav"), dtype="int16")
                if sr != 24000:
                    raise ValueError(f"{c['id']}: clip must be 24000 Hz, got {sr} Hz")
                pcms[c["id"]] = pcm
                hit = cache.get(c["id"])
                if (hit is None) or (hit["sha256"] != c["sha256"]) or (hit["model_id"] != model_id):
                    missing.append(c["id"])
            if missing and runner is None:
                runner = owned = AsrRunner(AsrConfig(language=cfg.language))
            for b in range(0, len(missing), cfg.asr_chunk):
                chunk = missing[b:b + cfg.asr_chunk]
                texts = runner.transcribe([_to_16k(pcms[cid]) for cid in chunk])
                if len(texts) != len(chunk):
                    raise ValueError(f"{part}: ASR returned {len(texts)} texts for {len(chunk)} clips")
                for cid, text in zip(chunk, texts):
                    sha = next(c["sha256"] for c in cuts if c["id"] == cid)
                    cache[cid] = {"id": cid, "sha256": sha, "model_id": model_id, "text": text}
            cache_path.write_text(_dump_jsonl(cache[c["id"]] for c in cuts))
            audio_info = work_dir / "audio" / f"{part}.json"
            counts[part] = {"clips": len(cuts), "transcripts_new": len(missing),
                            "transcripts_cached": len(cuts) - len(missing),
                            "source_clipped_runs": json.loads(audio_info.read_text()).get("clipped_runs_in")
                            if audio_info.exists() else None}
            for c in cuts:
                hyp = cache[c["id"]]["text"]
                metrics = {**cer_stats(c["text"], hyp, cfg.language),
                           **audio_metrics(pcms[c["id"]], 24000, floors[part], c["pad_start_s"], c["pad_end_s"],
                                           c["text"], cfg)}
                metrics = {k: _num(metrics[k]) for k in METRIC_KEYS}
                rows.append({"id": c["id"], "part": part, "metrics": metrics, "flags": c["flags"]})
                hyps[c["id"]] = hyp
                sha_by_id[c["id"]] = c["sha256"]
            pcms.clear()
    finally:
        if owned is not None:
            owned.close()

    res = judge(rows, cfg)
    ids_by_part = {part: [r["id"] for r in rows if r["part"] == part] for part in parts}
    for part in parts:
        (out / f"{part}.audit.jsonl").write_text(_dump_jsonl(
            {"id": r["id"], "part": part, "sha256": sha_by_id[r["id"]], "verdict": res.verdicts[r["id"]],
             "reasons": res.reasons[r["id"]], "metrics": r["metrics"], "hyp": hyps[r["id"]]}
            for r in rows if r["part"] == part))
    (out / "review.csv").write_text(_csv_text(("clip", "metric", "value", "threshold", "reason"), res.review_rows))
    (out / "listening.csv").write_text(_csv_text(
        ("rank", "clip", "path", "risk", "reasons"),
        [(rank, cid, f"cut/clips/{cid}.wav", risk, ";".join(codes)) for rank, cid, risk, codes in res.listening]))

    per_part = {part: {**counts[part], **_stats(ids_by_part[part], res)} for part in parts}
    source = [c["source_clipped_runs"] for c in counts.values() if c["source_clipped_runs"] is not None]
    total = {**{k: sum(c[k] for c in counts.values()) for k in ("clips", "transcripts_new", "transcripts_cached")},
             "source_clipped_runs": sum(source) if source else None, **_stats([r["id"] for r in rows], res)}
    spelled = [r["metrics"]["cer"] for r in rows if r["metrics"]["cer"] is not None]
    raw = [r["metrics"]["cer_raw"] for r in rows if r["metrics"]["cer_raw"] is not None]
    threshold = res.narrator["cer_threshold"]
    report = {"config": dataclasses.asdict(cfg), "asr_model": model_id, "parts": per_part, "total": total,
              "cer_distribution": {"raw": _distribution(raw), "spelled": _distribution(spelled),
                                   "share_over_threshold": {"with_spelling": _share_over(spelled, threshold),
                                                            "without_spelling": _share_over(raw, threshold)}},
              "guard_saved": res.narrator["guard_saved"], "narrator": res.narrator,
              "seconds": round(time.perf_counter() - t0, 3)}
    (out / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    (out / "report.md").write_text(_markdown(report))
    return report
