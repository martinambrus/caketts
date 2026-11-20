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
from typing import Mapping, Optional, Sequence

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
    judges: tuple = ("parakeet-tdt-0.6b-v3", "canary-1b-v2")


def validate_judges(cfg: AuditConfig) -> None:
    valid = ", ".join(ASR_BACKENDS)
    names = list(cfg.judges)
    if not names:
        raise ValueError(f"judges is empty; valid names: {valid}")
    if len(set(names)) != len(names):
        raise ValueError(f"judges has a duplicate name: {names}; valid names: {valid}")
    unknown = [n for n in names if n not in ASR_BACKENDS]
    if unknown:
        raise ValueError(f"unknown judge name(s) {unknown}; valid names: {valid}")


@dataclass
class JudgeResult:
    verdicts: dict
    reasons: dict
    review_rows: list
    listening: list
    narrator: dict
    exceeded_by: dict = field(default_factory=dict)


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
    single = len(cfg.judges) == 1

    def read(r: dict, name: str) -> tuple:
        m = r["metrics"]
        if single:
            return m["cer"], m["edits"]
        j = m.get("judges", {}).get(name)
        if j is None:
            raise ValueError(f"{r['id']}: metrics hold no values of judge {name}")
        return j["cer"], j["edits"]

    medians, thresholds, n_pool = {}, {}, {}
    for name in cfg.judges:
        pairs = [(read(r, name)[0], r["metrics"]["ref_chars"]) for r in rows]
        long_cer = [c for c, n in pairs if c is not None and n >= cfg.median_min_chars]
        all_cer = [c for c, _ in pairs if c is not None]
        pool = long_cer or all_cer
        medians[name] = float(np.median(pool)) if pool else 0.0
        thresholds[name] = medians[name] + cfg.cer_margin
        n_pool[name] = len(pool)
    rates = [r["metrics"]["rate"] for r in rows if r["metrics"]["rate"] is not None]
    levels = [r["metrics"]["lufs"] for r in rows if r["metrics"]["lufs"] is not None]
    rate_band = _band(rates, cfg.rate_sigma_floor, cfg, True)
    level_band = _band(levels, cfg.level_sigma_floor, cfg, False)
    guard_saved = {name: 0 for name in cfg.judges}
    verdicts, reasons_by, review_rows, risks, exceeded_by = {}, {}, [], {}, {}
    for r in rows:
        m = r["metrics"]
        found = []
        seen, exceeding = [], []
        for name in cfg.judges:
            cer, edits = read(r, name)
            seen.append((name, cer, thresholds[name]))
            if cer is not None and cer > thresholds[name]:
                if (m["ref_chars"] > cfg.short_chars) or (edits >= cfg.min_edits):
                    exceeding.append((name, cer, thresholds[name]))
                else:
                    guard_saved[name] += 1
        exceeded_by[r["id"]] = [name for name, _, _ in exceeding]
        cer_excess = 0.0
        if exceeding:
            if single:
                found.append(("cer", "cer", exceeding[0][1], exceeding[0][2]))
            else:
                code = "cer" if len(exceeding) == len(cfg.judges) else "cer_disputed"
                found += [(code, f"cer:{name}", cer, thr) for name, cer, thr in seen]
                cer_excess = 1 + min(2.0, max((cer - thr) / max(abs(thr), 1e-9) for _, cer, thr in exceeding))
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
        reasons_by[r["id"]] = list(dict.fromkeys(f[0] for f in found))
        review_rows += [(r["id"], key, value, threshold, code) for code, key, value, threshold in found]
        risks[r["id"]] = round(sum(1.0 if key == "cut_flag"
                                   else 1 + min(2.0, abs(value - threshold) / max(abs(threshold), 1e-9))
                                   for _, key, value, threshold in found if not key.startswith("cer:")) + cer_excess, 6)
        if any(c in cfg.drop_reasons for c in reasons_by[r["id"]]):
            verdicts[r["id"]] = "drop"
        else:
            verdicts[r["id"]] = "review" if found else "ok"
    flagged = [(i, r["id"]) for i, r in enumerate(rows) if reasons_by[r["id"]]]
    flagged.sort(key=lambda t: (-risks[t[1]], t[0]))
    listening = [(rank, cid, risks[cid], reasons_by[cid]) for rank, (_, cid) in enumerate(flagged[:cfg.listen_n], 1)]
    narrator = {"judges": {name: {"median_cer": medians[name], "cer_threshold": thresholds[name],
                                  "n_median": n_pool[name], "guard_saved": guard_saved[name]} for name in cfg.judges},
                "rate": _band_summary(rate_band, len(rates), True), "level": _band_summary(level_band, len(levels), False)}
    if single:
        narrator.update(narrator["judges"][cfg.judges[0]])
    return JudgeResult(verdicts, reasons_by, review_rows, listening, narrator, exceeded_by)


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
    several = len(report["judges"]) > 1
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
    nar = report["narrator"]
    lines += ["", "| CER | " + " | ".join(n for n, _ in QUANTILES) + " | mean |", "|---|" + "---|" * 7]
    dists = report["cer_distribution"] if several else {"": report["cer_distribution"]}
    for judge_name, dist in dists.items():
        for kind in ("raw", "spelled"):
            d = dist[kind] or {}
            label = f"{kind} {judge_name}".strip()
            lines.append(f"| {label} | " + " | ".join(str(d.get(n)) for n in [*(n for n, _ in QUANTILES), "mean"]) + " |")
    lines += ["", "| quantity | value |", "|---|---|"]
    for judge_name, dist in dists.items():
        share, tag = dist["share_over_threshold"], f" {judge_name}" if several else ""
        lines += [f"| share over threshold, digits spelled{tag} | {share['with_spelling']} |",
                  f"| share over threshold, digits not spelled{tag} | {share['without_spelling']} |"]
    if several:
        for judge_name, n in nar["judges"].items():
            lines += [f"| median CER {judge_name} | {n['median_cer']} |", f"| CER threshold {judge_name} | {n['cer_threshold']} |",
                      f"| clips in the median {judge_name} | {n['n_median']} |",
                      f"| guard saved {judge_name} | {n['guard_saved']} |"]
        lines += [f"| CER exceeded by {who} | {k} |" for who, k in report["total"]["cer_exceeded_by"].items()]
    else:
        lines += [f"| median CER | {nar['median_cer']} |", f"| CER threshold | {nar['cer_threshold']} |",
                  f"| clips in the median | {nar['n_median']} |", f"| guard saved | {report['guard_saved']} |"]
    lines += [f"| rate median / sigma / low / high / n | {' / '.join(str(nar['rate'][k]) for k in ('median', 'sigma', 'low', 'high', 'n'))} |",
              f"| level median / sigma / low / high / n | {' / '.join(str(nar['level'][k]) for k in ('median', 'sigma', 'low', 'high', 'n'))} |",
              f"| seconds | {report['seconds']} |"]
    return "\n".join(lines) + "\n"


def _read_clip(work_dir: Path, cid: str) -> np.ndarray:
    pcm, sr = sf.read(str(work_dir / "cut" / "clips" / f"{cid}.wav"), dtype="int16")
    if sr != 24000:
        raise ValueError(f"{cid}: clip must be 24000 Hz, got {sr} Hz")
    return pcm


def _transcribe_judge(work_dir: Path, out: Path, parts: Sequence[str], cuts_by_part: dict, name: str, model_id: str,
                      runner, suffix: str, cfg: AuditConfig) -> tuple:
    texts, counts, owned = {}, {}, None
    try:
        for part in parts:
            cuts = cuts_by_part[part]
            cache_path = out / f"{part}.asr{suffix}.jsonl"
            cache = {r["id"]: r for r in _read_jsonl(cache_path)} if cache_path.exists() else {}
            missing = [c["id"] for c in cuts
                       if (cache.get(c["id"]) is None) or (cache[c["id"]]["sha256"] != c["sha256"])
                       or (cache[c["id"]]["model_id"] != model_id)]
            if missing and runner is None:
                runner = owned = AsrRunner(AsrConfig(backend=name, language=cfg.language))
            sha = {c["id"]: c["sha256"] for c in cuts}
            for b in range(0, len(missing), cfg.asr_chunk):
                chunk = missing[b:b + cfg.asr_chunk]
                got = runner.transcribe([_to_16k(_read_clip(work_dir, cid)) for cid in chunk])
                if len(got) != len(chunk):
                    raise ValueError(f"{part}: ASR returned {len(got)} texts for {len(chunk)} clips")
                for cid, text in zip(chunk, got):
                    cache[cid] = {"id": cid, "sha256": sha[cid], "model_id": model_id, "text": text}
            cache_path.write_text(_dump_jsonl(cache[c["id"]] for c in cuts))
            texts[part] = {c["id"]: cache[c["id"]]["text"] for c in cuts}
            counts[part] = {"new": len(missing), "cached": len(cuts) - len(missing)}
    finally:
        if owned is not None:
            owned.close()
    return texts, counts


def run_audit(work_dir: Path, parts: Sequence[str], cfg: AuditConfig = AuditConfig(), runner=None,
              aligner_model_id: Optional[str] = None) -> dict:
    t0 = time.perf_counter()
    validate_judges(cfg)
    several = len(cfg.judges) > 1
    if runner is None:
        runners = {}
    elif isinstance(runner, Mapping):
        runners = dict(runner)
    elif several:
        raise ValueError("several judges need a mapping judge name -> runner, not one runner")
    else:
        runners = {cfg.judges[0]: runner}
    model_ids = {n: runners[n].model_id if n in runners else ASR_BACKENDS[n].model_id for n in cfg.judges}
    if aligner_model_id is not None:
        for n in cfg.judges:
            if model_ids[n] == aligner_model_id:
                raise ValueError(f"the aligner and the judge must be different models, both are {aligner_model_id}")
    work_dir = Path(work_dir)
    out = work_dir / "audit"
    out.mkdir(parents=True, exist_ok=True)
    floors = _floors(work_dir, parts)
    cuts_by_part = {part: _read_jsonl(work_dir / "cut" / f"{part}.cuts.jsonl") for part in parts}
    texts, jcounts = {}, {}
    for n in cfg.judges:
        texts[n], jcounts[n] = _transcribe_judge(work_dir, out, parts, cuts_by_part, n, model_ids[n], runners.get(n),
                                                 f".{n}" if several else "", cfg)
    rows, sha_by_id, hyps, hyps_by_judge, counts = [], {}, {}, {}, {}
    for part in parts:
        cuts = cuts_by_part[part]
        audio_info = work_dir / "audio" / f"{part}.json"
        counts[part] = {"clips": len(cuts), "transcripts_new": sum(jcounts[n][part]["new"] for n in cfg.judges),
                        "transcripts_cached": sum(jcounts[n][part]["cached"] for n in cfg.judges),
                        "source_clipped_runs": json.loads(audio_info.read_text()).get("clipped_runs_in")
                        if audio_info.exists() else None}
        if several:
            counts[part]["by_judge"] = {n: jcounts[n][part] for n in cfg.judges}
        for c in cuts:
            pcm = _read_clip(work_dir, c["id"])
            stats = {n: cer_stats(c["text"], texts[n][part][c["id"]], cfg.language) for n in cfg.judges}
            best = min(cfg.judges, key=lambda n: stats[n]["cer"])
            metrics = {**stats[best], **audio_metrics(pcm, 24000, floors[part], c["pad_start_s"], c["pad_end_s"],
                                                      c["text"], cfg)}
            metrics = {k: _num(metrics[k]) for k in METRIC_KEYS}
            if several:
                metrics["judges"] = {n: {k: _num(stats[n][k]) for k in ("cer", "cer_raw", "edits")} for n in cfg.judges}
                hyps_by_judge[c["id"]] = {n: texts[n][part][c["id"]] for n in cfg.judges}
            rows.append({"id": c["id"], "part": part, "metrics": metrics, "flags": c["flags"]})
            hyps[c["id"]] = texts[best][part][c["id"]]
            sha_by_id[c["id"]] = c["sha256"]

    res = judge(rows, cfg)
    ids_by_part = {part: [r["id"] for r in rows if r["part"] == part] for part in parts}
    for part in parts:
        (out / f"{part}.audit.jsonl").write_text(_dump_jsonl(
            {"id": r["id"], "part": part, "sha256": sha_by_id[r["id"]], "verdict": res.verdicts[r["id"]],
             "reasons": res.reasons[r["id"]], "metrics": r["metrics"], "hyp": hyps[r["id"]],
             **({"hyps": hyps_by_judge[r["id"]]} if several else {})}
            for r in rows if r["part"] == part))
    (out / "review.csv").write_text(_csv_text(("clip", "metric", "value", "threshold", "reason"), res.review_rows))
    (out / "listening.csv").write_text(_csv_text(
        ("rank", "clip", "path", "risk", "reasons"),
        [(rank, cid, f"cut/clips/{cid}.wav", risk, ";".join(codes)) for rank, cid, risk, codes in res.listening]))

    per_part = {part: {**counts[part], **_stats(ids_by_part[part], res)} for part in parts}
    source = [c["source_clipped_runs"] for c in counts.values() if c["source_clipped_runs"] is not None]
    total = {**{k: sum(c[k] for c in counts.values()) for k in ("clips", "transcripts_new", "transcripts_cached")},
             "source_clipped_runs": sum(source) if source else None, **_stats([r["id"] for r in rows], res)}
    nar = res.narrator
    if several:
        total["by_judge"] = {n: {k: sum(jcounts[n][p][k] for p in parts) for k in ("new", "cached")} for n in cfg.judges}
        total["cer_exceeded_by"] = dict(sorted(Counter("+".join(res.exceeded_by[r["id"]]) or "none" for r in rows).items()))

    def distribution(values: Sequence, threshold: float) -> dict:
        spelled = [v["cer"] for v in values if v["cer"] is not None]
        raw = [v["cer_raw"] for v in values if v["cer_raw"] is not None]
        return {"raw": _distribution(raw), "spelled": _distribution(spelled),
                "share_over_threshold": {"with_spelling": _share_over(spelled, threshold),
                                         "without_spelling": _share_over(raw, threshold)}}

    if several:
        cer_distribution = {n: distribution([r["metrics"]["judges"][n] for r in rows],
                                            nar["judges"][n]["cer_threshold"]) for n in cfg.judges}
        guard_saved = {n: nar["judges"][n]["guard_saved"] for n in cfg.judges}
    else:
        cer_distribution = distribution([r["metrics"] for r in rows], nar["cer_threshold"])
        guard_saved = nar["guard_saved"]
    report = {"config": dataclasses.asdict(cfg), "asr_model": " + ".join(model_ids[n] for n in cfg.judges),
              "judges": dict(model_ids), "parts": per_part, "total": total, "cer_distribution": cer_distribution,
              "guard_saved": guard_saved, "narrator": nar, "seconds": round(time.perf_counter() - t0, 3)}
    (out / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    (out / "report.md").write_text(_markdown(report))
    return report
