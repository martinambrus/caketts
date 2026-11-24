"""Audit stage tests (Step 4.2, T4 U3). Synthetic seeded audio and sentences written for the tests only; no book text."""
import csv
import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.data.prepare import audit
from src.data.prepare.asr import ASR_BACKENDS
from src.data.prepare.audit import AuditConfig, audio_metrics, cer_stats, judge, run_audit, spell_digits
from src.data.prepare.cut import CutConfig
from src.text.num2words_cs import num2words as cs_words

SR = 24000
PARAKEET, CANARY = "parakeet-tdt-0.6b-v3", "canary-1b-v2"
CFG = AuditConfig(judges=(PARAKEET,))


def _rms_scale(x, dbfs):
    return x * (10 ** (dbfs / 20) / np.sqrt(np.mean(x ** 2)))


def _voice(n, rng, dbfs=-20):
    t = np.arange(n) / SR
    f0 = 140 + 20 * np.sin(2 * np.pi * 1.5 * t + rng.uniform(0, 2 * np.pi))
    phase = 2 * np.pi * np.cumsum(f0) / SR
    x = sum(np.sin(k * phase) / k for k in range(1, 12))
    x = x * (1 + 0.5 * np.sin(2 * np.pi * 4.5 * t + rng.uniform(0, 2 * np.pi)))
    ramp = int(0.015 * SR)
    w = np.ones(n)
    r = 0.5 * (1 - np.cos(np.pi * np.arange(ramp) / ramp))
    w[:ramp] = r
    w[n - ramp:] = r[::-1]
    return _rms_scale(x * w, dbfs)


def clip_pcm(voiced_s=2.0, pad_s=0.15, seed=0, dbfs=-20):
    rng = np.random.default_rng(seed)
    pad = int(pad_s * SR)
    n = int(voiced_s * SR)
    x = rng.standard_normal(n + 2 * pad) * 10 ** (-60 / 20)
    x[pad:pad + n] += _voice(n, rng, dbfs)
    return np.clip(np.round(x * 32768), -32768, 32767).astype(np.int16)


def metrics_of(pcm, text="abcdefghij " * 6, pad=0.15, floor=-60.0, cfg=CFG):
    return audio_metrics(pcm, SR, floor, pad, pad, text, cfg)


def mrow(**kw):
    m = {"ref_chars": 60, "cer": 0.0, "edits": 0, "rate": 15.0, "edge_start_db": 0.0, "edge_end_db": 0.0,
         "clipped_runs": 0, "lufs": -23.0}
    m.update(kw)
    return m


def row(i, flags=(), part="p", **kw):
    return {"id": f"{part}-{i:05d}", "part": part, "metrics": mrow(**kw), "flags": list(flags)}


# ---------------------------------------------------------------- 1 spell_digits

class TestSpellDigits:
    def test_digit_run_becomes_cardinal(self):
        assert spell_digits("v roce 2024 padl sníh", "cs") == f"v roce {cs_words(2024)} padl sníh"

    def test_space_separated_thousands_is_one_run(self):
        assert spell_digits("je to 1 000 metrů", "cs") == f"je to {cs_words(1000)} metrů"

    def test_no_digits_unchanged(self):
        assert spell_digits("žádná čísla tu nejsou", "cs") == "žádná čísla tu nejsou"

    def test_other_language_unchanged(self):
        assert spell_digits("in 2024 it snowed", "en") == "in 2024 it snowed"

    def test_sixteen_digit_run_unchanged(self):
        text = "kód 1234567890123456 platí"
        assert spell_digits(text, "cs") == text

    def test_slovak_uses_the_slovak_module(self):
        from src.text.num2words_sk import num2words as sk_words
        assert spell_digits("rok 2024", "sk") == f"rok {sk_words(2024)}"


# ---------------------------------------------------------------- 2 cer_stats

class TestCerStats:
    def test_identical(self):
        s = cer_stats("Ahoj, světe!", "ahoj světe", "cs")
        assert s == {"ref_chars": 10, "cer": 0.0, "cer_raw": 0.0, "edits": 0}

    def test_number_words_against_digits(self):
        s = cer_stats(f"v roce {cs_words(2024)} padl sníh", "V roce 2024 padl sníh.", "cs")
        assert s["cer_raw"] > 0.4
        assert s["cer"] == 0.0
        assert s["edits"] == 0

    def test_empty_hypothesis(self):
        s = cer_stats("nějaká věta", "", "cs")
        assert s["cer"] == 1.0
        assert s["edits"] == s["ref_chars"] == 11

    def test_empty_reference(self):
        s = cer_stats("...", "něco", "cs")
        assert s["cer"] == 0.0 and s["ref_chars"] == 0


# ---------------------------------------------------------------- 3 the CER rule

class TestCerRule:
    def rows(self):
        long_ok = [row(i) for i in range(10)]
        shorts_bad = [row(10 + i, ref_chars=12, cer=0.5, edits=6) for i in range(15)]
        one_edit = row(30, ref_chars=12, cer=1 / 12, edits=1)
        three_edits = row(31, ref_chars=12, cer=3 / 12, edits=3)
        long_bad = row(32, ref_chars=60, cer=0.2, edits=12)
        return long_ok + shorts_bad + [one_edit, three_edits, long_bad]

    def test_median_uses_clips_of_30_chars_and_more(self):
        res = judge(self.rows(), CFG)
        assert res.narrator["median_cer"] == 0.0
        assert res.narrator["n_median"] == 11
        assert res.narrator["cer_threshold"] == pytest.approx(0.05)

    def test_short_clip_guard(self):
        res = judge(self.rows(), CFG)
        assert "cer" not in res.reasons["p-00030"]
        assert "cer" in res.reasons["p-00031"]
        assert "cer" in res.reasons["p-00032"]
        assert res.narrator["guard_saved"] == 1

    def test_all_clips_when_none_is_long(self):
        res = judge([row(i, ref_chars=12, cer=0.1) for i in range(3)], CFG)
        assert res.narrator["n_median"] == 3


# ---------------------------------------------------------------- 4 audio_metrics

class TestAudioMetrics:
    def test_rate_within_five_percent(self):
        text = "abcdefghij " * 12
        m = metrics_of(clip_pcm(voiced_s=2.0), text=text)
        assert m["rate"] == pytest.approx(120 / 2.0, rel=0.05)
        assert m["voiced_s"] == pytest.approx(2.0, rel=0.05)
        assert m["duration_s"] == pytest.approx(2.3)

    def test_rate_none_when_voiced_short(self):
        assert metrics_of(clip_pcm(voiced_s=0.8))["rate"] is None

    def test_edge_burst(self):
        pcm = clip_pcm().astype(np.float64)
        pcm[:240] = np.random.default_rng(3).standard_normal(240) * 0.1 * 32768
        m = metrics_of(pcm.astype(np.int16))
        assert m["edge_start_db"] > 12
        assert abs(m["edge_end_db"]) < 3
        res = judge([{"id": "c", "part": "p", "metrics": {**mrow(), **m}, "flags": []}], CFG)
        assert res.reasons["c"] == ["edge_noise_start"]

    def test_edge_none_for_short_pad(self):
        m = audio_metrics(clip_pcm(), SR, -60.0, 0.009, 0.15, "abc", CFG)
        assert m["edge_start_db"] is None and m["edge_end_db"] is not None
        m = audio_metrics(clip_pcm(), SR, -60.0, 0.15, 0.009, "abc", CFG)
        assert m["edge_end_db"] is None and m["edge_start_db"] is not None

    def test_clipping_needs_a_run_of_three(self):
        full = int(0.995 * 32768)
        pcm = clip_pcm(dbfs=-30)
        a = pcm.copy()
        a[5000:5003] = full
        b = pcm.copy()
        b[5000:5002] = full
        c = pcm.copy()
        c[5000:5003] = -full
        assert metrics_of(a)["clipped_runs"] == 1
        assert metrics_of(b)["clipped_runs"] == 0
        assert metrics_of(c)["clipped_runs"] == 1

    def test_lufs(self):
        base = clip_pcm(voiced_s=1.5, dbfs=-32)
        loud = np.round(base.astype(np.float64) * 10 ** (9 / 20)).astype(np.int16)
        assert metrics_of(clip_pcm(voiced_s=0.5, dbfs=-32))["lufs"] is None
        assert metrics_of(loud)["lufs"] - metrics_of(base)["lufs"] == pytest.approx(9, abs=0.5)

    def test_silent_clip_has_no_loudness(self):
        assert metrics_of(np.zeros(2 * SR, dtype=np.int16))["lufs"] is None


# ---------------------------------------------------------------- 5 judge

class TestJudge:
    def test_rate_high_on_one_clip(self):
        rows = [row(i) for i in range(25)] + [row(25, rate=45.0)]
        res = judge(rows, CFG)
        assert {k: v for k, v in res.reasons.items() if v} == {"p-00025": ["rate_high"]}
        assert res.narrator["rate"]["n"] == 26
        assert res.narrator["rate"]["median"] == pytest.approx(15.0)

    def test_rate_low_and_level_codes(self):
        rows = [row(i) for i in range(25)] + [row(25, rate=5.0, lufs=-40.0), row(26, lufs=-10.0)]
        res = judge(rows, CFG)
        assert res.reasons["p-00025"] == ["level_low", "rate_low"]
        assert res.reasons["p-00026"] == ["level_high"]

    def test_under_min_clips_no_rate_or_level_reason(self):
        rows = [row(i) for i in range(18)] + [row(18, rate=45.0, lufs=-5.0)]
        res = judge(rows, CFG)
        assert all(not v for v in res.reasons.values())
        assert res.narrator["rate"]["n"] == 19

    def test_clipping_row(self):
        res = judge([row(0, clipped_runs=2)], CFG)
        assert res.review_rows == [("p-00000", "clipped_runs", 2, 0, "clipping")]

    def rows_of_verdicts(self):
        return [row(0, ref_chars=60, cer=0.5, edits=30), row(1, edge_end_db=20.0), row(2, flags=["no_quiet_cut"]),
                row(3), row(4, flags=["short_pause"]), row(5, flags=["over_cap"])]

    def test_verdict_table(self):
        res = judge(self.rows_of_verdicts(), CFG)
        assert res.verdicts == {"p-00000": "drop", "p-00001": "review", "p-00002": "review", "p-00003": "ok",
                                "p-00004": "ok", "p-00005": "review"}
        assert res.reasons["p-00004"] == []
        assert res.reasons["p-00001"] == ["edge_noise_end"]
        assert res.reasons["p-00002"] == ["no_quiet_cut"]

    def test_empty_drop_reasons_makes_cer_a_review(self):
        res = judge(self.rows_of_verdicts(), dataclasses.replace(CFG, drop_reasons=()))
        assert res.verdicts["p-00000"] == "review"

    def test_review_rows_carry_the_violated_threshold(self):
        res = judge(self.rows_of_verdicts(), CFG)
        assert ("p-00000", "cer", 0.5, pytest.approx(0.05), "cer") in res.review_rows
        assert ("p-00001", "edge_end_db", 20.0, 15.0, "edge_noise_end") in res.review_rows
        assert ("p-00002", "cut_flag", 1, 0, "no_quiet_cut") in res.review_rows
        assert not any(r[0] == "p-00004" for r in res.review_rows)

    def test_edge_level_defaults_to_the_cut_quiet_level(self):
        quiet = CutConfig().quiet_db
        res = judge([row(0, edge_start_db=quiet - 1.0), row(1, edge_start_db=quiet + 1.0)], CFG)
        assert res.reasons["p-00000"] == [] and res.reasons["p-00001"] == ["edge_noise_start"]
        assert ("p-00001", "edge_start_db", quiet + 1.0, quiet, "edge_noise_start") in res.review_rows

    def test_review_rows_in_clip_then_reason_order(self):
        rows = [row(0, edge_start_db=20.0, clipped_runs=1, flags=["over_cap"]), row(1, clipped_runs=1)]
        res = judge(rows, CFG)
        assert [(r[0], r[4]) for r in res.review_rows] == [("p-00000", "clipping"), ("p-00000", "edge_noise_start"),
                                                            ("p-00000", "over_cap"), ("p-00001", "clipping")]

    def listening_rows(self):
        return [row(0, edge_start_db=20.0), row(1, edge_start_db=100.0), row(2), row(3, edge_start_db=20.0),
                row(4, flags=["over_cap"]), row(5, edge_start_db=17.5)]

    def test_listening_sorted_by_risk_then_clip_order(self):
        res = judge(self.listening_rows(), dataclasses.replace(CFG, listen_n=4))
        assert [(r[0], r[1]) for r in res.listening] == [(1, "p-00001"), (2, "p-00000"), (3, "p-00003"), (4, "p-00005")]
        assert res.listening[0][2] == pytest.approx(3.0)
        assert res.listening[1][2] == pytest.approx(1 + 5 / 15)
        assert res.listening[0][3] == ["edge_noise_start"]

    def test_cut_flag_risk_is_exactly_one(self):
        res = judge([row(0, flags=["no_quiet_cut"])], CFG)
        assert res.listening == [(1, "p-00000", 1.0, ["no_quiet_cut"])]

    def test_listening_length_is_min_of_listen_n_and_flagged(self):
        res = judge(self.listening_rows(), CFG)
        assert len(res.listening) == 5
        res = judge(self.listening_rows(), dataclasses.replace(CFG, listen_n=2))
        assert len(res.listening) == 2

    def test_none_metrics_raise_no_reason(self):
        res = judge([row(0, cer=None, rate=None, edge_start_db=None, edge_end_db=None, lufs=None)], CFG)
        assert res.reasons["p-00000"] == [] and res.verdicts["p-00000"] == "ok"


# ---------------------------------------------------------------- 6 run_audit

TEXTS = [
    "Včera večer jsme šli pomalu domů.", "Na stole ležela stará mapa údolí.", "Potom zazvonil telefon v předsíni.",
    "V roce 2024 napadl první sníh brzy.", "Starý most nad řekou se zachvěl.", "Děti utíkaly po louce za psem.",
    "Ráno bylo chladné a úplně tiché.", "Otevřel dveře a pomalu vešel dovnitř.", "Lampa v okně svítila celou noc.",
    "Vlak přijel na nádraží s velkým zpožděním.", "Všichni se sešli u velkého kulatého stolu.",
    "Dlouho jsme mlčeli a poslouchali vítr.",
]
SPELLED = [t.replace("2024", cs_words(2024)) for t in TEXTS]
HYPS = [t.lower().replace(".", "") for t in TEXTS]
HYPS[3] = "V roce 2024 napadl první sníh brzy."
HYPS[7] = "Zítra odpoledne pojedeme autobusem do hor."
ALL_HYPS = HYPS


def build_work(root, spell=True):
    work = root / "work"
    (work / "cut" / "clips").mkdir(parents=True)
    report = {"parts": {"a1": {"floor_db": -60.0}, "a2": {"floor_db": -60.0}}}
    (work / "cut" / "report.json").write_text(json.dumps(report))
    k = 0
    for part in ("a1", "a2"):
        lines = []
        for i in range(6):
            cid = f"{part}-{i:05d}"
            pcm = clip_pcm(voiced_s=1.4 + 0.1 * i, seed=k)
            sf.write(str(work / "cut" / "clips" / f"{cid}.wav"), pcm, SR, subtype="PCM_16")
            text = (SPELLED if spell else TEXTS)[k]
            lines.append({"id": cid, "units": [cid], "text": text, "boundary": "mid", "part": part, "start_sample": 0,
                          "end_sample": len(pcm), "pad_start_s": 0.15, "pad_end_s": 0.15, "gap_before_s": None,
                          "gap_after_s": None, "flags": ["no_quiet_cut"] if k == 5 else [],
                          "sha256": hashlib.sha256(pcm.tobytes()).hexdigest()})
            k += 1
        (work / "cut" / f"{part}.cuts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
    (work / "audio").mkdir()
    (work / "audio" / "a1.json").write_text(json.dumps({"clipped_runs_in": 4}))
    return work


class Stub:
    model_id = "stub/asr"

    def __init__(self, texts=ALL_HYPS):
        self.texts = list(texts)
        self.calls = []

    def transcribe(self, clips):
        assert all(c.dtype == np.float32 and c.ndim == 1 for c in clips)
        self.calls.append([len(c) for c in clips])
        done = sum(len(c) for c in self.calls[:-1])
        return self.texts[done:done + len(clips)]


class Boom:
    model_id = "stub/asr"

    def transcribe(self, clips):
        raise AssertionError("transcribe must not run")


def read_rows(work, part):
    return [json.loads(x) for x in (work / "audit" / f"{part}.audit.jsonl").read_text().splitlines()]


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.reader(fh))


@pytest.fixture
def work(tmp_path):
    return build_work(tmp_path)


class TestRunAudit:
    CFG = AuditConfig(asr_chunk=5, judges=(PARAKEET,))

    def test_files_written_and_chunked(self, work):
        stub = Stub()
        report = run_audit(work, ["a1", "a2"], self.CFG, runner=stub)
        for name in ("a1.asr.jsonl", "a2.asr.jsonl", "a1.audit.jsonl", "a2.audit.jsonl", "review.csv", "listening.csv",
                     "report.json", "report.md"):
            assert (work / "audit" / name).is_file(), name
        assert [len(c) for c in stub.calls] == [5, 1, 5, 1]
        assert report["parts"]["a1"]["clips"] == 6 and report["total"]["clips"] == 12
        assert report["total"]["transcripts_new"] == 12 and report["total"]["transcripts_cached"] == 0
        assert report["parts"]["a1"]["source_clipped_runs"] == 4
        assert report["parts"]["a2"]["source_clipped_runs"] is None
        assert report["asr_model"] == "stub/asr"
        assert json.loads((work / "audit" / "report.json").read_text())["total"]["clips"] == 12

    def test_audit_rows_and_verdicts(self, work):
        run_audit(work, ["a1", "a2"], self.CFG, runner=Stub())
        rows = read_rows(work, "a1") + read_rows(work, "a2")
        by = {r["id"]: r for r in rows}
        assert [r["id"] for r in rows] == [f"{p}-{i:05d}" for p in ("a1", "a2") for i in range(6)]
        for part in ("a1", "a2"):
            for line in (work / "cut" / f"{part}.cuts.jsonl").read_text().splitlines():
                rec = json.loads(line)
                assert by[rec["id"]]["sha256"] == rec["sha256"]
        digits = by["a1-00003"]
        assert digits["metrics"]["cer"] == 0.0 and digits["metrics"]["cer_raw"] > 0.1
        assert digits["verdict"] == "ok" and digits["hyp"] == HYPS[3]
        wrong = by["a2-00001"]
        assert wrong["verdict"] == "drop" and "cer" in wrong["reasons"]
        assert by["a1-00005"]["verdict"] == "review" and by["a1-00005"]["reasons"] == ["no_quiet_cut"]
        assert set(by["a1-00000"]["metrics"]) == {"duration_s", "ref_chars", "cer", "cer_raw", "edits", "rate", "voiced_s",
                                                  "edge_start_db", "edge_end_db", "clipped_runs", "lufs"}

    def test_review_csv_equals_judge_rows(self, work):
        run_audit(work, ["a1", "a2"], self.CFG, runner=Stub())
        rows = read_rows(work, "a1") + read_rows(work, "a2")
        flags = {}
        for part in ("a1", "a2"):
            for line in (work / "cut" / f"{part}.cuts.jsonl").read_text().splitlines():
                r = json.loads(line)
                flags[r["id"]] = r["flags"]
        res = judge([{"id": r["id"], "part": r["part"], "metrics": r["metrics"], "flags": flags[r["id"]]} for r in rows],
                    self.CFG)
        got = read_csv(work / "audit" / "review.csv")
        assert got[0] == ["clip", "metric", "value", "threshold", "reason"]
        assert len(got) - 1 == len(res.review_rows) > 0
        for g, w in zip(got[1:], res.review_rows):
            assert (g[0], g[1], g[4]) == (w[0], w[1], w[4])
            assert float(g[2]) == pytest.approx(w[2], abs=1e-5) and float(g[3]) == pytest.approx(w[3], abs=1e-5)
        listen = read_csv(work / "audit" / "listening.csv")
        assert listen[0] == ["rank", "clip", "path", "risk", "reasons"]
        assert [r[1] for r in listen[1:]] == [r[1] for r in res.listening]
        assert listen[1][2] == f"cut/clips/{listen[1][1]}.wav"

    def test_reports_hold_no_text(self, work):
        run_audit(work, ["a1", "a2"], self.CFG, runner=Stub())
        blob = (work / "audit" / "report.json").read_text() + (work / "audit" / "report.md").read_text()
        for s in TEXTS + SPELLED + HYPS:
            assert s not in blob
            assert s.lower().replace(".", "") not in blob.lower()
        assert "napadl" not in blob and "autobusem" not in blob
        rep = json.loads((work / "audit" / "report.json").read_text())
        assert set(rep["cer_distribution"]) == {"raw", "spelled", "share_over_threshold"}
        assert set(rep["cer_distribution"]["spelled"]) == {"p10", "p25", "p50", "p75", "p90", "p95", "mean"}
        assert rep["cer_distribution"]["share_over_threshold"]["with_spelling"] < rep["cer_distribution"]["share_over_threshold"]["without_spelling"]
        assert "verdicts" in rep["total"] and "reasons" in rep["total"] and "narrator" in rep and "guard_saved" in rep

    def test_two_runs_are_byte_identical(self, tmp_path, work):
        other = build_work(tmp_path / "other")
        run_audit(work, ["a1", "a2"], self.CFG, runner=Stub())
        run_audit(other, ["a1", "a2"], self.CFG, runner=Stub())
        for name in ("a1.audit.jsonl", "a2.audit.jsonl", "a1.asr.jsonl", "review.csv", "listening.csv"):
            assert (work / "audit" / name).read_bytes() == (other / "audit" / name).read_bytes(), name

    def test_cached_run_needs_no_runner(self, work):
        run_audit(work, ["a1", "a2"], self.CFG, runner=Stub())
        before = {p.name: p.read_bytes() for p in (work / "audit").glob("*") if p.name not in ("report.json", "report.md")}
        report = run_audit(work, ["a1", "a2"], self.CFG, runner=Boom())
        assert report["total"]["transcripts_cached"] == 12 and report["total"]["transcripts_new"] == 0
        after = {p.name: p.read_bytes() for p in (work / "audit").glob("*") if p.name not in ("report.json", "report.md")}
        assert before == after

    def test_cache_cannot_cross_book_languages(self, work):
        import dataclasses

        run_audit(work, ["a1", "a2"], self.CFG, runner=Stub())
        report = run_audit(work, ["a1", "a2"], dataclasses.replace(self.CFG, language="sk"), runner=Stub())
        assert report["total"]["transcripts_new"] == 12
        assert report["total"]["transcripts_cached"] == 0

    @pytest.mark.parametrize("evidence", ["valid", "wrong_language", "missing_part", "changed_audio", "no_report"])
    def test_legacy_cache_requires_matching_producer_part_and_audio(self, work, evidence):
        import dataclasses

        run_audit(work, ["a1", "a2"], self.CFG, runner=Stub())
        for path in (work / "audit").glob("*.asr.jsonl"):
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            for row in rows:
                row.pop("provenance")
                if evidence == "changed_audio":
                    row["sha256"] = "old-pcm"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        path = work / "audit/report.json"
        old = json.loads(path.read_text())
        old.pop("asr_provenance")
        old["config"].pop("terms")
        if evidence == "wrong_language":
            old["config"]["language"] = "sk"
        elif evidence == "missing_part":
            old["parts"].pop("a2")
        path.write_text(json.dumps(old))
        if evidence == "no_report":
            path.unlink()
        # CER thresholds do not affect recognition or valid legacy reuse.
        cfg = dataclasses.replace(self.CFG, cer_margin=.19)
        runner = Boom() if evidence == "valid" else Stub()
        report = run_audit(work, ["a1", "a2"], cfg, runner=runner)
        assert report["transcripts_migrated"] == (12 if evidence == "valid" else 0)
        assert report["total"]["transcripts_new"] == (0 if evidence == "valid" else 12)
        row = json.loads((work / "audit/a1.asr.jsonl").read_text().splitlines()[0])
        assert row["provenance"]["book_language"] == "cs"
        if evidence == "valid":
            assert row["migrated_from"]["language_control"] == "automatic"

    def test_term_alias_comparison_is_diagnostic_and_does_not_change_raw_cache_or_gate(self, work, tmp_path):
        import dataclasses

        path = work / "cut/a1.cuts.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["text"] = "Zorim"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        stub = Stub()
        stub.texts[0] = "Sorim"
        run_audit(work, ["a1", "a2"], self.CFG, runner=stub)
        before = read_rows(work, "a1")[0]
        cache_before = (work / "audit/a1.asr.jsonl").read_bytes()
        policy = tmp_path / "terms.tsv"
        policy.write_text("canonical\talias\tlanguage\tkind\nZorim\tSorim\tcs\tname\n")
        cfg = dataclasses.replace(self.CFG, terms=str(policy))
        report = run_audit(work, ["a1", "a2"], cfg, runner=Boom())
        after = read_rows(work, "a1")[0]
        assert before["hyp"] == after["hyp"] == "Sorim"
        assert before["verdict"] == after["verdict"] and before["reasons"] == after["reasons"]
        assert before["metrics"]["cer"] == after["metrics"]["cer"] == 0.2
        assert after["metrics"]["term_comparison"][PARAKEET]["cer"] == 0
        assert report["terminology"]["diagnostic_only"] is True
        assert (work / "audit/a1.asr.jsonl").read_bytes() == cache_before

    def test_changed_sha256_is_transcribed_again(self, work):
        run_audit(work, ["a1", "a2"], self.CFG, runner=Stub())
        pcm = clip_pcm(voiced_s=1.7, seed=99)
        sf.write(str(work / "cut" / "clips" / "a1-00002.wav"), pcm, SR, subtype="PCM_16")
        path = work / "cut" / "a1.cuts.jsonl"
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        rows[2]["sha256"] = hashlib.sha256(pcm.tobytes()).hexdigest()
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        stub = Stub(["Nějaký nový text věty."])
        report = run_audit(work, ["a1", "a2"], self.CFG, runner=stub)
        assert len(stub.calls) == 1 and len(stub.calls[0]) == 1
        assert report["total"]["transcripts_new"] == 1 and report["total"]["transcripts_cached"] == 11
        assert read_rows(work, "a1")[2]["hyp"] == "Nějaký nový text věty."

    def test_other_model_id_is_transcribed_again(self, work):
        run_audit(work, ["a1", "a2"], self.CFG, runner=Stub())
        other = Stub()
        other.model_id = "stub/other"
        report = run_audit(work, ["a1", "a2"], self.CFG, runner=other)
        assert report["total"]["transcripts_new"] == 12

    def test_missing_floor_db_raises(self, work):
        (work / "cut" / "report.json").write_text(json.dumps({"parts": {"a1": {"floor_db": -60.0}, "a2": {}}}))
        with pytest.raises(ValueError, match="a2"):
            run_audit(work, ["a1", "a2"], self.CFG, runner=Stub())


# ---------------------------------------------------------------- 7 light import, 8 real model

def test_import_is_light():
    code = ("import sys, src.data.prepare.audit; "
            "bad = [m for m in ('torch', 'transformers') if m in sys.modules]; "
            "sys.exit(1 if bad else 0)")
    env = {**os.environ, "PYTHONPATH": os.getcwd()}
    assert subprocess.run([sys.executable, "-c", code], env=env).returncode == 0


def cached(repo_id: str) -> bool:
    from huggingface_hub import try_to_load_from_cache
    return isinstance(try_to_load_from_cache(repo_id, "config.json"), str)


@pytest.mark.skipif(not cached("nvidia/parakeet-tdt-0.6b-v3"), reason="parakeet not in the HF cache")
def test_real_parakeet(tmp_path):
    work = build_work(tmp_path)
    for part in ("a1", "a2"):
        path = work / "cut" / f"{part}.cuts.jsonl"
        keep = path.read_text().splitlines()[: 2 if part == "a1" else 1]
        path.write_text("".join(x + "\n" for x in keep))
    report = run_audit(work, ["a1", "a2"], AuditConfig(judges=(PARAKEET,)))
    assert report["total"]["clips"] == 3
    rows = read_rows(work, "a1") + read_rows(work, "a2")
    assert all(isinstance(r["hyp"], str) for r in rows)
    assert (work / "audit" / "report.md").is_file()


# ---------------------------------------------------------------- 9 two judges

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "audit_single_judge.json").read_text())


def jrow(i, ref_chars=60, flags=(), **per_judge):
    judges = {n: {"cer": c, "cer_raw": c, "edits": e} for n, (c, e) in per_judge.items()}
    return {"id": f"p-{i:05d}", "part": "p", "metrics": mrow(ref_chars=ref_chars, judges=judges), "flags": list(flags)}


def pool(**cers):
    return [jrow(i, **{n: (c, 0) for n, c in cers.items()}) for i in range(10)]


def cfg_of(*names, **kw):
    return AuditConfig(judges=tuple(names), **kw)


class TestOracleParity:
    @pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda c: c["name"])
    def test_one_judge_equals_the_captured_rule(self, case):
        conf = {**case["config"], "drop_reasons": tuple(case["config"]["drop_reasons"])}
        res = judge(FIXTURE["rows"], AuditConfig(judges=(PARAKEET,), **conf))
        want = case["expected"]
        assert res.verdicts == want["verdicts"]
        assert res.reasons == want["reasons"]
        assert [list(r) for r in res.review_rows] == want["review_rows"]
        assert [list(r) for r in res.listening] == want["listening"]
        for key in ("median_cer", "cer_threshold", "n_median", "guard_saved", "rate", "level"):
            assert res.narrator[key] == want["narrator"][key]

    def test_default_judges_are_two_different_backends(self):
        assert AuditConfig().judges == (PARAKEET, CANARY)


class TestTwoJudges:
    A, B, C = "A", "B", "C"

    def judged(self, clips, *names, **kw):
        return judge(pool(**{n: 0.0 for n in names}) + clips, cfg_of(*names, **kw))

    def test_both_exceed_drops_with_two_rows(self):
        res = self.judged([jrow(10, A=(0.5, 30), B=(0.4, 24))], "A", "B")
        assert res.reasons["p-00010"] == ["cer"] and res.verdicts["p-00010"] == "drop"
        rows = [r for r in res.review_rows if r[0] == "p-00010"]
        assert rows == [("p-00010", "cer:A", 0.5, pytest.approx(0.05), "cer"),
                        ("p-00010", "cer:B", 0.4, pytest.approx(0.05), "cer")]

    @pytest.mark.parametrize("a,b", [(0.5, 0.0), (0.0, 0.5)])
    def test_one_exceeds_is_disputed_review(self, a, b):
        res = self.judged([jrow(10, A=(a, 20), B=(b, 20))], "A", "B")
        assert res.reasons["p-00010"] == ["cer_disputed"] and res.verdicts["p-00010"] == "review"
        rows = [r for r in res.review_rows if r[0] == "p-00010"]
        assert [(r[1], r[2], r[4]) for r in rows] == [("cer:A", a, "cer_disputed"), ("cer:B", b, "cer_disputed")]
        assert (rows[0][2] > rows[0][3]) == (a > 0) and (rows[1][2] > rows[1][3]) == (b > 0)

    def test_neither_exceeds_is_ok(self):
        res = self.judged([jrow(10, A=(0.01, 1), B=(0.0, 0))], "A", "B")
        assert res.verdicts["p-00010"] == "ok" and res.reasons["p-00010"] == []
        assert not [r for r in res.review_rows if r[0] == "p-00010"]

    def test_three_judges_two_exceed_is_disputed(self):
        res = self.judged([jrow(10, A=(0.5, 20), B=(0.5, 20), C=(0.0, 0))], "A", "B", "C")
        assert res.reasons["p-00010"] == ["cer_disputed"]
        assert [r[1] for r in res.review_rows if r[0] == "p-00010"] == ["cer:A", "cer:B", "cer:C"]
        res = self.judged([jrow(10, A=(0.5, 20), B=(0.5, 20), C=(0.5, 20))], "A", "B", "C")
        assert res.reasons["p-00010"] == ["cer"] and res.verdicts["p-00010"] == "drop"

    def test_each_judge_has_its_own_median(self):
        rows = [jrow(i, A=(0.30, 0), B=(0.0, 0)) for i in range(10)] + [jrow(10, A=(0.25, 15), B=(0.25, 15))]
        res = judge(rows, cfg_of("A", "B"))
        nar = res.narrator["judges"]
        assert nar["A"]["median_cer"] == pytest.approx(0.30) and nar["B"]["median_cer"] == pytest.approx(0.0)
        assert nar["A"]["cer_threshold"] == pytest.approx(0.35) and nar["B"]["cer_threshold"] == pytest.approx(0.05)
        assert nar["A"]["n_median"] == nar["B"]["n_median"] == 11
        assert res.reasons["p-00010"] == ["cer_disputed"]
        assert "median_cer" not in res.narrator and "guard_saved" not in res.narrator

    @pytest.mark.parametrize("edits_a,edits_b", [(2, 4), (4, 2)])
    def test_short_clip_guard_is_per_judge(self, edits_a, edits_b):
        res = self.judged([jrow(10, ref_chars=12, A=(0.5, edits_a), B=(0.5, edits_b))], "A", "B")
        assert res.reasons["p-00010"] == ["cer_disputed"]
        saved = res.narrator["judges"]
        assert (saved["A"]["guard_saved"], saved["B"]["guard_saved"]) == ((1, 0) if edits_a == 2 else (0, 1))

    def test_exceeded_by_lists_the_judges_the_guard_did_not_save(self):
        clips = [jrow(10, ref_chars=12, A=(0.5, 4), B=(0.5, 2)), jrow(11, A=(0.5, 20), B=(0.5, 20)),
                 jrow(12, A=(0.0, 0), B=(0.5, 20)), jrow(13, A=(0.0, 0), B=(0.0, 0))]
        res = self.judged(clips, "A", "B")
        assert [res.exceeded_by[f"p-{i:05d}"] for i in (10, 11, 12, 13)] == [["A"], ["A", "B"], ["B"], []]

    def test_disputed_drops_only_when_listed(self):
        clips = [jrow(10, A=(0.5, 20), B=(0.0, 0))]
        res = self.judged(clips, "A", "B", drop_reasons=("cer", "cer_disputed"))
        assert res.verdicts["p-00010"] == "drop"

    def test_risk_counts_the_cer_reason_once(self):
        res = self.judged([jrow(10, A=(0.07, 5), B=(0.08, 6))], "A", "B")
        assert res.listening == [(1, "p-00010", pytest.approx(1 + 0.6), ["cer"])]

    def test_missing_judge_metrics_raise(self):
        rows = pool(A=0.0) + [jrow(10, A=(0.1, 5))]
        with pytest.raises(ValueError, match=r"p-00000.*B|B.*p-00000"):
            judge(rows, cfg_of("A", "B"))


class LogStub(Stub):
    def __init__(self, name, model_id, texts, log):
        super().__init__(texts)
        self.name, self.model_id, self.log = name, model_id, log

    def transcribe(self, clips):
        self.log.append(self.name)
        return super().transcribe(clips)


class BoomId(Boom):
    def __init__(self, model_id):
        self.model_id = model_id


def two_runners(log=None):
    log = [] if log is None else log
    hyps_a, hyps_b = list(HYPS), list(HYPS)
    hyps_a[9] = "Jiná věta úplně o ničem jiném."
    hyps_b[2] = "Dnes odpoledne pršelo nad městem."
    return {PARAKEET: LogStub(PARAKEET, "stub/a", hyps_a, log), CANARY: LogStub(CANARY, "stub/b", hyps_b, log)}, log


TWO = AuditConfig(asr_chunk=5, judges=(PARAKEET, CANARY))


class TestRunAuditTwoJudges:
    def test_two_runners_files_rows_and_order(self, work):
        runners, log = two_runners()
        report = run_audit(work, ["a1", "a2"], TWO, runner=runners)
        for part in ("a1", "a2"):
            for name in (PARAKEET, CANARY):
                assert (work / "audit" / f"{part}.asr.{name}.jsonl").is_file()
            assert not (work / "audit" / f"{part}.asr.jsonl").exists()
        assert log == [PARAKEET] * 4 + [CANARY] * 4
        by = {r["id"]: r for r in read_rows(work, "a1") + read_rows(work, "a2")}
        assert by["a2-00001"]["verdict"] == "drop" and by["a2-00001"]["reasons"] == ["cer"]
        assert by["a1-00002"]["verdict"] == "review" and by["a1-00002"]["reasons"] == ["cer_disputed"]
        assert by["a2-00003"]["verdict"] == "review" and by["a2-00003"]["reasons"] == ["cer_disputed"]
        assert by["a1-00002"]["hyp"] == HYPS[2] and by["a2-00003"]["hyp"] == HYPS[9]
        assert by["a1-00002"]["hyps"] == {PARAKEET: HYPS[2], CANARY: "Dnes odpoledne pršelo nad městem."}
        for r in by.values():
            assert set(r["metrics"]["judges"]) == {PARAKEET, CANARY}
            assert set(r["metrics"]["judges"][PARAKEET]) == {"cer", "cer_raw", "edits"}
            assert r["metrics"]["cer"] == min(j["cer"] for j in r["metrics"]["judges"].values())
        assert by["a1-00000"]["verdict"] == "ok"
        csv_rows = read_csv(work / "audit" / "review.csv")
        assert {(r[1], r[4]) for r in csv_rows if r[0] == "a1-00002"} == {(f"cer:{PARAKEET}", "cer_disputed"),
                                                                           (f"cer:{CANARY}", "cer_disputed")}
        assert report["judges"] == {PARAKEET: "stub/a", CANARY: "stub/b"}
        assert report["asr_model"] == "stub/a + stub/b"
        assert report["total"]["transcripts_new"] == 24 and report["total"]["transcripts_cached"] == 0
        assert report["total"]["by_judge"] == {PARAKEET: {"new": 12, "cached": 0}, CANARY: {"new": 12, "cached": 0}}
        assert report["parts"]["a1"]["by_judge"][CANARY] == {"new": 6, "cached": 0}
        assert set(report["guard_saved"]) == {PARAKEET, CANARY}
        assert set(report["cer_distribution"]) == {PARAKEET, CANARY}
        assert set(report["cer_distribution"][CANARY]) == {"raw", "spelled", "share_over_threshold"}
        assert set(report["narrator"]["judges"]) == {PARAKEET, CANARY}

    def test_cer_exceeded_by_counts(self, work):
        runners, _ = two_runners()
        report = run_audit(work, ["a1", "a2"], TWO, runner=runners)
        assert report["total"]["cer_exceeded_by"] == {"none": 9, PARAKEET: 1, CANARY: 1, f"{PARAKEET}+{CANARY}": 1}
        assert report["total"]["cer_exceeded_by"][f"{PARAKEET}+{CANARY}"] == report["total"]["reasons"]["cer"]["clips"]

    def test_reports_hold_no_text(self, work):
        runners, _ = two_runners()
        run_audit(work, ["a1", "a2"], TWO, runner=runners)
        blob = ((work / "audit" / "report.json").read_text() + (work / "audit" / "report.md").read_text()).lower()
        for s in TEXTS + SPELLED + HYPS + ["Jiná věta úplně o ničem jiném.", "Dnes odpoledne pršelo nad městem."]:
            assert s.lower().replace(".", "") not in blob
        assert "napadl" not in blob and "autobusem" not in blob and "pršelo" not in blob

    def test_rerun_needs_no_transcription(self, work):
        runners, _ = two_runners()
        run_audit(work, ["a1", "a2"], TWO, runner=runners)
        report = run_audit(work, ["a1", "a2"], TWO, runner={PARAKEET: BoomId("stub/a"), CANARY: BoomId("stub/b")})
        assert report["total"]["transcripts_new"] == 0 and report["total"]["transcripts_cached"] == 24

    def test_one_judge_model_change_retranscribes_only_that_judge(self, work):
        runners, _ = two_runners()
        run_audit(work, ["a1", "a2"], TWO, runner=runners)
        new_b = LogStub(CANARY, "stub/b2", HYPS, [])
        report = run_audit(work, ["a1", "a2"], TWO, runner={PARAKEET: BoomId("stub/a"), CANARY: new_b})
        assert report["total"]["by_judge"] == {PARAKEET: {"new": 0, "cached": 12}, CANARY: {"new": 12, "cached": 0}}
        assert len(new_b.calls) == 4

    def test_aligner_equal_to_a_judge_raises_before_transcribing(self, work):
        runners, log = two_runners()
        with pytest.raises(ValueError, match="must be different"):
            run_audit(work, ["a1", "a2"], TWO, runner=runners, aligner_model_id="stub/b")
        assert log == [] and not any(r.calls for r in runners.values())

    def test_aligner_equal_to_a_default_backend_raises_before_loading(self, work, monkeypatch):
        def load(*a, **k):
            raise AssertionError("a model was loaded")
        monkeypatch.setattr(audit, "AsrRunner", load)
        with pytest.raises(ValueError, match="must be different"):
            run_audit(work, ["a1", "a2"], AuditConfig(), aligner_model_id=ASR_BACKENDS[CANARY].model_id)

    def test_bare_runner_with_two_judges_raises(self, work):
        with pytest.raises(ValueError, match="mapping"):
            run_audit(work, ["a1", "a2"], TWO, runner=Stub())

    @pytest.mark.parametrize("judges", [(), (PARAKEET, PARAKEET), (PARAKEET, "nope")])
    def test_bad_judges_raise(self, work, judges):
        with pytest.raises(ValueError):
            run_audit(work, ["a1", "a2"], AuditConfig(judges=judges), runner=Stub())
        with pytest.raises(ValueError):
            audit.validate_judges(AuditConfig(judges=judges))

    def test_unknown_judge_message_lists_valid_names(self):
        with pytest.raises(ValueError) as e:
            audit.validate_judges(AuditConfig(judges=("nope",)))
        assert "nope" in str(e.value) and PARAKEET in str(e.value) and CANARY in str(e.value)

    def test_one_judge_bare_runner_keeps_the_file_shapes(self, work):
        report = run_audit(work, ["a1", "a2"], AuditConfig(asr_chunk=5, judges=(PARAKEET,)), runner=Stub())
        assert (work / "audit" / "a1.asr.jsonl").is_file()
        assert not list((work / "audit").glob("*.asr.*.jsonl"))
        for r in read_rows(work, "a1"):
            assert "hyps" not in r and "judges" not in r["metrics"]
        assert report["asr_model"] == "stub/asr" and isinstance(report["guard_saved"], int)
        assert "by_judge" not in report["total"] and "cer_exceeded_by" not in report["total"]
