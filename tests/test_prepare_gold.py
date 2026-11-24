"""Gold-set scorer tests (src/data/prepare/gold.py). Synthetic data only; times in the unit tests use sample_rate 1000 so samples are ms."""
import hashlib
import json
import math

import numpy as np
import pytest
import soundfile as sf

from src.data.prepare import cut, gold
from src.data.prepare.cut import CutConfig
from src.data.prepare.gold import GoldConfig
from test_cut import make_part, make_units, stub_spans, sp, pz, write_work

SR = 24000


def write_cuts(root, part, edges, pad_start=0.1, pad_end=0.1, gaps=None, text="CLIPMARK"):
    """edges: [(refined_start_s, refined_end_s)]; gaps: gap_after_s per clip (default 0.5, last None)."""
    (root / "cut").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (rs, re) in enumerate(edges):
        g = gaps[i] if gaps is not None else (None if i == len(edges) - 1 else 0.5)
        rows.append({"id": f"{part}-c{i}", "units": [f"{part}-u{i}"], "text": text, "boundary": "mid", "part": part,
                     "start_sample": round((rs - pad_start) * SR), "end_sample": round((re + pad_end) * SR),
                     "pad_start_s": pad_start, "pad_end_s": pad_end, "gap_before_s": None, "gap_after_s": g,
                     "flags": [], "sha256": "0"})
    (root / "cut" / f"{part}.cuts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def write_align(root, part, spans):
    """spans: [(start_s, end_s, status)]."""
    (root / "align").mkdir(parents=True, exist_ok=True)
    rows = [{"id": f"{part}-u{i}", "start_s": s, "end_s": e, "words": [], "status": st} for i, (s, e, st) in enumerate(spans)]
    (root / "align" / f"{part}.align.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def mk_gold(spans, pauses=(), part="a", reliable=None):
    """spans in seconds -> gold dict at sample_rate 1000."""
    reliable = reliable or [True] * len(spans)
    return {"sample_rate": 1000,
            "parts": {part: [{"clip": f"T-{i}", "text": "GOLDMARK", "speech_start": round(s * 1000), "speech_end": round(e * 1000),
                              "end_reliable": r} for i, ((s, e), r) in enumerate(zip(spans, reliable))]},
            "pauses": [{"part": part, "start": round(s * 1000), "end": round(e * 1000)} for s, e in pauses]}


GOLD5 = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]


def refined(spans, d_start, d_end):
    return [(s + a / 1000, e + b / 1000) for (s, e), a, b in zip(spans, d_start, d_end)]


def score(root, g, iou=0.5, part="a"):
    return gold.score_run(g, root, [part], iou)


def test_exact_offsets_statistics(tmp_path):
    write_cuts(tmp_path, "a", refined(GOLD5, [10, -20, 30, 60, -110], [0, 5, -5, 40, 80]))
    m = score(tmp_path, mk_gold(GOLD5))["parts"]["a"]
    assert m["gold_spans"] == 5 and m["matched"] == 5
    assert m["start_error"] == {"n": 5, "median_ms": 30.0, "p90_ms": 90.0, "bias_ms": 10.0,
                                "within_20ms": 0.4, "within_50ms": 0.6, "within_100ms": 0.8}
    assert m["end_error"] == {"n": 5, "median_ms": 5.0, "p90_ms": 64.0, "bias_ms": 5.0,
                              "within_20ms": 0.6, "within_50ms": 0.8, "within_100ms": 1.0}
    assert m["unsafe_cut_rate"] == 0 and m["aligner"] is None


def test_end_reliable_false_drops_only_the_end_error(tmp_path):
    write_cuts(tmp_path, "a", refined(GOLD5[:3], [10, 10, 10], [500, 20, 20]))
    m = score(tmp_path, mk_gold(GOLD5[:3], reliable=[False, True, True]))["parts"]["a"]
    assert m["start_error"]["n"] == 3
    assert m["end_error"]["n"] == 2 and m["end_error"]["median_ms"] == 20.0


def test_unsafe_negative_control(tmp_path):
    g = mk_gold([(1, 2), (3, 4)])
    write_cuts(tmp_path, "a", [(0.9, 2.1), (3.1, 4.1)], pad_start=0.1, pad_end=0.1)
    good = score(tmp_path, g)
    assert good["parts"]["a"]["unsafe_cut_rate"] == 0 and good["unsafe"] == []
    write_cuts(tmp_path, "a", [(1.6, 2.5), (3.0, 4.0)])
    bad = score(tmp_path, g)
    assert bad["parts"]["a"]["unsafe_cut_rate"] == 0.5
    assert bad["parts"]["a"]["unsafe_edges"] == 1 and bad["parts"]["a"]["unsafe_spans"] == 1
    assert bad["unsafe"] == [{"part": "a", "gold_clip": "T-0", "edge": "start", "t_s": 1.5, "gold_start_s": 1.0, "gold_end_s": 2.0,
                              "clips_overlapping": 1}]
    write_cuts(tmp_path, "a", [(0.5, 1.5), (3.0, 4.0)])
    ends = score(tmp_path, g)["unsafe"]
    assert [(u["gold_clip"], u["edge"], u["t_s"]) for u in ends] == [("T-0", "end", 1.6)]


def test_single_clip_rate_leaves_out_spans_with_a_pause_of_their_own(tmp_path):
    g = mk_gold([(1, 2), (3, 6)])
    write_cuts(tmp_path, "a", [(1.1, 1.9), (3.0, 4.4), (4.8, 6.0)], pad_start=0.1, pad_end=0.1)
    m = score(tmp_path, g)
    assert [(u["gold_clip"], u["clips_overlapping"]) for u in m["unsafe"]] == [("T-1", 2), ("T-1", 2)]
    m = m["parts"]["a"]
    assert (m["unsafe_spans"], m["unsafe_cut_rate"]) == (1, 0.5)
    assert (m["single_clip_spans"], m["unsafe_spans_single_clip"], m["unsafe_cut_rate_single_clip"]) == (1, 0, 0.0)
    write_cuts(tmp_path, "a", [(1.6, 2.5), (3.0, 4.4), (4.8, 6.0)], pad_start=0.1, pad_end=0.1)
    t = score(tmp_path, g)["total"]
    assert (t["unsafe_spans"], t["unsafe_cut_rate"]) == (2, 1.0)
    assert (t["single_clip_spans"], t["unsafe_spans_single_clip"], t["unsafe_cut_rate_single_clip"]) == (1, 1, 1.0)


def test_single_clip_rate_is_none_without_such_spans(tmp_path):
    write_cuts(tmp_path, "a", [(1, 2), (3, 4)], gaps=[0.5, None])
    m = score(tmp_path, mk_gold([], pauses=[(2, 3)]))["parts"]["a"]
    assert m["single_clip_spans"] == 0 and m["unsafe_cut_rate_single_clip"] is None


def test_edge_touching_gold_boundary_is_safe(tmp_path):
    write_cuts(tmp_path, "a", [(1.1, 1.9)], pad_start=0.1, pad_end=0.1)
    r = score(tmp_path, mk_gold([(1, 2)]))
    assert r["parts"]["a"]["unsafe_edges"] == 0 and r["parts"]["a"]["unsafe_cut_rate"] == 0


def test_unsafe_rate_none_without_spans(tmp_path):
    write_cuts(tmp_path, "a", [(1, 2), (3, 4)], gaps=[0.5, None])
    m = score(tmp_path, mk_gold([], pauses=[(2, 3)]))["parts"]["a"]
    assert m["gold_spans"] == 0 and m["unsafe_cut_rate"] is None and m["start_error"]["n"] == 0
    assert m["start_error"]["median_ms"] is None


def test_iou_clip_covering_two_spans_matches_neither(tmp_path):
    write_cuts(tmp_path, "a", [(1, 5)])
    assert score(tmp_path, mk_gold([(1, 2), (3, 4)]))["parts"]["a"]["matched"] == 0


def test_iou_exactly_at_threshold_matches(tmp_path):
    write_cuts(tmp_path, "a", [(1, 2)])
    g = mk_gold([(1, 3)])
    assert score(tmp_path, g, 0.5)["parts"]["a"]["matched"] == 1
    assert score(tmp_path, g, 0.51)["parts"]["a"]["matched"] == 0


def test_iou_is_one_to_one(tmp_path):
    write_cuts(tmp_path, "a", [(1.2, 3.2)])
    assert score(tmp_path, mk_gold([(1, 3), (1.5, 3.5)]))["parts"]["a"]["matched"] == 1
    write_cuts(tmp_path, "a", [(1, 2.8), (1.5, 3)])
    m = score(tmp_path, mk_gold([(1, 3)]))["parts"]["a"]
    assert m["matched"] == 1 and m["start_error"]["median_ms"] == 0.0


def test_iou_greedy_takes_the_highest_pair_first(tmp_path):
    write_cuts(tmp_path, "a", [(1.0, 3.0), (1.1, 3.1)])
    m = score(tmp_path, mk_gold([(1.1, 3.1)]))["parts"]["a"]
    assert m["matched"] == 1 and m["start_error"]["median_ms"] == 0.0


def test_pauses_centre_unmatched_and_null_gap(tmp_path):
    write_cuts(tmp_path, "a", [(1, 2), (3, 4), (5, 6)], pad_start=0.2, pad_end=0.05, gaps=[1.0, None, None])
    g = mk_gold([], pauses=[(2.1, 3.1), (4.0, 5.0)])
    p = score(tmp_path, g)["parts"]["a"]["pauses"]
    assert p["gold"] == 2 and p["matched"] == 1
    assert p["centre_error"]["n"] == 1 and p["centre_error"]["bias_ms"] == -100.0 and p["centre_error"]["median_ms"] == 100.0


def test_pause_goes_to_largest_overlap(tmp_path):
    write_cuts(tmp_path, "a", [(1, 2), (2.5, 3), (4, 5)], gaps=[0.5, 1.0, None])
    p = score(tmp_path, mk_gold([], pauses=[(2.4, 3.6)]))["parts"]["a"]["pauses"]
    assert p["matched"] == 1 and p["centre_error"]["bias_ms"] == pytest.approx(500.0)


def test_aligner_metrics_and_candidates(tmp_path):
    write_cuts(tmp_path, "a", refined(GOLD5[:3], [0, 0, 0], [0, 0, 0]))
    g = mk_gold(GOLD5[:3])
    write_align(tmp_path, "a", [(1.04, 1.945, "ok"), (3.0, 4.0, "low_conf"), (None, None, "unmatched")])
    al = score(tmp_path, g)["parts"]["a"]["aligner"]
    assert al["matched"] == 1
    assert al["start_error"]["n"] == 1 and al["start_error"]["bias_ms"] == 40.0
    assert al["end_error"]["n"] == 1 and al["end_error"]["bias_ms"] == -55.0 and al["end_error"]["median_ms"] == 55.0
    (tmp_path / "align" / "a.align.jsonl").unlink()
    assert score(tmp_path, g)["parts"]["a"]["aligner"] is None


def write_labels(path, text):
    path.write_bytes(text.encode())
    return path


def test_audacity_parsing(tmp_path):
    a = write_labels(tmp_path / "a.txt", "\\\t100.0\t200.0\r\n1.0005\t2.5\tlong breath\r\n\r\n0.5\t0.75\r\n3\t4\tx\n")
    b = write_labels(tmp_path / "b.txt", "1\t2\n")
    g = gold.gold_from_audacity({"pb": b, "pa": a}, 44100)
    assert g["sample_rate"] == 44100 and g["parts"] == {"pb": [], "pa": []}
    assert [(p["part"], p["start"], p["end"], p["label"]) for p in g["pauses"]] == [
        ("pb", 44100, 88200, ""), ("pa", 22050, 33075, ""), ("pa", round(1.0005 * 44100), 110250, "long breath"),
        ("pa", 132300, 176400, "x")]
    assert g["pauses"][1]["start_s"] == 0.5 and g["pauses"][1]["duration_s"] == 0.25
    assert gold.gold_from_audacity({"pa": a})["sample_rate"] == 44100


@pytest.mark.parametrize("body,line", [("1\tabc\tx\n", 1), ("\n1\t1\n", 2), ("2\t1\n", 1), ("-1\t2\n", 1), ("1\n", 1), ("nan\t2\n", 1)])
def test_audacity_errors_name_file_and_line(tmp_path, body, line):
    f = write_labels(tmp_path / "bad.txt", body)
    with pytest.raises(ValueError) as e:
        gold.gold_from_audacity({"a": f})
    assert "bad.txt" in str(e.value) and f":{line}" in str(e.value)


@pytest.mark.parametrize("mutate,key", [
    (lambda g: g.update(sample_rate=0), "sample_rate"),
    (lambda g: g.update(sample_rate=44100.0), "sample_rate"),
    (lambda g: g.update(parts=[]), "parts"),
    (lambda g: g["parts"]["a"].append({"speech_start": 5, "speech_end": 5}), "speech_start"),
    (lambda g: g["parts"]["a"].append({"speech_start": 1.5, "speech_end": 5}), "speech_start"),
    (lambda g: g["parts"]["a"].append({"speech_start": 1, "speech_end": 5, "end_reliable": "no"}), "end_reliable"),
    (lambda g: g.update(pauses=[{"part": "a", "start": 9, "end": 3}]), "pauses"),
    (lambda g: g.update(pauses={}), "pauses"),
])
def test_load_gold_validation(tmp_path, mutate, key):
    g = mk_gold([(1, 2)])
    mutate(g)
    p = tmp_path / "g.json"
    p.write_text(json.dumps(g))
    with pytest.raises(ValueError) as e:
        gold.load_gold(p)
    assert "g.json" in str(e.value) and key in str(e.value)


def test_load_gold_defaults(tmp_path):
    p = tmp_path / "g.json"
    p.write_text(json.dumps({"sample_rate": 10, "parts": {"a": [{"speech_start": 1, "speech_end": 2, "extra": 1}]}}))
    g = gold.load_gold(p)
    assert g["pauses"] == [] and g["parts"]["a"][0]["end_reliable"] is True and g["parts"]["a"][0].get("clip") is None


def run_setup(tmp_path):
    g = mk_gold(GOLD5[:2], pauses=[(2.1, 2.9)])
    gp = tmp_path / "gold.json"
    gp.write_text(json.dumps(g))
    work = tmp_path / "work"
    write_cuts(work, "a", refined(GOLD5[:2], [10, 10], [10, 10]), gaps=[0.5, None])
    return gp, work


def test_run_gold_writes_identical_reports_without_text(tmp_path):
    gp, work = run_setup(tmp_path)
    rep = gold.run_gold(work, ["a", "zz"], GoldConfig(file=str(gp)))
    assert rep["parts_without_gold"] == ["zz"] and list(rep["parts"]) == ["a"]
    first = {n: (work / "gold" / n).read_bytes() for n in ("report.json", "report.md")}
    assert json.loads(first["report.json"]) == rep
    for blob in first.values():
        assert b"GOLDMARK" not in blob and b"CLIPMARK" not in blob
    assert b"0.5" in first["report.md"] and b"IoU" in first["report.md"]
    gold.run_gold(work, ["a", "zz"], GoldConfig(file=str(gp)))
    assert {n: (work / "gold" / n).read_bytes() for n in first} == first


def test_run_gold_audacity_source(tmp_path):
    gp, work = run_setup(tmp_path)
    lab = write_labels(tmp_path / "a.txt", "2.1\t2.9\tp\n")
    rep = gold.run_gold(work, ["a"], GoldConfig(audacity={"a": str(lab)}))
    assert rep["parts"]["a"]["pauses"]["gold"] == 1 and rep["parts"]["a"]["gold_spans"] == 0
    assert rep["sample_rate"] == 44100


def test_run_gold_source_rule_and_parts(tmp_path):
    gp, work = run_setup(tmp_path)
    with pytest.raises(ValueError):
        gold.run_gold(work, ["a"], GoldConfig())
    with pytest.raises(ValueError):
        gold.run_gold(work, ["a"], GoldConfig(file=str(gp), audacity={"a": str(gp)}))
    with pytest.raises(ValueError):
        gold.run_gold(work, ["zz"], GoldConfig(file=str(gp)))
    with pytest.raises(FileNotFoundError, match="a.cuts.jsonl"):
        gold.run_gold(tmp_path / "empty", ["a"], GoldConfig(file=str(gp)))


def test_total_pools_parts(tmp_path):
    g = mk_gold([(1, 2), (3, 4)])
    g["parts"]["b"] = mk_gold([(1, 2)])["parts"]["a"]
    write_cuts(tmp_path, "a", refined([(1, 2), (3, 4)], [10, 10], [10, 10]))
    write_cuts(tmp_path, "b", refined([(1, 2)], [40], [40]))
    r = gold.score_run(g, tmp_path, ["a", "b"])
    t = r["total"]
    assert t["gold_spans"] == 3 and t["matched"] == 3 and t["start_error"]["n"] == 3
    assert t["start_error"]["median_ms"] == 10.0 and t["start_error"]["p90_ms"] == 34.0


def test_end_to_end_with_the_real_cutter(tmp_path):
    items = [pz(0.5)]
    for i in range(5):
        items += [sp(1.2 + 0.3 * (i % 3), est_s=3.0), pz(0.8) if i < 4 else pz(0.5)]
    pcm, truth = make_part(items, 5)
    units = make_units(items, "p")
    write_work(tmp_path, {"p": (pcm, units, stub_spans(units, truth, 0.02, 5))})
    cut.run_cut(tmp_path, ["p"], CutConfig())
    rate = 44100

    def gold_of(shift):
        return {"sample_rate": rate, "parts": {"p": [
            {"clip": u["id"], "speech_start": round(s / SR * rate) + shift, "speech_end": round(e / SR * rate) + shift}
            for u, (s, e) in zip(units, truth["speech"])]}, "pauses": []}

    ok = gold.score_run(gold_of(0), tmp_path, ["p"])["parts"]["p"]
    assert ok["unsafe_cut_rate"] == 0 and ok["matched"] == ok["gold_spans"] == 5
    for k in ("start_error", "end_error"):
        assert ok[k]["within_50ms"] == 1.0 and ok[k]["median_ms"] <= 20.0
    assert ok["aligner"]["matched"] == 5
    bad = gold.score_run(gold_of(int(0.3 * rate)), tmp_path, ["p"])["parts"]["p"]
    assert bad["matched"] < ok["matched"] or bad["start_error"]["median_ms"] > ok["start_error"]["median_ms"]


# ---------------------------------------------------------------- lost speech

F = 240
THR = -45.0
LOUD, QUIET = -30.0, -70.0


def lv(n, loud):
    """frame levels: n frames, LOUD inside the [a, b) frame ranges of loud, QUIET elsewhere."""
    out = np.full(n, QUIET)
    for a, b in loud:
        out[a:b] = LOUD
    return out


def lost(spans, clips, level, floor=-60.0):
    return gold._lost_speech([(a * F, b * F) for a, b in spans], [(a * F, b * F) for a, b in clips], level, F, THR, floor)


def zero():
    return {"head_s": 0.0, "inner_s": 0.0, "tail_s": 0.0, "lost_s": 0.0, "max_over_floor_db": None}


def test_core_fully_covered_span_loses_nothing():
    assert lost([(0, 10)], [(0, 10)], lv(10, [(0, 10)])) == [zero()]
    assert lost([(2, 8)], [(0, 10)], lv(10, [(0, 10)])) == [zero()]


def test_core_head_inner_tail_counts():
    r = lost([(0, 10)], [(3, 10)], lv(10, [(0, 10)]))[0]
    assert (r["head_s"], r["inner_s"], r["tail_s"], r["lost_s"]) == (0.03, 0.0, 0.0, 0.03)
    r = lost([(0, 10)], [(0, 6)], lv(10, [(0, 10)]))[0]
    assert (r["head_s"], r["inner_s"], r["tail_s"], r["lost_s"]) == (0.0, 0.0, 0.04, 0.04)
    r = lost([(0, 10)], [(0, 3), (6, 10)], lv(10, [(0, 10)]))[0]
    assert (r["head_s"], r["inner_s"], r["tail_s"], r["lost_s"]) == (0.0, 0.03, 0.0, 0.03)
    r = lost([(0, 12)], [(2, 4), (7, 9)], lv(12, [(0, 12)]))[0]
    assert (r["head_s"], r["inner_s"], r["tail_s"], r["lost_s"]) == (0.02, 0.03, 0.03, 0.08)
    assert r["max_over_floor_db"] == 30.0


def test_core_quiet_gap_is_not_lost():
    assert lost([(0, 10)], [(0, 3), (6, 10)], lv(10, [(0, 3), (6, 10)])) == [zero()]
    assert lost([(0, 10)], [(2, 10)], lv(10, [(2, 10)])) == [zero()]


def test_core_partly_inside_frames_are_not_judged():
    level = lv(10, [(0, 10)])
    r = gold._lost_speech([(F // 2, 10 * F - F // 2)], [(F, 9 * F)], level, F, THR, -60.0)
    assert r == [zero()]


def test_core_level_equal_to_threshold_is_not_speech():
    level = lv(10, [(0, 3), (6, 10)])
    level[3:6] = THR
    assert lost([(0, 10)], [(0, 3), (6, 10)], level) == [zero()]
    level[3:6] = THR + 0.1
    assert lost([(0, 10)], [(0, 3), (6, 10)], level)[0]["inner_s"] == 0.03


def test_core_touching_clips_count_as_one():
    level = lv(10, [(0, 10)])
    assert gold._lost_speech([(0, 10 * F)], [(0, 5 * F + 120), (5 * F + 120, 10 * F)], level, F, THR, -60.0) == [zero()]
    r = gold._lost_speech([(0, 10 * F)], [(0, 5 * F + 120), (5 * F + 121, 10 * F)], level, F, THR, -60.0)
    assert r[0]["inner_s"] == 0.01


def test_core_span_without_an_overlapping_clip_is_not_judged():
    level = lv(20, [(0, 20)])
    assert lost([(0, 10)], [(12, 20)], level) == [None]
    assert lost([(5, 10)], [(0, 5)], level) == [None]
    assert lost([(0, 10), (12, 20)], [(12, 20)], level) == [None, zero()]


def audio(root, part, segs, total_s=10.0, floor=-60.0, speech_db=15.0, report=True):
    """segs: [(start_s, end_s, dbfs)] constant square-wave levels on a zero background; start/end on 10 ms frames."""
    pcm = np.zeros(round(total_s * SR), dtype=np.int16)
    for a, b, db in segs:
        pcm[round(a * SR):round(b * SR)] = round(32768 * 10 ** (db / 20))
        pcm[round(a * SR):round(b * SR):2] *= -1
    (root / "audio").mkdir(parents=True, exist_ok=True)
    sf.write(str(root / "audio" / f"{part}.24k.wav"), pcm, SR, subtype="PCM_16")
    if report:
        (root / "cut").mkdir(parents=True, exist_ok=True)
        rp = root / "cut" / "report.json"
        parts = json.loads(rp.read_text())["parts"] if rp.exists() else {}
        parts[part] = {"floor_db": floor, "speech_thr_db": None if floor is None else floor + speech_db}
        rp.write_text(json.dumps({"parts": parts}))


def lost_of(root, g, part="a", **kw):
    r = gold.score_run(g, root, [part], 0.5, **kw)
    return r["parts"][part]["lost_speech"], r


def test_lost_speech_trimmed_clip_fails_and_is_named(tmp_path):
    g = mk_gold([(1, 2), (3, 4)])
    audio(tmp_path, "a", [(1, 2, -30), (3, 4, -30)])
    write_cuts(tmp_path, "a", [(1.0, 2.0), (3.3, 4.0)], pad_start=0, pad_end=0)
    ls, r = lost_of(tmp_path, g)
    assert ls["covered_spans"] == 2 and ls["lost_spans"] == 1 and ls["lost_speech_rate"] == 0.5
    assert ls["lost_s"] == 0.3 and ls["floor_db"] == -60.0 and ls["speech_thr_db"] == -45.0
    assert r["lost"] == [{"part": "a", "gold_clip": "T-1", "matched": True, "end_reliable": True, "lost_s": 0.3, "head_s": 0.3,
                          "inner_s": 0.0, "tail_s": 0.0, "max_over_floor_db": 30.0, "words": None, "rejected_words": None}]
    assert ls["matched_covered"] == 2 and ls["matched_lost_spans"] == 1 and ls["matched_lost_speech_rate"] == 0.5
    write_cuts(tmp_path, "a", [(1.0, 2.0), (3.0, 4.0)], pad_start=0, pad_end=0)
    ls, r = lost_of(tmp_path, g)
    assert ls["matched_lost_speech_rate"] == 0 and ls["lost_speech_rate"] == 0 and ls["lost_s"] == 0 and r["lost"] == []


def test_lost_speech_tail_and_inner_split(tmp_path):
    g = mk_gold([(1, 3)])
    audio(tmp_path, "a", [(1, 3, -30)])
    write_cuts(tmp_path, "a", [(1.0, 1.5), (1.7, 2.6)], pad_start=0, pad_end=0)
    row = lost_of(tmp_path, g)[1]["lost"][0]
    assert (row["head_s"], row["inner_s"], row["tail_s"], row["lost_s"]) == (0.0, 0.2, 0.4, 0.6)


def test_lost_speech_end_unreliable_span_is_scanned_only_up_to_its_end(tmp_path):
    g = mk_gold([(1, 2)], reliable=[False])
    audio(tmp_path, "a", [(1, 2.5, -30)])
    write_cuts(tmp_path, "a", [(1.0, 1.7)], pad_start=0, pad_end=0)
    ls, r = lost_of(tmp_path, g)
    assert r["lost"][0]["tail_s"] == 0.3 and r["lost"][0]["end_reliable"] is False and ls["lost_s"] == 0.3
    write_cuts(tmp_path, "a", [(1.0, 2.0)], pad_start=0, pad_end=0)
    assert lost_of(tmp_path, g)[0]["lost_s"] == 0


def test_lost_speech_matched_vs_covered(tmp_path):
    g = mk_gold([(1, 3), (5, 6)])
    audio(tmp_path, "a", [(1, 3, -30), (5, 6, -30)])
    write_cuts(tmp_path, "a", [(1.0, 1.8), (5.0, 6.0)], pad_start=0, pad_end=0)
    ls, r = lost_of(tmp_path, g)
    assert r["parts"]["a"]["matched"] == 1
    assert ls["covered_spans"] == 2 and ls["lost_spans"] == 1 and ls["lost_speech_rate"] == 0.5
    assert ls["matched_covered"] == 1 and ls["matched_lost_spans"] == 0 and ls["matched_lost_speech_rate"] == 0
    assert [(x["gold_clip"], x["matched"]) for x in r["lost"]] == [("T-0", False)]


def test_lost_speech_unjudged_span_and_empty_rates(tmp_path):
    g = mk_gold([(1, 2), (6, 7)])
    audio(tmp_path, "a", [(1, 2, -30), (6, 7, -30)])
    write_cuts(tmp_path, "a", [(1.0, 2.0)], pad_start=0, pad_end=0)
    ls = lost_of(tmp_path, g)[0]
    assert ls["covered_spans"] == 1 and ls["lost_speech_rate"] == 0
    write_cuts(tmp_path, "a", [(3.0, 4.0)], pad_start=0, pad_end=0)
    ls = lost_of(tmp_path, g)[0]
    assert ls["covered_spans"] == 0 and ls["lost_speech_rate"] is None and ls["matched_lost_speech_rate"] is None


def test_lost_speech_db_override(tmp_path):
    g = mk_gold([(1, 2)])
    audio(tmp_path, "a", [(1, 2, -42)])
    write_cuts(tmp_path, "a", [(1.0, 1.5)], pad_start=0, pad_end=0)
    assert lost_of(tmp_path, g)[0]["lost_s"] == 0.5
    ls = lost_of(tmp_path, g, speech_db=15.0)[0]
    assert ls["lost_s"] == 0.5 and ls["speech_thr_db"] == -45.0
    ls = lost_of(tmp_path, g, speech_db=20.0)[0]
    assert ls["lost_s"] == 0 and ls["speech_thr_db"] == -40.0


OLD_KEYS = {"gold_spans", "matched", "unsafe_edges", "unsafe_spans", "unsafe_cut_rate", "single_clip_spans", "unsafe_spans_single_clip",
            "unsafe_cut_rate_single_clip", "start_error", "end_error", "aligner", "pauses"}


def test_lost_speech_is_none_without_audio_report_or_floor(tmp_path):
    g = mk_gold([(1, 2)])
    write_cuts(tmp_path, "a", [(1.0, 1.5)], pad_start=0, pad_end=0)
    r = score(tmp_path, g)
    assert r["parts"]["a"]["lost_speech"] is None and r["total"]["lost_speech"] is None and r["lost"] == []
    assert set(r["parts"]["a"]) - {"lost_speech"} == OLD_KEYS
    audio(tmp_path, "a", [(1, 2, -30)], report=False)
    assert score(tmp_path, g)["parts"]["a"]["lost_speech"] is None
    audio(tmp_path, "a", [(1, 2, -30)], floor=None)
    assert score(tmp_path, g)["parts"]["a"]["lost_speech"] is None
    assert lost_of(tmp_path, g, speech_db=15.0)[0] is None
    audio(tmp_path, "a", [(1, 2, -30)])
    assert score(tmp_path, g)["parts"]["a"]["lost_speech"]["lost_s"] == 0.5


def test_lost_speech_total_pools_counts(tmp_path):
    g = mk_gold([(1, 2), (3, 4)])
    g["parts"]["b"] = mk_gold([(1, 2)])["parts"]["a"]
    audio(tmp_path, "a", [(1, 2, -30), (3, 4, -30)])
    audio(tmp_path, "b", [(1, 2, -30)])
    write_cuts(tmp_path, "a", [(1.0, 2.0), (3.2, 4.0)], pad_start=0, pad_end=0)
    write_cuts(tmp_path, "b", [(1.0, 2.0)], pad_start=0, pad_end=0)
    r = gold.score_run(g, tmp_path, ["a", "b"])
    t = r["total"]["lost_speech"]
    assert t["covered_spans"] == 3 and t["lost_spans"] == 1 and t["lost_speech_rate"] == round(1 / 3, 6)
    assert t["matched_covered"] == 3 and t["matched_lost_spans"] == 1 and t["lost_s"] == 0.2
    assert t["floor_db"] is None and t["speech_thr_db"] is None
    assert r["parts"]["b"]["lost_speech"]["lost_speech_rate"] == 0


def test_run_gold_lost_speech_reports_are_identical_and_free_of_text(tmp_path):
    work = tmp_path / "work"
    g = mk_gold([(1, 2), (3, 4)])
    gp = tmp_path / "gold.json"
    gp.write_text(json.dumps(g))
    audio(work, "a", [(1, 2, -30), (3, 4, -30)])
    write_cuts(work, "a", [(1.0, 2.0), (3.3, 4.0)], pad_start=0, pad_end=0)
    rep = gold.run_gold(work, ["a"], GoldConfig(file=str(gp)))
    assert rep["parts"]["a"]["lost_speech"]["lost_spans"] == 1
    first = {n: (work / "gold" / n).read_bytes() for n in ("report.json", "report.md")}
    for blob in first.values():
        assert b"GOLDMARK" not in blob and b"CLIPMARK" not in blob
    md = first["report.md"].decode()
    assert "## Lost speech" in md and "lost rate" in md and "T-1" in md
    gold.run_gold(work, ["a"], GoldConfig(file=str(gp)))
    assert {n: (work / "gold" / n).read_bytes() for n in first} == first
    write_cuts(work, "a", [(1.0, 2.0), (3.0, 4.0)], pad_start=0, pad_end=0)
    gold.run_gold(work, ["a"], GoldConfig(file=str(gp)))
    assert "## Lost speech\n\nnone" in (work / "gold" / "report.md").read_text()


def cutter_work(tmp_path):
    items = [pz(0.5)]
    for i in range(5):
        items += [sp(1.2 + 0.3 * (i % 3), est_s=3.0), pz(0.8) if i < 4 else pz(0.5)]
    pcm, truth = make_part(items, 5)
    units = make_units(items, "p")
    write_work(tmp_path, {"p": (pcm, units, stub_spans(units, truth, 0.02, 5))})
    cut.run_cut(tmp_path, ["p"], CutConfig())
    return units, truth


def rewrite_clips(root, edges, pad=None):
    path = root / "cut" / "p.cuts.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert len(rows) == len(edges)
    for row, (a, b) in zip(rows, edges):
        row["start_sample"], row["end_sample"] = a, b
        if pad is not None:
            row["pad_start_s"] = row["pad_end_s"] = pad
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def real_gold(units, truth, widen=0):
    return {"sample_rate": SR, "parts": {"p": [{"clip": u["id"], "speech_start": s - widen, "speech_end": e + widen}
                                                  for u, (s, e) in zip(units, truth["speech"])]}, "pauses": []}


def test_lost_speech_negative_control_with_the_real_cutter(tmp_path):
    units, truth = cutter_work(tmp_path)
    ok = gold.score_run(real_gold(units, truth), tmp_path, ["p"])
    assert ok["parts"]["p"]["lost_speech"]["matched_lost_speech_rate"] == 0 and ok["lost"] == []
    assert ok["parts"]["p"]["lost_speech"]["covered_spans"] == 5
    inside = int(0.3 * SR)
    rewrite_clips(tmp_path, [(s + inside, e + 2400) for s, e in truth["speech"]])
    bad = gold.score_run(real_gold(units, truth), tmp_path, ["p"])
    assert bad["parts"]["p"]["lost_speech"]["matched_lost_speech_rate"] > 0
    assert bad["parts"]["p"]["unsafe_cut_rate"] > 0
    assert [x["gold_clip"] for x in bad["lost"]] == [u["id"] for u in units] and {x["part"] for x in bad["lost"]} == {"p"}
    for x in bad["lost"]:
        assert abs(x["head_s"] - 0.3) <= 0.03 and x["tail_s"] == 0.0


def test_unsafe_edge_in_a_quiet_pause_is_not_lost_speech(tmp_path):
    units, truth = cutter_work(tmp_path)
    pad, widen = int(0.1 * SR), int(0.3 * SR)
    rewrite_clips(tmp_path, [(s - pad, e + pad) for s, e in truth["speech"]])
    r = gold.score_run(real_gold(units, truth, widen), tmp_path, ["p"])["parts"]["p"]
    assert r["unsafe_cut_rate"] > 0 and r["lost_speech"]["lost_spans"] == 0 and r["lost_speech"]["covered_spans"] == 5


# ---------------------------------------------------------------- ASR-confirmed lost words

def lostw(spans, clips, level, mids, floor=-60.0, rejected=()):
    return gold._lost_speech([(a * F, b * F) for a, b in spans], [(a * F, b * F) for a, b in clips], level, F, THR, floor,
                             word_mids=None if mids is None else np.array(sorted(mids)), rejected=rejected)


def mid(f):
    return (f + 0.5) * F / SR


def test_core_word_on_a_lost_frame_counts_once():
    level = lv(30, [(0, 30)])
    assert lostw([(0, 20)], [(0, 10)], level, [mid(12)])[0]["words"] == 1
    assert lostw([(0, 20)], [(0, 10)], level, [mid(12), mid(13)])[0]["words"] == 2


def test_core_word_tolerance_is_five_frames():
    level = lv(30, [(0, 30)])
    assert lostw([(0, 20)], [(0, 10)], level, [mid(5)])[0]["words"] == 1
    assert lostw([(0, 20)], [(0, 10)], level, [mid(4)])[0]["words"] == 0
    assert lostw([(0, 20)], [(0, 10)], level, [mid(4), mid(5)])[0]["words"] == 1


def test_core_word_tolerance_is_five_frames_after_the_lost_frames_too():
    level = lv(30, [(0, 30)])
    assert lostw([(0, 20)], [(10, 20)], level, [mid(14)])[0]["words"] == 1
    assert lostw([(0, 20)], [(10, 20)], level, [mid(15)])[0]["words"] == 0


def test_core_word_outside_the_span_does_not_count():
    level = lv(30, [(0, 30)])
    assert lostw([(0, 20)], [(0, 10)], level, [mid(22)])[0]["words"] == 0
    assert lostw([(0, 20)], [(0, 10)], level, [mid(22), mid(15)])[0]["words"] == 1


def test_core_lost_run_without_a_word_has_words_zero():
    level = lv(30, [(0, 30)])
    r = lostw([(0, 20)], [(0, 10)], level, [])[0]
    assert r["lost_s"] == 0.1 and r["words"] == 0
    assert lostw([(0, 10)], [(0, 10)], level, [mid(5)])[0]["words"] == 0


def test_core_without_word_mids_words_is_none():
    level = lv(30, [(0, 30)])
    r = lostw([(0, 20)], [(0, 10)], level, None)[0]
    assert r["lost_s"] == 0.1 and r.get("words") is None


def write_asr(root, part, words, **extra):
    (root / "align").mkdir(parents=True, exist_ok=True)
    source = root / "audio" / f"{part}.16k.wav"
    source.write_bytes((root / "audio" / f"{part}.24k.wav").read_bytes())
    (root / "align" / f"{part}.asr.json").write_text(json.dumps({"audio_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "config": {}, "model_id": "stub",
                                                                 "words": [[w, a, b] for w, a, b in words], **extra}))


@pytest.mark.parametrize("identity", ["stale", "missing", "source_missing"])
def test_existing_asr_cache_requires_matching_audio_identity(tmp_path, identity):
    g = mk_gold([(1, 2)])
    audio(tmp_path, "a", [(1, 2, -30)])
    write_cuts(tmp_path, "a", [(1, 2)])
    write_asr(tmp_path, "a", [word_at(1.5)])
    path = tmp_path / "align" / "a.asr.json"
    cache = json.loads(path.read_text())
    if identity == "missing":
        del cache["audio_sha256"]
    elif identity == "stale":
        cache["audio_sha256"] = "0" * 64
    else:
        (tmp_path / "audio" / "a.16k.wav").unlink()
    path.write_text(json.dumps(cache))
    with pytest.raises(ValueError, match="a.asr.json.*audio"):
        score(tmp_path, g)


def word_at(t, w=0.2):
    return ("w", t - w / 2, t + w / 2)


def test_word_fields_with_the_cache(tmp_path):
    g = mk_gold([(1, 2), (3, 4)])
    audio(tmp_path, "a", [(1, 2, -30), (3, 4, -30)])
    write_cuts(tmp_path, "a", [(1.0, 2.0), (3.3, 4.0)], pad_start=0, pad_end=0)
    write_asr(tmp_path, "a", [word_at(1.5), word_at(3.1), word_at(3.7)])
    ls, r = lost_of(tmp_path, g)
    assert ls["lost_spans"] == 1 and ls["matched_lost_speech_rate"] == 0.5
    assert (ls["word_spans"], ls["lost_words"], ls["word_rate"]) == (1, 1, 0.5)
    assert (ls["matched_word_spans"], ls["matched_word_rate"]) == (1, 0.5)
    assert ls["criterion"] == "words" and ls["criterion_rate"] == 0.5
    assert [(x["gold_clip"], x["words"]) for x in r["lost"]] == [("T-1", 1)]
    assert r["total"]["lost_speech"]["criterion"] == "words" and r["total"]["lost_speech"]["lost_words"] == 1


def test_no_cache_means_energy_criterion(tmp_path):
    g = mk_gold([(1, 2), (3, 4)])
    audio(tmp_path, "a", [(1, 2, -30), (3, 4, -30)])
    write_cuts(tmp_path, "a", [(1.0, 2.0), (3.3, 4.0)], pad_start=0, pad_end=0)
    ls, r = lost_of(tmp_path, g)
    for k in ("word_spans", "lost_words", "word_rate", "matched_word_spans", "matched_word_rate"):
        assert ls[k] is None
    assert ls["criterion"] == "energy" and ls["criterion_rate"] == ls["matched_lost_speech_rate"] == 0.5
    assert r["lost"][0].get("words") is None


def test_total_words_only_when_every_part_has_the_cache(tmp_path):
    g = mk_gold([(1, 2), (3, 4)])
    g["parts"]["b"] = mk_gold([(1, 2)])["parts"]["a"]
    audio(tmp_path, "a", [(1, 2, -30), (3, 4, -30)])
    audio(tmp_path, "b", [(1, 2, -30)])
    write_cuts(tmp_path, "a", [(1.0, 2.0), (3.2, 4.0)], pad_start=0, pad_end=0)
    write_cuts(tmp_path, "b", [(1.3, 2.0)], pad_start=0, pad_end=0)
    write_asr(tmp_path, "a", [word_at(3.1)])
    t = gold.score_run(g, tmp_path, ["a", "b"])["total"]["lost_speech"]
    assert t["criterion"] == "energy" and t["criterion_rate"] == t["matched_lost_speech_rate"] == round(2 / 3, 6)
    for k in ("word_spans", "lost_words", "word_rate", "matched_word_spans", "matched_word_rate"):
        assert t[k] is None
    write_asr(tmp_path, "b", [word_at(1.1)])
    r = gold.score_run(g, tmp_path, ["a", "b"])
    t = r["total"]["lost_speech"]
    assert t["criterion"] == "words" and (t["word_spans"], t["lost_words"], t["word_rate"]) == (2, 2, round(2 / 3, 6))
    assert (t["matched_word_spans"], t["matched_word_rate"]) == (2, round(2 / 3, 6)) and t["criterion_rate"] == round(2 / 3, 6)
    assert r["parts"]["b"]["lost_speech"]["criterion"] == "words"


@pytest.mark.parametrize("content", ["not json", "[]", json.dumps({"words": 3}), json.dumps({"words": [["w", 1.0]]}),
                                     json.dumps({"audio_sha256": "0"}), json.dumps({"words": [["w", "x", 2.0]]})])
def test_corrupt_asr_cache_raises_naming_the_file(tmp_path, content):
    g = mk_gold([(1, 2)])
    audio(tmp_path, "a", [(1, 2, -30)])
    write_cuts(tmp_path, "a", [(1.0, 2.0)], pad_start=0, pad_end=0)
    (tmp_path / "align").mkdir(exist_ok=True)
    (tmp_path / "align" / "a.asr.json").write_text(content)
    with pytest.raises(ValueError, match="a.asr.json"):
        score(tmp_path, g)


def asr_per_unit(root, truth, at=0.5):
    write_asr(root, "p", [word_at((s + at * (e - s)) / SR) for s, e in truth["speech"]])


def test_lost_words_negative_control_with_the_real_cutter(tmp_path):
    units, truth = cutter_work(tmp_path)
    asr_per_unit(tmp_path, truth)
    ls = gold.score_run(real_gold(units, truth), tmp_path, ["p"])["parts"]["p"]["lost_speech"]
    assert ls["criterion"] == "words" and ls["matched_word_rate"] == 0 and ls["criterion_rate"] == 0 and ls["lost_words"] == 0
    rewrite_clips(tmp_path, [(s + int(0.3 * SR), e + 2400) for s, e in truth["speech"]])
    write_asr(tmp_path, "p", [word_at(s / SR + 0.1) for s, _ in truth["speech"]])
    bad = gold.score_run(real_gold(units, truth), tmp_path, ["p"])
    ls = bad["parts"]["p"]["lost_speech"]
    assert ls["matched_word_rate"] > 0 and ls["lost_words"] >= 1 and ls["criterion_rate"] == ls["matched_word_rate"]
    assert ls["criterion"] == "words" and ls["matched_word_spans"] >= 1
    assert bad["lost"] and all(x["words"] >= 1 for x in bad["lost"])
    assert [x["gold_clip"] for x in bad["lost"]] == [u["id"] for u in units]


def test_breath_in_a_pause_is_lost_energy_but_not_a_lost_word(tmp_path):
    items = [pz(0.5), sp(1.2, est_s=3.0), pz(1.2, "breath", 0.5), sp(1.2, est_s=3.0), pz(0.5)]
    pcm, truth = make_part(items, 5)
    units = make_units(items, "p")
    write_work(tmp_path, {"p": (pcm, units, stub_spans(units, truth, 0.02, 5))})
    cut.run_cut(tmp_path, ["p"], CutConfig())
    rewrite_clips(tmp_path, list(truth["speech"]), pad=0.0)
    (s0, e0), (s1, e1) = truth["speech"]
    g = {"sample_rate": SR, "parts": {"p": [{"clip": "p-0", "speech_start": s0, "speech_end": e0 + int(0.9 * SR)}]}, "pauses": []}
    asr_per_unit(tmp_path, truth)
    ls = gold.score_run(g, tmp_path, ["p"])["parts"]["p"]["lost_speech"]
    assert ls["lost_s"] > 0 and ls["matched_covered"] == 1 and ls["matched_lost_spans"] == 1
    assert ls["lost_words"] == 0 and ls["word_spans"] == 0 and ls["criterion"] == "words" and ls["criterion_rate"] == 0


def test_run_gold_word_reports_are_identical_and_free_of_text(tmp_path):
    work = tmp_path / "work"
    gp = tmp_path / "gold.json"
    gp.write_text(json.dumps(mk_gold([(1, 2), (3, 4)])))
    audio(work, "a", [(1, 2, -30), (3, 4, -30)])
    write_cuts(work, "a", [(1.0, 2.0), (3.3, 4.0)], pad_start=0, pad_end=0)
    write_asr(work, "a", [("GOLDMARK", 3.0, 3.2), ("CLIPMARK", 1.4, 1.6)])
    rep = gold.run_gold(work, ["a"], GoldConfig(file=str(gp)))
    assert rep["parts"]["a"]["lost_speech"]["lost_words"] == 1
    first = {n: (work / "gold" / n).read_bytes() for n in ("report.json", "report.md")}
    for blob in first.values():
        assert b"GOLDMARK" not in blob and b"CLIPMARK" not in blob
    md = first["report.md"].decode()
    assert "lost words" in md and "word rate" in md and "criterion rate" in md and "words 1" in md
    gold.run_gold(work, ["a"], GoldConfig(file=str(gp)))
    assert {n: (work / "gold" / n).read_bytes() for n in first} == first


# ---------------------------------------------------------------- words of rejected units apart from the words lost by the cut

def test_core_word_in_a_rejected_region_is_counted_apart():
    level = lv(30, [(0, 30)])
    r = lostw([(0, 20)], [(0, 10)], level, [mid(12)], rejected=[(0.11, 0.15)])[0]
    assert (r["words"], r["rejected_words"]) == (0, 1)
    r = lostw([(0, 20)], [(0, 10)], level, [mid(12)], rejected=[(0.2, 0.3)])[0]
    assert (r["words"], r["rejected_words"]) == (1, 0)


def test_core_rejected_region_is_half_open():
    level = lv(30, [(0, 30)])
    r = lostw([(0, 20)], [(0, 10)], level, [mid(12), mid(15)], rejected=[(mid(12), mid(15))])[0]
    assert (r["words"], r["rejected_words"]) == (1, 1)


def test_core_both_kinds_in_one_span_and_no_regions():
    level = lv(30, [(0, 30)])
    r = lostw([(0, 20)], [(0, 10)], level, [mid(12), mid(16)], rejected=[(0.11, 0.15)])[0]
    assert (r["words"], r["rejected_words"]) == (1, 1)
    r = lostw([(0, 20)], [(0, 10)], level, [mid(12), mid(16)])[0]
    assert (r["words"], r["rejected_words"]) == (2, 0)
    assert "rejected_words" not in lostw([(0, 20)], [(0, 10)], level, None, rejected=[(0.0, 9.0)])[0]


def row(status, s=None, e=None):
    return {"id": "u", "status": status, "start_s": s, "end_s": e, "words": []}


def test_rejected_regions_rule():
    m = gold.REJECTED_MARGIN_S
    assert m == 0.08
    assert gold._rejected_regions(None) == [] and gold._rejected_regions([]) == []
    assert gold._rejected_regions([row("ok", 1, 2), row("ok", 3, 4)]) == []
    assert gold._rejected_regions([row("ok", 1, 2), row("low_conf", 2.5, 2.9), row("ok", 3, 4)]) == [(2 + m, 3 - m)]
    assert gold._rejected_regions([row("low_conf", 0.5, 0.9), row("ok", 3, 4)]) == [(0.0, 3 - m)]
    assert gold._rejected_regions([row("ok", 1, 2), row("low_conf", 2.5, 2.9)]) == [(2 + m, math.inf)]
    assert gold._rejected_regions([row("unmatched"), row("ok", 3, 4)]) == [(0.0, 3 - m)]
    assert gold._rejected_regions([row("ok", 1, 2), row("unmatched"), row("unmatched"), row("ok", 5, 6)]) == [(2 + m, 5 - m)]
    assert gold._rejected_regions([row("ok", 1, 2), row("ok"), row("ok", 5, 6)]) == [(2 + m, 5 - m)]
    assert gold._rejected_regions([row("ok"), row("ok", 5, 6)]) == [(0.0, 5 - m)]
    assert gold._rejected_regions([row("ok", 1, 2), row("low_conf"), row("ok", 2.05, 3)]) == []
    assert gold._rejected_regions([row("ok", 1, 2), row("low_conf"), row("ok", 3, 4), row("unmatched")]) == [(2 + m, 3 - m), (4 + m, math.inf)]


def gap_work(root, middle):
    audio(root, "a", [(1, 4, -30)])
    write_cuts(root, "a", [(1.0, 3.0)], pad_start=0, pad_end=0)
    write_align(root, "a", [(1.0, 2.9, "ok"), (3.1, 3.9, middle), (5.0, 6.0, "ok")])
    write_asr(root, "a", [word_at(1.5), word_at(3.5)])
    return lost_of(root, mk_gold([(1, 4)]))


def test_words_of_a_rejected_unit_are_not_a_lost_word(tmp_path):
    ls, r = gap_work(tmp_path, "low_conf")
    assert ls["lost_spans"] == 1 and (ls["word_spans"], ls["lost_words"], ls["word_rate"]) == (0, 0, 0.0)
    assert (ls["rejected_word_spans"], ls["rejected_words"], ls["matched_rejected_word_spans"]) == (1, 1, 1)
    assert ls["matched_word_rate"] == 0 and ls["criterion"] == "words" and ls["criterion_rate"] == 0
    assert [(x["gold_clip"], x["words"], x["rejected_words"]) for x in r["lost"]] == [("T-0", 0, 1)]


def test_negative_control_the_same_word_between_two_ok_units_is_lost(tmp_path):
    ls, r = gap_work(tmp_path, "ok")
    assert (ls["word_spans"], ls["lost_words"], ls["rejected_word_spans"], ls["rejected_words"]) == (1, 1, 0, 0)
    assert ls["criterion_rate"] == 1.0 and r["lost"][0]["words"] == 1 and r["lost"][0]["rejected_words"] == 0


def test_rejected_word_fields_are_none_without_the_cache(tmp_path):
    audio(tmp_path, "a", [(1, 4, -30)])
    write_cuts(tmp_path, "a", [(1.0, 3.0)], pad_start=0, pad_end=0)
    write_align(tmp_path, "a", [(1.0, 2.9, "ok"), (3.1, 3.9, "low_conf"), (5.0, 6.0, "ok")])
    ls, r = lost_of(tmp_path, mk_gold([(1, 4)]))
    for k in ("rejected_word_spans", "rejected_words", "matched_rejected_word_spans"):
        assert ls[k] is None
    assert r["lost"][0]["rejected_words"] is None


def test_total_rejected_words_pool_only_when_every_part_has_the_cache(tmp_path):
    g = mk_gold([(1, 4)])
    g["parts"]["b"] = mk_gold([(1, 4)])["parts"]["a"]
    for part in "ab":
        audio(tmp_path, part, [(1, 4, -30)])
        write_cuts(tmp_path, part, [(1.0, 3.0)], pad_start=0, pad_end=0)
        write_align(tmp_path, part, [(1.0, 2.9, "ok"), (3.1, 3.9, "low_conf"), (5.0, 6.0, "ok")])
    write_asr(tmp_path, "a", [word_at(3.5), word_at(3.6)])
    t = gold.score_run(g, tmp_path, ["a", "b"])["total"]["lost_speech"]
    for k in ("rejected_word_spans", "rejected_words", "matched_rejected_word_spans"):
        assert t[k] is None
    write_asr(tmp_path, "b", [word_at(3.5)])
    t = gold.score_run(g, tmp_path, ["a", "b"])["total"]["lost_speech"]
    assert (t["rejected_word_spans"], t["rejected_words"], t["matched_rejected_word_spans"]) == (2, 3, 2)
    assert (t["word_spans"], t["lost_words"], t["criterion_rate"]) == (0, 0, 0.0)


def test_run_gold_rejected_word_reports_show_the_column_and_hold_no_text(tmp_path):
    work = tmp_path / "work"
    gp = tmp_path / "gold.json"
    gp.write_text(json.dumps(mk_gold([(1, 4)])))
    audio(work, "a", [(1, 4, -30)])
    write_cuts(work, "a", [(1.0, 3.0)], pad_start=0, pad_end=0)
    write_align(work, "a", [(1.0, 2.9, "ok"), (3.1, 3.9, "low_conf"), (5.0, 6.0, "ok")])
    write_asr(work, "a", [("GOLDMARK", 3.4, 3.6)])
    rep = gold.run_gold(work, ["a"], GoldConfig(file=str(gp)))
    assert rep["parts"]["a"]["lost_speech"]["rejected_words"] == 1
    for n in ("report.json", "report.md"):
        blob = (work / "gold" / n).read_bytes()
        assert b"GOLDMARK" not in blob and b"CLIPMARK" not in blob
    md = (work / "gold" / "report.md").read_text()
    head = next(x for x in md.splitlines() if x.startswith("| part"))
    assert "| lost words | rejected words | word rate |" in head
    assert "\n|" + "---|" * 23 + "\n" in md
    assert "rejected-unit words 1" in md and "words 0" in md
