"""Gold-set scorer tests (src/data/prepare/gold.py). Synthetic data only; times in the unit tests use sample_rate 1000 so samples are ms."""
import json

import pytest

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
    assert bad["unsafe"] == [{"part": "a", "gold_clip": "T-0", "edge": "start", "t_s": 1.5, "gold_start_s": 1.0, "gold_end_s": 2.0}]
    write_cuts(tmp_path, "a", [(0.5, 1.5), (3.0, 4.0)])
    ends = score(tmp_path, g)["unsafe"]
    assert [(u["gold_clip"], u["edge"], u["t_s"]) for u in ends] == [("T-0", "end", 1.6)]


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
