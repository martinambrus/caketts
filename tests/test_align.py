"""Alignment tests: the long-form logic (pure functions on stub data)."""
import subprocess
import sys
import time

import numpy as np
import pytest

from src.data.prepare.align.longform import (OVERLAP_MAX_S, SUSPECT_MARGIN_S, Block, assign_status, fold,
                                             match_words, plan_blocks, resolve_monotonic, skipped_gaps, word_times)
from src.data.prepare.align.types import AlignConfig, Gap, UnitSpan, WordSpan
from src.data.prepare.asr import AsrWord


# ---- longform ----

def _lf_asr(n, step=1.0, length=0.9):
    return [AsrWord(f"w{j}", j * step, j * step + length) for j in range(n)]


def _lf_unit(uid, start, end, score=0.9, flags=None, n_words=2):
    k = (end - start) / n_words
    words = [WordSpan(f"{uid}{i}", start + i * k, start + (i + 1) * k, score) for i in range(n_words)]
    return UnitSpan(uid, start, end, words, score, score, "ok", list(flags or []))


def _lf_cfg(**kw):
    return AlignConfig(**{"margin_s": 1.0, "block_max_s": 600.0, **kw})


def test_fold():
    assert fold("Dobře") == "dobre"
    assert fold("74") == ""
    assert fold("Žluťoučký,") == "zlutoucky"
    assert fold("«Ďábel»") == "dabel"


def test_match_identical():
    keys = ["a", "b", "c"]
    assert match_words(keys, keys) == [(0, 0), (1, 1), (2, 2)]


def test_match_substitution():
    assert match_words(["a", "b", "c"], ["a", "x", "c"]) == [(0, 0), (2, 2)]


def test_match_inserted_asr_run():
    assert match_words(["a", "b", "c"], ["a", "x", "y", "z", "b", "c"]) == [(0, 0), (1, 4), (2, 5)]


def test_match_deleted_text_run():
    assert match_words(["a", "x", "y", "b", "c"], ["a", "b", "c"]) == [(0, 0), (3, 1), (4, 2)]


def test_match_later_copy_keeps_match():
    assert match_words(["a", "b", "a", "b"], ["a", "b"]) == [(2, 0), (3, 1)]
    assert match_words(["z", "z"], ["z"]) == [(1, 0)]


def test_match_empty_keys_never_match():
    assert match_words(["", "a", ""], ["", "a", ""]) == [(1, 1)]
    assert match_words([""], [""]) == []


def test_match_empty_lists():
    assert match_words([], ["a"]) == []
    assert match_words(["a"], []) == []
    assert match_words([], []) == []


def test_match_too_large():
    with pytest.raises(ValueError, match="split the part"):
        match_words(["a"] * 13000, ["a"] * 13000)


def test_match_3000_words_fast():
    rng = np.random.default_rng(0)
    text = [f"k{v}" for v in rng.integers(0, 400, 3000)]
    asr = list(text)
    del asr[100:130]
    asr.insert(2000, "extra")
    t0 = time.time()
    pairs = match_words(text, asr)
    assert time.time() - t0 < 10
    assert len(pairs) >= 2960
    assert all(text[i] == asr[j] for i, j in pairs)
    assert all(a[0] < b[0] and a[1] < b[1] for a, b in zip(pairs, pairs[1:]))


def test_word_times_interpolates():
    asr = [AsrWord("a", 2.0, 2.5), AsrWord("b", 4.0, 4.5)]
    start, end, rate = word_times(5, [(1, 0), (3, 1)], asr, 10.0)
    assert rate == pytest.approx(1.0)
    assert start == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0])
    assert end == pytest.approx([1.5, 2.5, 3.5, 4.5, 5.5])


def test_word_times_clamps():
    asr = [AsrWord("a", 1.0, 1.5)]
    start, end, rate = word_times(4, [(1, 0)], asr, 8.0)
    assert rate == pytest.approx(2.0)
    assert start == pytest.approx([0.0, 1.0, 3.0, 5.0])
    assert end[0] == 0.0 and end[1] == pytest.approx(1.5)
    asr = [AsrWord("a", 0.0, 1.0), AsrWord("b", 3.0, 4.0)]
    start, end, rate = word_times(4, [(0, 0), (1, 1)], asr, 5.0)
    assert rate == pytest.approx(3.0)
    assert start == pytest.approx([0.0, 3.0, 5.0, 5.0])
    assert end == pytest.approx([1.0, 4.0, 5.0, 5.0])
    assert (end >= start).all()


def test_word_times_no_pairs_uniform():
    start, end, rate = word_times(4, [], [], 8.0)
    assert rate == pytest.approx(2.0)
    assert start == pytest.approx([0.0, 2.0, 4.0, 6.0])
    assert end == pytest.approx([2.0, 4.0, 6.0, 8.0])


def _lf_identity(n):
    return [(i, i) for i in range(n)]


def test_plan_faithful_one_block():
    plan = plan_blocks([5, 5], _lf_identity(10), _lf_asr(10), 12.0, _lf_cfg())
    assert len(plan.blocks) == 1
    block = plan.blocks[0]
    assert (block.first, block.last, block.suspect) == (0, 1, False)
    assert block.t0 == 0.0 and block.t1 == pytest.approx(10.9)
    assert plan.gaps == [] and plan.flags == {}


def test_plan_block_max_split():
    cfg = _lf_cfg(block_max_s=12.0)
    plan = plan_blocks([5] * 8, _lf_identity(40), _lf_asr(40), 40.0, cfg)
    assert len(plan.blocks) > 1
    assert all(b.t1 - b.t0 <= 12.0 for b in plan.blocks)
    assert [b.first for b in plan.blocks][0] == 0 and plan.blocks[-1].last == 7
    assert all(a.last + 1 == b.first for a, b in zip(plan.blocks, plan.blocks[1:]))
    assert not any(b.suspect for b in plan.blocks)


def test_plan_single_unit_never_cut():
    plan = plan_blocks([30], _lf_identity(30), _lf_asr(30), 30.0, _lf_cfg(block_max_s=10.0))
    assert len(plan.blocks) == 1 and plan.blocks[0].first == plan.blocks[0].last == 0


def _lf_deleted_scenario():
    # text units of 5 words; the narrator read an extra 6 words after unit 0 that are not in the text
    asr = [AsrWord(f"w{j}", j * 1.0, j * 1.0 + 0.9) for j in range(16)]
    pairs = [(i, i) for i in range(5)] + [(i, i + 6) for i in range(5, 10)]
    return [5, 5], pairs, asr


def test_plan_deleted_text_gap_and_break():
    words, pairs, asr = _lf_deleted_scenario()
    plan = plan_blocks(words, pairs, asr, 16.0, _lf_cfg())
    assert plan.gaps == [Gap(5.0, 10.9, "extra_speech")]
    assert [(b.first, b.last, b.suspect) for b in plan.blocks] == [(0, 0, False), (1, 1, False)]


def test_plan_duplicated_text_suspect_block():
    # unit 1 is a copy of unit 0's text that was read once (the later copy matched)
    asr = _lf_asr(10)
    pairs = [(0, 0), (1, 1)] + [(5 + k, 2 + k) for k in range(3)]
    plan = plan_blocks([2, 3, 3], pairs, asr[:5], 6.0, _lf_cfg())
    assert [(b.first, b.last, b.suspect) for b in plan.blocks] == [(0, 0, False), (1, 1, True), (2, 2, False)]
    suspect = plan.blocks[1]
    assert suspect.t0 == pytest.approx(asr[1].end - SUSPECT_MARGIN_S)
    assert suspect.t1 == pytest.approx(asr[2].start + SUSPECT_MARGIN_S)


def test_plan_suspect_window_edges_and_min_width():
    asr = _lf_asr(3)
    plan = plan_blocks([2, 2], [(2, 0), (3, 1)], asr, 5.0, _lf_cfg())
    assert plan.blocks[0].suspect and plan.blocks[0].t0 == 0.0
    plan = plan_blocks([2, 2], [(0, 0), (1, 1)], asr, 5.0, _lf_cfg())
    assert plan.blocks[1].suspect and plan.blocks[1].t1 == 5.0
    asr = [AsrWord("a", 0.0, 1.0), AsrWord("b", 0.9, 2.0)]
    plan = plan_blocks([1, 1, 1], [(0, 0), (2, 1)], asr, 3.0, _lf_cfg())
    sus = plan.blocks[1]
    assert sus.suspect and sus.t1 - sus.t0 >= 2 * SUSPECT_MARGIN_S - 1e-9


def test_plan_extra_speech_inside_unit_flags_only():
    asr = [AsrWord(f"w{j}", j * 1.0, j * 1.0 + 0.9) for j in range(10)]
    pairs = [(0, 0), (1, 1), (2, 8), (3, 9)]
    plan = plan_blocks([4], pairs, asr, 10.0, _lf_cfg())
    assert plan.gaps == [Gap(2.0, 7.9, "extra_speech")]
    assert plan.flags == {0: {"gap_inside"}}
    assert len(plan.blocks) == 1


def test_plan_zero_word_unit_skipped():
    plan = plan_blocks([3, 0, 3], _lf_identity(6), _lf_asr(6), 8.0, _lf_cfg())
    assert [(b.first, b.last) for b in plan.blocks] == [(0, 2)]
    plan = plan_blocks([0, 3], _lf_identity(3), _lf_asr(3), 4.0, _lf_cfg())
    assert [(b.first, b.last) for b in plan.blocks] == [(1, 1)]


def test_plan_no_pairs():
    plan = plan_blocks([5, 5], [], _lf_asr(0), 10.0, _lf_cfg())
    assert plan.gaps == [] and plan.flags == {}
    assert plan.blocks == [Block(0, 1, 0.0, 10.0, False)]
    plan = plan_blocks([5, 5], [], [], 100.0, _lf_cfg(block_max_s=60.0, margin_s=1.0))
    assert len(plan.blocks) == 2 and not any(b.suspect for b in plan.blocks)


def test_resolve_midpoint():
    a, b = _lf_unit("a", 0.0, 2.1), _lf_unit("b", 2.0, 4.0)
    resolve_monotonic([a, b], set())
    assert a.end_s == pytest.approx(2.05) and b.start_s == pytest.approx(2.05)
    assert a.words[-1].end == pytest.approx(2.05) and b.words[0].start == pytest.approx(2.05)
    assert "overlap_fixed" in a.flags and "overlap_fixed" in b.flags
    assert "overlap" not in a.flags and "overlap" not in b.flags


def test_resolve_big_overlap_flags_both():
    a, b = _lf_unit("a", 0.0, 3.0), _lf_unit("b", 2.0, 5.0)
    assert 1.0 > OVERLAP_MAX_S
    resolve_monotonic([a, b], set())
    assert a.end_s == b.start_s == pytest.approx(2.5)
    assert "overlap" in a.flags and "overlap" in b.flags


def test_resolve_suspect_loses():
    a, s, c = _lf_unit("a", 0.0, 2.0), _lf_unit("s", 1.5, 3.0), _lf_unit("c", 2.5, 4.0)
    resolve_monotonic([a, s, c], {"s"})
    assert s.start_s is None and s.end_s is None and s.words == [] and s.status == "unmatched"
    assert s.score_mean == s.score_min == 0.0 and "overlap" in s.flags
    assert a.end_s == pytest.approx(2.0) and c.start_s == pytest.approx(2.5)


def test_resolve_suspect_previous_loses_and_comparison_continues():
    a, s, c = _lf_unit("a", 0.0, 2.0), _lf_unit("s", 1.0, 3.5), _lf_unit("c", 2.5, 4.0)
    resolve_monotonic([a, s, c], {"s"})
    assert s.start_s is None
    assert a.end_s == pytest.approx(2.0) and c.start_s == pytest.approx(2.5)


def test_resolve_collapse():
    a, b, c = _lf_unit("a", 0.0, 5.0), _lf_unit("b", 1.0, 2.0), _lf_unit("c", 4.9, 8.0)
    resolve_monotonic([a, b, c], set())
    placed = [u for u in (a, b, c) if u.start_s is not None]
    assert len(placed) < 3
    assert all(u.start_s < u.end_s for u in placed)
    assert all(p.end_s <= q.start_s for p, q in zip(placed, placed[1:]))
    unmatched = [u for u in (a, b, c) if u.start_s is None]
    assert all(u.status == "unmatched" and u.words == [] for u in unmatched)


def test_resolve_unplaced_units_untouched():
    a, u, b = _lf_unit("a", 0.0, 1.0), _lf_unit("u", 0.0, 0.0), _lf_unit("b", 1.0, 2.0)
    u.start_s = u.end_s = None
    u.words, u.status = [], "unmatched"
    resolve_monotonic([a, u, b], set())
    assert a.end_s == 1.0 and b.start_s == 1.0 and u.status == "unmatched"


def test_resolve_random_spans_invariants():
    rng = np.random.default_rng(1)
    for _ in range(200):
        n = int(rng.integers(2, 12))
        units, t = [], 0.0
        for k in range(n):
            start = max(0.0, t + float(rng.normal(0, 1.0)))
            end = start + float(rng.uniform(0.1, 3.0))
            units.append(_lf_unit(f"u{k}", start, end, n_words=int(rng.integers(1, 4))))
            t = end
        suspects = {u.id for u in units if rng.random() < 0.3}
        resolve_monotonic(units, suspects)
        placed = [u for u in units if u.start_s is not None]
        for u in placed:
            assert u.start_s < u.end_s
            assert u.words[0].start >= u.start_s - 1e-9 and u.words[-1].end <= u.end_s + 1e-9
            assert all(w.start <= w.end for w in u.words)
            assert all(p.end <= q.start + 1e-9 for p, q in zip(u.words, u.words[1:]))
        for p, q in zip(placed, placed[1:]):
            assert p.end_s <= q.start_s + 1e-9
        for u in units:
            if u.start_s is None:
                assert u.status == "unmatched" and u.words == []


def _lf_scored_units(scores, edge=None):
    units = []
    for k, s in enumerate(scores):
        u = _lf_unit(f"u{k}", k * 2.0, k * 2.0 + 1.5, score=s)
        if edge is not None and k in edge:
            u.words = [WordSpan("x", k * 2.0, k * 2.0 + 0.5, edge[k]), u.words[1]]
        units.append(u)
    return units


def _lf_counts(units):
    return {s: sum(u.status == s for u in units) for s in ("ok", "low_conf", "unmatched")}


def test_assign_status_distribution():
    rng = np.random.default_rng(2)
    scores = list(0.95 + rng.normal(0, 0.005, 95)) + [0.6] * 3 + [0.2] * 2
    units = _lf_scored_units(scores)
    th = assign_status(units, AlignConfig())
    assert _lf_counts(units) == {"ok": 95, "low_conf": 3, "unmatched": 2}
    assert th["n_placed"] == 100
    assert th["score_median"] == pytest.approx(np.median(scores))
    assert th["ok_min"] == pytest.approx(th["score_median"] - 3.0 * th["score_sigma"])
    assert th["unmatched_max"] == pytest.approx(0.5 * th["score_median"])
    assert set(th) == {"score_median", "score_sigma", "ok_min", "unmatched_max", "edge_median", "edge_sigma",
                       "edge_ok_min", "n_placed"}
    cleared = [u for u in units if u.status == "unmatched"]
    assert all(u.start_s is None and u.words == [] and u.score_mean == 0.0 for u in cleared)


def test_assign_status_sigma_floor():
    units = _lf_scored_units([0.95] * 30 + [0.9])
    th = assign_status(units, AlignConfig())
    assert th["score_sigma"] == pytest.approx(0.02)
    assert units[-1].status == "ok"
    units = _lf_scored_units([0.95] * 30 + [0.85])
    assign_status(units, AlignConfig())
    assert units[-1].status == "low_conf"


def test_assign_status_edge_criterion():
    units = _lf_scored_units([0.95] * 30, edge={4: 0.3})
    th = assign_status(units, AlignConfig())
    assert units[4].status == "low_conf"
    assert th["edge_ok_min"] < th["edge_median"]
    assert _lf_counts(units) == {"ok": 29, "low_conf": 1, "unmatched": 0}


def test_assign_status_flags_force_low_conf():
    units = _lf_scored_units([0.95] * 10)
    units[2].flags.append("overlap")
    units[3].flags.append("no_tokens")
    units[4].flags.append("overlap_fixed")
    assign_status(units, AlignConfig())
    assert [u.status for u in units[2:5]] == ["low_conf", "low_conf", "ok"]


def test_assign_status_keeps_unmatched_and_handles_none_placed():
    u = _lf_scored_units([0.9, 0.9])
    u[1].start_s = u[1].end_s = None
    u[1].words, u[1].status = [], "unmatched"
    th = assign_status(u, AlignConfig())
    assert th["n_placed"] == 1 and u[1].status == "unmatched"
    assert assign_status([], AlignConfig())["n_placed"] == 0


def _lf_gap_units(statuses, flags=None):
    units = []
    for k, st in enumerate(statuses):
        if st == "unmatched":
            units.append(UnitSpan(f"u{k}", None, None, [], 0.0, 0.0, "unmatched", list((flags or {}).get(k, []))))
        else:
            units.append(_lf_unit(f"u{k}", k * 2.0, k * 2.0 + 1.5))
    return units


def test_skipped_gaps():
    units = _lf_gap_units(["ok", "unmatched", "unmatched", "ok", "unmatched"])
    assert skipped_gaps(units, 20.0) == [Gap(1.5, 6.0, "skipped_text"), Gap(7.5, 20.0, "skipped_text")]
    units = _lf_gap_units(["unmatched", "ok"])
    assert skipped_gaps(units, 20.0) == [Gap(0.0, 2.0, "skipped_text")]


def test_skipped_gaps_ignores_empty_units():
    units = _lf_gap_units(["ok", "unmatched", "ok"], flags={1: ["empty"]})
    assert skipped_gaps(units, 20.0) == []


def test_longform_import_is_light():
    code = ("import sys; import src.data.prepare.align.longform; "
            "bad = [m for m in ('torch', 'transformers', 'uroman') if m in sys.modules]; "
            "sys.exit(1 if bad else 0)")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
