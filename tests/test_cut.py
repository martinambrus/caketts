"""Cut stage acceptance harness (spec: T2 cut-spec.md). Synthetic audio only: seeded numpy voice, breath and noise, no wav in git.

Oracle = the truth layout of make_part(); tolerance one 10 ms frame unless stated. Tests call the module through `cut.` so
the negative-control plugin (harness/negcontrol_plugin.py) can swap seams.
"""
import copy
import dataclasses
import functools
import hashlib
import json
import re

import numpy as np
import pytest
import soundfile as sf

from src.data.prepare import cut
from src.data.prepare.cut import CutConfig, CutError, Clip

SR = 24000
F = 240
PAD = 0.15
TOL = 0.03
WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)

SENTENCES = [
    "The kettle began to whistle, and nobody moved.",
    "A narrow path ran along the river.",
    "She counted the steps, slowly, one by one.",
    "Rain tapped on the roof of the shed.",
    "He folded the map, put it away, and sighed.",
    "Nothing in the garden had changed since spring.",
    "The bell rang twice, then fell silent.",
    "They carried the boxes up to the attic.",
    "Morning came late, grey and quiet, over the hills.",
    "A single lamp burned in the window.",
    "The train, already late, slowed near the bridge.",
    "Somebody had left the gate open.",
]


# ---------------------------------------------------------------- synthetic fixture

def _samples(sec):
    return int(round(sec * 100)) * F


def sp(dur, boundary="mid", **kw):
    return {"k": "speech", "dur": dur, "boundary": boundary, **kw}


def pz(dur, kind="quiet", at=0.0):
    return {"k": "pause", "dur": dur, "type": kind, "at": at}


def _rms_scale(x, dbfs):
    return x * (10 ** (dbfs / 20) / np.sqrt(np.mean(x ** 2)))


def _ramp_window(n, ramp):
    w = np.ones(n)
    r = 0.5 * (1 - np.cos(np.pi * np.arange(ramp) / ramp))
    w[:ramp] = r
    w[n - ramp:] = r[::-1]
    return w


def _voice(n, rng):
    t = np.arange(n) / SR
    f0 = 140 + 20 * np.sin(2 * np.pi * 1.5 * t + rng.uniform(0, 2 * np.pi))
    phase = 2 * np.pi * np.cumsum(f0) / SR
    x = sum(np.sin(k * phase) / k for k in range(1, 12))
    x = x * (1 + 0.5 * np.sin(2 * np.pi * 4.5 * t + rng.uniform(0, 2 * np.pi)))
    return _rms_scale(x * _ramp_window(n, int(0.015 * SR)), -20)


def _breath(n, rng):
    spec = np.fft.rfft(rng.standard_normal(n))
    freq = np.fft.rfftfreq(n, 1 / SR)
    spec[(freq < 300) | (freq > 3000)] = 0
    x = np.fft.irfft(spec, n) * _ramp_window(n, int(0.03 * SR))
    return _rms_scale(x, -35)


def make_part(items, seed):
    """-> (pcm int16, truth). truth: speech [(s, e)] per unit (ramp start/end), pause_after {unit idx: {start, end, dur, type, breath}},
    breaths [(start, end)], n."""
    rng = np.random.default_rng(seed)
    lens = [_samples(it["dur"]) for it in items]
    x = rng.standard_normal(sum(lens)) * 10 ** (-60 / 20)
    pos, speech, breaths, pause_after = 0, [], [], {}
    pend = None
    for it, n in zip(items, lens):
        a, b = pos, pos + n
        pos = b
        if it["k"] == "speech":
            x[a:b] += _voice(n, rng)
            if pend is not None:
                pend["end"] = a
                pend["dur"] = (a - pend["start"]) / SR
                pause_after[len(speech) - 1] = pend
                pend = None
            speech.append((a, b))
            continue
        br = None
        if it["type"] == "breath":
            bs = a + _samples(it["at"])
            br = (bs, bs + _samples(0.30))
            x[br[0]:br[1]] += _breath(br[1] - br[0], rng)
            breaths.append(br)
        elif it["type"] == "noise":
            x[a:b] += rng.standard_normal(n) * 10 ** (-30 / 20)
        if speech:
            pend = {"start": a, "end": None, "dur": None, "type": it["type"], "breath": br}
    pcm = np.clip(np.round(x * 32768), -32768, 32767).astype(np.int16)
    return pcm, {"n": len(pcm), "speech": speech, "pause_after": pause_after, "breaths": breaths}


def make_units(items, prefix="p"):
    units, para, chap = [], 0, 0
    for i, it in enumerate([it for it in items if it["k"] == "speech"]):
        text = SENTENCES[i % len(SENTENCES)]
        units.append({"id": f"{prefix}-{i:05d}", "text": text, "boundary": it["boundary"], "est_s": it.get("est_s", 3.0),
                      "paragraph": it.get("paragraph", para), "chapter": it.get("chapter", chap),
                      "heading": it.get("heading", False), "words": WORD_RE.findall(text)})
        if it["boundary"] in ("para_end", "chapter_end"):
            para += 1
        if it["boundary"] == "chapter_end":
            chap += 1
    return units


def stub_spans(units, truth, jitter_s, seed, outlier=None, status=None, missing=(), swap=None, words=True):
    """One aligner span per unit: truth edges + U(-j, +j) with j capped at half the adjacent true pause (so spans never overlap).
    outlier (idx, "start"|"end", delta_s) shifts one edge; status {idx: str}; missing idx are dropped; swap (i, j) exchanges times."""
    rng = np.random.default_rng(seed)
    n, pa = len(units), truth["pause_after"]
    se = []
    for i, (s, e) in enumerate(truth["speech"]):
        jb = jitter_s if i == 0 else min(jitter_s, pa[i - 1]["dur"] / 2)
        ja = jitter_s if i == n - 1 else min(jitter_s, pa[i]["dur"] / 2)
        se.append([s + int(round(rng.uniform(-jb, jb) * SR)), e + int(round(rng.uniform(-ja, ja) * SR))])
    if outlier is not None:
        idx, side, delta = outlier
        se[idx][0 if side == "start" else 1] += int(round(delta * SR))
    if swap is not None:
        i, j = swap
        se[i], se[j] = se[j], se[i]
    spans = []
    for i, (u, (s, e)) in enumerate(zip(units, se)):
        if i in missing:
            continue
        sp_ = {"id": u["id"], "start_s": s / SR, "end_s": e / SR, "status": (status or {}).get(i, "ok")}
        if words and u["words"]:
            step = (e - s) / SR / len(u["words"])
            sp_["words"] = [{"text": w, "start": s / SR + k * step, "end": s / SR + (k + 0.7) * step, "score": 0.9}
                            for k, w in enumerate(u["words"])]
        spans.append(sp_)
    return spans


STD_SPEECH = [1.2, 1.8, 2.2, 1.5, 2.6, 3.0, 1.4, 2.0, 1.1, 2.4, 1.7, 2.8, 1.3, 2.0]
STD_PAUSES = [(0.80, "quiet", 0.0, "mid"), (0.12, "quiet", 0.0, "mid"), (1.50, "quiet", 0.0, "mid"),
              (0.30, "quiet", 0.0, "mid"), (1.20, "breath", 0.5, "mid"), (2.00, "quiet", 0.0, "para_end"),
              (0.40, "quiet", 0.0, "mid"), (0.80, "noise", 0.0, "mid"), (1.00, "quiet", 0.0, "mid"),
              (0.60, "quiet", 0.0, "mid"), (3.00, "quiet", 0.0, "chapter_end"), (0.90, "quiet", 0.0, "mid"),
              (1.50, "quiet", 0.0, "mid")]
NOISE_UNIT = 7
BREATH_UNIT = 4
SHORT_UNIT = 1


def std_items():
    items = [pz(0.5)]
    for i, d in enumerate(STD_SPEECH):
        last = i == len(STD_SPEECH) - 1
        items.append(sp(d, "chapter_end" if last else STD_PAUSES[i][3], est_s=3.0 + i % 3))
        items.append(pz(0.5) if last else pz(STD_PAUSES[i][0], STD_PAUSES[i][1], STD_PAUSES[i][2]))
    return items


@functools.lru_cache(maxsize=None)
def std():
    items = std_items()
    pcm, truth = make_part(items, 11)
    return {"items": items, "pcm": pcm, "truth": truth, "units": make_units(items)}


@functools.lru_cache(maxsize=None)
def std_plan(jitter, outlier=None, seed=3):
    d = std()
    spans = stub_spans(d["units"], d["truth"], jitter, seed, outlier=outlier)
    return cut.plan_part("p", d["pcm"], SR, d["units"], spans, CutConfig())


def frame_db_np(pcm):
    n = len(pcm) // F
    x = pcm[:n * F].astype(np.float64).reshape(n, F) / 32768
    return 10 * np.log10(np.mean(x ** 2, axis=1) + 1e-10)


def clip_of(plan, units, i):
    return next(c for c in plan.clips if units[i]["id"] in c.units)


def clip_pairs(plan, units):
    """(clip, next clip, i) for neighbouring clips whose units are list-adjacent; i = unit index of the first clip's last unit."""
    idx = {u["id"]: k for k, u in enumerate(units)}
    cl = sorted(plan.clips, key=lambda c: c.start_sample)
    for c1, c2 in zip(cl, cl[1:]):
        i, j = idx[c1.units[-1]], idx[c2.units[0]]
        if j == i + 1:
            yield c1, c2, i


def assert_holds_true_speech(plan, truth, units):
    idx = {u["id"]: k for k, u in enumerate(units)}
    for c in plan.clips:
        for uid in c.units:
            s, e = truth["speech"][idx[uid]]
            assert c.start_sample <= s + F, f"clip {c.id} starts {c.start_sample - s} samples after true speech start of {uid}"
            assert c.end_sample >= e - F, f"clip {c.id} ends {e - c.end_sample} samples before true speech end of {uid}"


# ---------------------------------------------------------------- fixture sanity (independent numpy, passes on the skeleton)

def test_fixture_energy_levels():
    d = std()
    pcm, truth = d["pcm"], d["truth"]
    assert pcm.dtype == np.int16
    db = frame_db_np(pcm)
    floor = np.percentile(db, 5)
    lead = db[:40]
    assert np.all(np.abs(lead - (-60)) < 2.5), (lead.min(), lead.max())
    for s, e in truth["speech"]:
        inner = db[(s + 15 * 24 + F - 1) // F + 1:(e - 15 * 24 - F) // F]
        assert inner.size > 50 and inner.min() > -35, inner.min()
    for bs, be in truth["breaths"]:
        inner = db[(bs + 30 * 24) // F + 1:(be - 30 * 24) // F - 1]
        assert inner.size > 10 and inner.min() > floor + 12
    p = truth["pause_after"][NOISE_UNIT]
    inner = db[p["start"] // F + 1:p["end"] // F - 1]
    assert inner.min() > floor + 12
    quiet = truth["pause_after"][0]
    q = db[quiet["start"] // F + 1:quiet["end"] // F - 1]
    assert np.all(np.abs(q - (-60)) < 2.5)
    speech_n = sum(e - s for s, e in truth["speech"])
    assert 1 - speech_n / len(pcm) >= 0.35
    assert len(truth["speech"]) == 14 and len(truth["breaths"]) == 1


def test_fixture_is_deterministic():
    a, _ = make_part(std_items(), 11)
    assert np.array_equal(a, std()["pcm"])
    b, _ = make_part(std_items(), 12)
    assert not np.array_equal(a, b)


def test_config_defaults_match_spec():
    c = CutConfig()
    assert (c.pad_s, c.frame_s, c.search_s, c.min_quiet_s, c.quiet_db, c.speech_db) == (0.15, 0.010, 0.08, 0.04, 15.0, 15.0)
    assert (c.floor_percentile, c.max_est_s, c.max_clip_s, c.min_split_words, c.close_s, c.bridge_s) == (5.0, 17.0, 20.0, 3, 0.03, 0.35)


def test_trivial_functions():
    assert cut.to_sample(0.15, SR) == 3600 and cut.to_sample(1.0 / 3, SR) == 8000
    assert cut.clip_bounds(10000, 20000, 9000, 21000, 3600) == (9000, 21000)
    assert cut.clip_bounds(10000, 20000, 0, 99999, 3600) == (6400, 23600)
    assert cut.GAP_TYPES == {"mid": "sentence", "para_end": "paragraph", "chapter_end": "chapter"}
    units, spans = [{"id": "x"}], [{"id": "x", "start_s": 0.0, "end_s": 1.0}]
    assert cut.split_over_cap(units, spans, CutConfig(), SR) == (units, spans)


# ---------------------------------------------------------------- 1-3 clip edges, pads, speech containment

def _true_pair_pause(truth, i):
    p = truth["pause_after"][i]
    return p if p["dur"] >= 0.32 else None


@pytest.mark.parametrize("jitter", [0.02, 0.08], ids=["jit20ms", "jit80ms"])
def test_clip_edges_inside_true_pauses(jitter):
    d, plan = std(), std_plan(jitter)
    checked = 0
    for c1, c2, i in clip_pairs(plan, d["units"]):
        p = _true_pair_pause(d["truth"], i)
        if p is None:
            continue
        checked += 1
        assert p["start"] <= c1.end_sample <= p["end"], f"{c1.id}: end {c1.end_sample} outside pause {p['start']}..{p['end']}"
        assert p["start"] <= c2.start_sample <= p["end"], f"{c2.id}: start {c2.start_sample} outside pause {p['start']}..{p['end']}"
    assert checked >= 10
    for bs, be in d["truth"]["breaths"]:
        for c in plan.clips:
            for edge in (c.start_sample, c.end_sample):
                assert not (bs < edge < be), f"clip {c.id} edge {edge} cuts the breath {bs}..{be}"


@pytest.mark.parametrize("jitter", [0.02, 0.06], ids=["jit20ms", "jit60ms"])
def test_uniform_pads_in_pauses(jitter):
    d, plan = std(), std_plan(jitter)
    checked = 0
    for c1, c2, i in clip_pairs(plan, d["units"]):
        p = _true_pair_pause(d["truth"], i)
        if p is None or p["type"] == "noise":
            continue
        checked += 1
        trail, lead = (c1.end_sample - p["start"]) / SR, (p["end"] - c2.start_sample) / SR
        assert abs(trail - PAD) <= TOL, f"{c1.id}: trail pad {trail:.3f}"
        assert abs(lead - PAD) <= TOL, f"{c2.id}: lead pad {lead:.3f}"
    assert checked >= 9


@pytest.mark.parametrize("outlier,pair", [((2, "end", 0.10), 2), ((5, "start", -0.10), 4)],
                         ids=["end_100ms_into_pause", "start_100ms_into_pause"])
def test_wrong_aligner_edge_cut_stays_in_true_pause(outlier, pair):
    d, plan = std(), std_plan(0.02, outlier)
    found = [(c1, c2) for c1, c2, i in clip_pairs(plan, d["units"]) if i == pair]
    assert found
    c1, c2 = found[0]
    p = d["truth"]["pause_after"][pair]
    assert p["start"] <= c1.end_sample <= p["end"] and p["start"] <= c2.start_sample <= p["end"]


@pytest.mark.parametrize("outlier,pair", [((2, "end", -0.20), 2), ((5, "start", 0.20), 4)],
                         ids=["end_200ms_inside_speech", "start_200ms_inside_speech"])
def test_pads_stay_uniform_when_the_aligner_edge_is_far_inside_the_speech(outlier, pair):
    d, plan = std(), std_plan(0.02, outlier)
    found = [(c1, c2) for c1, c2, i in clip_pairs(plan, d["units"]) if i == pair]
    assert found
    c1, c2 = found[0]
    p = d["truth"]["pause_after"][pair]
    assert abs((c1.end_sample - p["start"]) / SR - PAD) <= TOL
    assert abs((p["end"] - c2.start_sample) / SR - PAD) <= TOL
    assert_holds_true_speech(plan, d["truth"], d["units"])


@pytest.mark.parametrize("jitter,outlier", [(0.0, None), (0.02, None), (0.06, None), (0.08, None),
                                            (0.02, (2, "end", 0.10)), (0.02, (5, "start", -0.10)),
                                            (0.02, (8, "end", -0.10)), (0.02, (6, "start", 0.10))],
                         ids=["j0", "j20", "j60", "j80", "end_out", "start_out", "end_in_speech", "start_in_speech"])
def test_every_clip_contains_its_true_speech(jitter, outlier):
    d = std()
    assert_holds_true_speech(std_plan(jitter, outlier), d["truth"], d["units"])


# ---------------------------------------------------------------- seams on hand-made frame energies (spec 3 and 5)

def _db(n, loud=(), quiet_level=-60.0):
    db = np.full(n, quiet_level)
    for a, b in loud:
        db[a:b] = -20.0
    return db


def test_refine_edges_snap_to_speech_frames():
    cfg, db = CutConfig(), _db(300, [(0, 100), (150, 300)])
    assert cut.refine_gap(db, -60.0, 103 * F, 147 * F, cfg, SR) == (100 * F, 150 * F)
    assert cut.refine_gap(db, -60.0, 99 * F, 151 * F, cfg, SR) == (100 * F, 150 * F)
    assert cut.refine_gap(db, -60.0, 120 * F, 130 * F, cfg, SR) == (120 * F, 130 * F)
    assert cut.refine_end(db, -60.0, 104 * F, cfg, SR) == 100 * F
    assert cut.refine_end(db, -60.0, 125 * F, cfg, SR) == 125 * F
    assert cut.refine_start(db, -60.0, 146 * F, cfg, SR) == 150 * F
    assert cut.refine_start(db, -60.0, 125 * F, cfg, SR) == 125 * F


def test_refine_gap_never_crosses_the_midpoint():
    db = _db(300, [(0, 300)])
    e_ref, s_ref = cut.refine_gap(db, -60.0, 100 * F, 110 * F, CutConfig(), SR)
    assert e_ref <= s_ref and e_ref <= 105 * F + F and s_ref >= 105 * F - F


def test_refine_follows_a_decay_beyond_the_search_window():
    cfg, db = CutConfig(), _db(300, [(0, 100)])
    db[100:130] = -40.0
    assert cut.refine_end(db, -60.0, 101 * F, cfg, SR) == 130 * F
    assert cut.refine_gap(db, -60.0, 101 * F, 200 * F, cfg, SR)[0] == 130 * F
    onset = _db(300, [(150, 300)])
    onset[120:150] = -40.0
    assert cut.refine_start(onset, -60.0, 149 * F, cfg, SR) == 120 * F
    assert cut.refine_gap(onset, -60.0, 50 * F, 149 * F, cfg, SR)[1] == 120 * F


def test_refine_bridges_a_dip_of_two_frames_but_not_three():
    cfg = CutConfig(quiet_db=12.0, bridge_s=0.0)
    assert cut.refine_end(_db(300, [(0, 100), (102, 110)]), -60.0, 99 * F, cfg, SR) == 110 * F
    assert cut.refine_end(_db(300, [(0, 100), (103, 110)]), -60.0, 99 * F, cfg, SR) == 100 * F
    assert cut.refine_start(_db(300, [(190, 300), (180, 188)]), -60.0, 191 * F, cfg, SR) == 180 * F
    assert cut.refine_start(_db(300, [(190, 300), (180, 187)]), -60.0, 191 * F, cfg, SR) == 190 * F


def test_bridge_defaults():
    c = CutConfig()
    assert c.quiet_db == 15.0 and c.bridge_s == 0.35


def _hand(segs, seed=4):
    """-> pcm int16 and the segment bounds in samples; segs are (kind, dur) with kind floor, voice or breath."""
    rng = np.random.default_rng(seed)
    lens = [_samples(d) for _, d in segs]
    x = rng.standard_normal(sum(lens)) * 10 ** (-60 / 20)
    pos, bounds = 0, []
    for (kind, _), n in zip(segs, lens):
        if kind == "voice":
            x[pos:pos + n] += _voice(n, rng)
        elif kind == "breath":
            x[pos:pos + n] += _breath(n, rng)
        bounds.append((pos, pos + n))
        pos += n
    return np.clip(np.round(x * 32768), -32768, 32767).astype(np.int16), bounds


def _plan_hand(pcm, edges, cfg):
    """edges: aligner (start, end) in samples per unit."""
    units = make_units([sp(1.0)] * (len(edges) - 1) + [sp(1.0, "para_end")])
    truth = {"speech": edges, "pause_after": {i: {"dur": (edges[i + 1][0] - edges[i][1]) / SR} for i in range(len(edges) - 1)}}
    return cut.plan_part("p", pcm, SR, units, stub_spans(units, truth, 0.0, 1), cfg), units


def _late_word_part():
    pcm, b = _hand([("floor", 0.6), ("voice", 0.3), ("floor", 0.12), ("voice", 2.0), ("floor", 1.0), ("voice", 1.5), ("floor", 0.6)])
    return pcm, b, [(b[3][0] + int(0.02 * SR), b[3][1]), (b[5][0], b[5][1])]


def test_late_first_word_is_inside_the_clip():
    pcm, b, edges = _late_word_part()
    plan, _ = _plan_hand(pcm, edges, CutConfig())
    c = plan.clips[0]
    assert abs(c.speech_start - b[1][0]) <= F
    assert c.start_sample <= b[1][0] and abs(c.pad_start_s - PAD) <= 0.01


def test_late_first_word_is_cut_off_without_bridging():
    pcm, b, edges = _late_word_part()
    plan, _ = _plan_hand(pcm, edges, CutConfig(bridge_s=0.0))
    assert plan.clips[0].start_sample > b[1][0]


def test_late_start_inside_a_breath_clips_start_before_the_breath():
    pcm, b = _hand([("floor", 0.6), ("breath", 0.3), ("floor", 0.1), ("voice", 2.0), ("floor", 1.0), ("voice", 1.5), ("floor", 0.6)])
    plan, _ = _plan_hand(pcm, [(b[3][0] + int(0.02 * SR), b[3][1]), (b[5][0], b[5][1])], CutConfig())
    c = plan.clips[0]
    db = frame_db_np(pcm)
    assert abs(c.start_sample - (b[1][0] - int(PAD * SR))) <= 4 * F
    assert db[c.start_sample // F] <= np.percentile(db, 5) + 15.0


def test_bridging_both_sides_into_a_breath_keeps_a_quiet_cut():
    pcm, b = _hand([("floor", 0.6), ("voice", 1.5), ("floor", 0.2), ("breath", 0.3), ("floor", 0.2), ("voice", 1.5), ("floor", 0.6)])
    plan, _ = _plan_hand(pcm, [(b[1][0], b[1][1]), (b[5][0], b[5][1])], CutConfig())
    db = frame_db_np(pcm)
    assert len(plan.clips) == 2 and not any("no_quiet_cut" in c.flags for c in plan.clips)
    assert db[plan.clips[0].end_sample // F] <= np.percentile(db, 5) + 15.0
    assert plan.clips[0].end_sample <= plan.clips[1].start_sample


def test_bridge_stays_on_its_side_of_the_midpoint():
    cfg, thr = CutConfig(), -45.0
    db = _db(300, [(0, 100), (121, 130), (140, 300)])
    assert cut._extend(db, thr, 99, 3, 120, True, 35) == 99
    assert cut.refine_gap(db, -60.0, 100 * F, 140 * F, cfg, SR) == (100 * F, 121 * F)
    near_e = _db(300, [(0, 100), (125, 135), (200, 300)])
    assert cut.refine_gap(near_e, -60.0, 100 * F, 200 * F, cfg, SR) == (135 * F, 200 * F)
    near_s = _db(300, [(0, 100), (165, 175), (200, 300)])
    assert cut.refine_gap(near_s, -60.0, 100 * F, 200 * F, cfg, SR) == (100 * F, 165 * F)
    assert cut._extend(near_s, thr, 99, 3, 150, True, 60) == 99
    assert cut._extend(db, thr, 99, 0, 120, True, 35) == 99


def test_bridge_needs_a_burst_not_a_blip():
    db = _db(300, [(0, 100), (120, 122)])
    assert cut.refine_end(db, -60.0, 99 * F, CutConfig(), SR) == 100 * F
    assert cut.refine_end(_db(300, [(0, 100), (120, 123)]), -60.0, 99 * F, CutConfig(), SR) == 123 * F


def test_bridge_fallback_prefers_the_start_side_bridge():
    db = _db(400, [(0, 100), (110, 190), (200, 300)])
    assert cut.refine_gap(db, -60.0, 100 * F, 200 * F, CutConfig(), SR) == (100 * F, 150 * F)


def test_bridge_applies_at_the_first_start_and_the_last_end():
    cfg, off = CutConfig(), CutConfig(bridge_s=0.0)
    first = _db(300, [(0, 30), (60, 300)])
    assert cut.refine_start(first, -60.0, 62 * F, cfg, SR) == 0
    assert cut.refine_start(first, -60.0, 62 * F, off, SR) == 60 * F
    last = _db(300, [(0, 100), (130, 160)])
    assert cut.refine_end(last, -60.0, 99 * F, cfg, SR) == 160 * F
    assert cut.refine_end(last, -60.0, 99 * F, off, SR) == 100 * F


@pytest.mark.parametrize("loud,lo,hi,expect", [
    ([(0, 20), (26, 40), (50, 100)], 15 * F, 60 * F, 45 * F),
    ([(0, 20), (26, 40), (46, 100)], 0, 100 * F, 23 * F),
    ([(0, 20), (23, 40), (43, 100)], 0, 100 * F, None),
    ([(0, 100)], 0, 100 * F, None),
    ([(0, 50), (90, 100)], 0, 50 * F, None),
    ([(0, 20), (28, 100)], 22 * F + 10, 100 * F, (23 * F + 28 * F) // 2),
], ids=["longest", "tie_earliest", "runs_under_40ms", "all_speech_level", "run_outside_range", "whole_frames_only"])
def test_choose_cut_picks_the_centre_of_the_longest_quiet_run(loud, lo, hi, expect):
    assert cut.choose_cut(_db(100, loud), -60.0, lo, hi, CutConfig(), SR) == expect


def test_choose_cut_ignores_breath_level_frames():
    db = _db(100, [(0, 20), (60, 100)])
    db[35:45] = -40.0
    c = cut.choose_cut(db, -60.0, 20 * F, 60 * F, CutConfig(), SR)
    assert c is not None and not (35 * F <= c < 45 * F)


# ---------------------------------------------------------------- 4 short pauses and merging

def _two(boundary1="mid", gap=0.12, est=(3.0, 3.0), second=None, first=None):
    items = [pz(0.5), sp(1.5, boundary1, est_s=est[0], **(first or {})), pz(gap),
             sp(1.5, "para_end", est_s=est[1], **(second or {})), pz(0.5)]
    pcm, truth = make_part(items, 5)
    units = make_units(items)
    return pcm, truth, units, stub_spans(units, truth, 0.02, 1)


def test_short_pause_merges_into_one_clip():
    pcm, truth, units, spans = _two()
    plan = cut.plan_part("p", pcm, SR, units, spans, CutConfig())
    assert len(plan.clips) == 1
    c = plan.clips[0]
    assert c.units == [units[0]["id"], units[1]["id"]] and c.id == units[0]["id"]
    assert c.text == units[0]["text"] + " " + units[1]["text"]
    assert c.boundary == "para_end"
    assert len(plan.gaps) == 1 and plan.gaps[0][0] == "sentence" and abs(plan.gaps[0][1] - 0.12) <= TOL
    assert_holds_true_speech(plan, truth, units)


@pytest.mark.parametrize("kw", [
    {"boundary1": "para_end", "second": {"paragraph": 0, "chapter": 0}},
    {"second": {"paragraph": 7}},
    {"first": {"heading": True}},
    {"second": {"heading": True}},
    {"est": (9.0, 9.0)},
], ids=["para_end", "paragraph_differs", "first_heading", "second_heading", "est_over_17"])
def test_short_pause_not_mergeable_gives_two_touching_clips(kw):
    pcm, truth, units, spans = _two(**kw)
    plan = cut.plan_part("p", pcm, SR, units, spans, CutConfig())
    assert [c.units for c in plan.clips] == [[units[0]["id"]], [units[1]["id"]]]
    c1, c2 = plan.clips
    assert "short_pause" in c1.flags and "short_pause" in c2.flags
    assert c1.end_sample == c2.start_sample
    p = truth["pause_after"][0]
    assert p["start"] <= c1.end_sample <= p["end"]
    assert abs(c1.pad_end_s - 0.06) <= TOL and abs(c2.pad_start_s - 0.06) <= TOL
    assert len(plan.gaps) == 1 and abs(plan.gaps[0][1] - 0.12) <= TOL
    assert_holds_true_speech(plan, truth, units)


# ---------------------------------------------------------------- 5 no quiet run

def test_noise_pause_flags_no_quiet_cut_and_has_no_gap():
    d, plan = std(), std_plan(0.02)
    c1, c2 = clip_of(plan, d["units"], NOISE_UNIT), clip_of(plan, d["units"], NOISE_UNIT + 1)
    assert "no_quiet_cut" in c1.flags and "no_quiet_cut" in c2.flags
    assert c1.end_sample <= c2.start_sample
    assert c1.gap_after_s is None and c2.gap_before_s is None
    assert [c.id for c in plan.clips if "no_quiet_cut" in c.flags] == [c1.id, c2.id]
    assert len(plan.gaps) == len(d["units"]) - 1 - 1


# ---------------------------------------------------------------- 6 rejects and barriers

def test_rejects_and_barriers():
    items = [pz(0.5)]
    for i in range(9):
        items += [sp(1.5, est_s=3.0), pz(0.8) if i < 8 else pz(0.5)]
    pcm, truth = make_part(items, 21)
    units = make_units(items)
    spans = stub_spans(units, truth, 0.02, 2, status={2: "low_conf", 5: "unmatched"}, missing=(7,))
    plan = cut.plan_part("p", pcm, SR, units, spans, CutConfig())
    assert {r["id"]: r["reason"] for r in plan.rejects} == {units[2]["id"]: "low_conf", units[5]["id"]: "unmatched",
                                                            units[7]["id"]: "no_span"}
    listed = {u for c in plan.clips for u in c.units}
    assert listed == {units[i]["id"] for i in (0, 1, 3, 4, 6, 8)}
    for barrier, before, after in ((2, 1, 3), (5, 4, 6)):
        s, e = truth["speech"][barrier]
        assert clip_of(plan, units, before).end_sample <= s
        assert clip_of(plan, units, after).start_sample >= e
    g = {i: clip_of(plan, units, i) for i in (0, 1, 3, 4, 6, 8)}
    assert g[0].gap_before_s is None and g[8].gap_after_s is None
    assert abs(g[0].gap_after_s - 0.8) <= TOL and abs(g[1].gap_before_s - 0.8) <= TOL
    assert abs(g[3].gap_after_s - 0.8) <= TOL and abs(g[4].gap_before_s - 0.8) <= TOL
    assert g[1].gap_after_s is None and g[3].gap_before_s is None
    assert g[4].gap_after_s is None and g[6].gap_before_s is None
    assert g[6].gap_after_s is None and g[8].gap_before_s is None
    assert_holds_true_speech(plan, truth, units)


# ---------------------------------------------------------------- 7 span validation

def _four():
    items = [pz(0.5)] + [x for _ in range(4) for x in (sp(1.5), pz(0.8))]
    pcm, truth = make_part(items, 31)
    units = make_units(items)
    return pcm, units, stub_spans(units, truth, 0.0, 1)


def _m_overlap(sp_):
    sp_[1]["start_s"] = (cut.to_sample(sp_[0]["end_s"], SR) - 1) / SR


def _m_duplicate(sp_):
    sp_[1]["start_s"], sp_[1]["end_s"] = sp_[0]["start_s"], sp_[0]["end_s"]


def _m_swapped(sp_):
    for k in ("start_s", "end_s"):
        sp_[1][k], sp_[2][k] = sp_[2][k], sp_[1][k]


def _m_same_id(sp_):
    sp_.append(copy.deepcopy(sp_[0]))


def _m_unknown(sp_):
    sp_.append({"id": "nope-99999", "start_s": 20.0, "end_s": 21.0, "status": "ok"})


def _m_empty(sp_):
    sp_[1]["end_s"] = sp_[1]["start_s"]


def _m_negative(sp_):
    sp_[1]["end_s"] = sp_[1]["start_s"] - 0.1


@pytest.mark.parametrize("via", ["validate_spans", "plan_part"])
@pytest.mark.parametrize("mutate,names", [
    (_m_overlap, ["p-00000", "p-00001"]), (_m_duplicate, ["p-00000", "p-00001"]), (_m_swapped, ["p-00001", "p-00002"]),
    (_m_same_id, ["p-00000"]), (_m_unknown, ["nope-99999"]), (_m_empty, ["p-00001"]), (_m_negative, ["p-00001"]),
], ids=["overlap_1_sample", "duplicate_span", "swapped", "same_id_twice", "unknown_id", "end_eq_start", "end_lt_start"])
def test_span_validation_raises_naming_the_ids(via, mutate, names):
    pcm, units, spans = _four()
    mutate(spans)
    call = (lambda: cut.validate_spans("p", units, spans, SR)) if via == "validate_spans" else \
           (lambda: cut.plan_part("p", pcm, SR, units, spans, CutConfig()))
    with pytest.raises(CutError) as ei:
        call()
    for n in names:
        assert n in str(ei.value)


def test_touching_spans_are_valid():
    pcm, units, spans = _four()
    spans[1]["start_s"] = spans[0]["end_s"]
    seq, rej = cut.validate_spans("p", units, spans, SR)
    assert len(seq) == 4 and rej == []
    assert all(it.ok for it in seq)
    assert seq[1].s == seq[0].e == cut.to_sample(spans[0]["end_s"], SR)
    cut.plan_part("p", pcm, SR, units, spans, CutConfig())


def test_a_late_touching_aligner_end_never_makes_the_gap_negative():
    items = [pz(0.5), sp(1.5, "para_end"), pz(1.0), sp(1.5), pz(0.5)]
    pcm, truth = make_part(items, 71)
    units = make_units(items)
    touch = truth["speech"][0][1] + SR + 103
    spans = [{"id": units[0]["id"], "start_s": truth["speech"][0][0] / SR, "end_s": touch / SR, "status": "ok"},
             {"id": units[1]["id"], "start_s": touch / SR, "end_s": truth["speech"][1][1] / SR, "status": "ok"}]
    plan = cut.plan_part("p", pcm, SR, units, spans, CutConfig())
    assert len(plan.clips) == 2 and plan.clips[0].end_sample <= plan.clips[1].start_sample
    assert_holds_true_speech(plan, truth, units)


# ---------------------------------------------------------------- 8 hard assertions

def _clip(cid, a, b, ss, se, als=None, ale=None):
    return Clip(id=cid, units=[cid], text="x", boundary="mid", part="p", start_sample=a, end_sample=b, speech_start=ss,
                speech_end=se, al_start=ss if als is None else als, al_end=se if ale is None else ale, pad_start_s=0.0,
                pad_end_s=0.0, gap_before_s=None, gap_after_s=None, flags=[], sha256="0" * 64)


def _valid_clips():
    return [_clip("c0", 1000, 5000, 1500, 4500), _clip("c1", 5000, 9000, 5500, 8500, als=4900, ale=8600)]


def test_check_clips_passes_a_valid_set():
    cut.check_clips(_valid_clips(), 10000, 100)


@pytest.mark.parametrize("bad", [
    _clip("bad", 6000, 6000, 6000, 6000), _clip("bad", 6500, 6000, 6500, 6000),
    _clip("bad", 4999, 9000, 5500, 8500), _clip("bad", 5000, 9000, 4990, 8500),
    _clip("bad", 5000, 9000, 5500, 9100), _clip("bad", 5000, 9000, 5500, 8500, als=4800),
    _clip("bad", 5000, 9000, 5500, 8500, ale=9200), _clip("bad", -1, 5000, 1500, 4500),
    _clip("bad", 5000, 10001, 5500, 8500),
], ids=["a_eq_b", "a_gt_b", "overlap_1_sample", "start_after_speech_start", "end_before_speech_end",
        "aligner_start_beyond_tol", "aligner_end_beyond_tol", "negative", "past_end"])
def test_check_clips_raises_naming_the_clip(bad):
    clips = [_valid_clips()[0], bad]
    with pytest.raises(CutError) as ei:
        cut.check_clips(clips, 10000, 100)
    assert "bad" in str(ei.value)


def test_check_clips_raises_when_lengths_exceed_the_part():
    clips = [_clip("big0", 0, 10000, 100, 9900), _clip("big1", 0, 10000, 100, 9900)]
    with pytest.raises(CutError):
        cut.check_clips(clips, 10000, 100)


def test_check_unique_hashes():
    cut.check_unique_hashes([("dup-a", "h1"), ("dup-b", "h2")])
    with pytest.raises(CutError) as ei:
        cut.check_unique_hashes([("dup-a", "h1"), ("dup-b", "h1")])
    assert "dup-a" in str(ei.value) and "dup-b" in str(ei.value)


# ---------------------------------------------------------------- 9 duplicate audio

def test_duplicate_audio_in_one_part_raises():
    block_items = [pz(0.5), sp(2.0), pz(0.5)]
    blk, truth = make_part(block_items, 41)
    units = make_units([sp(2.0), sp(2.0)])
    L = len(blk)
    s, e = truth["speech"][0]
    tt = {"speech": [(s, e), (s + L, e + L)], "pause_after": {0: {"dur": 1.0}}}
    spans = stub_spans(units, tt, 0.0, 1)
    with pytest.raises(CutError) as ei:
        cut.plan_part("p", np.concatenate([blk, blk]), SR, units, spans, CutConfig())
    assert units[0]["id"] in str(ei.value) and units[1]["id"] in str(ei.value)
    other, _ = make_part(block_items, 42)
    plan = cut.plan_part("p", np.concatenate([blk, other]), SR, units, spans, CutConfig())
    assert len(plan.clips) == 2


# ---------------------------------------------------------------- 10 determinism and 13 run_cut

def write_work(root, parts):
    for sub in ("text", "align", "audio"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    for name, (pcm, units, spans) in parts.items():
        (root / "text" / f"{name}.units.jsonl").write_text("".join(json.dumps(u) + "\n" for u in units))
        (root / "align" / f"{name}.align.jsonl").write_text("".join(json.dumps(s) + "\n" for s in spans))
        sf.write(str(root / "audio" / f"{name}.24k.wav"), pcm, SR, subtype="PCM_16")


def small_part(prefix, seed, n=5, status=None):
    items = [pz(0.5)]
    for i in range(n):
        items += [sp(1.2 + 0.3 * (i % 3), est_s=3.0), pz(0.12 if (i == 0 and prefix == "a") else 0.8) if i < n - 1 else pz(0.5)]
    pcm, truth = make_part(items, seed)
    units = make_units(items, prefix)
    return pcm, units, stub_spans(units, truth, 0.02, seed, status=status)


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted((root / "cut").rglob("*"))
            if p.is_file() and not p.name.startswith("report")}


def test_plan_part_is_deterministic():
    d = std()
    spans = stub_spans(d["units"], d["truth"], 0.04, 7)
    a = cut.plan_part("p", d["pcm"], SR, d["units"], spans, CutConfig())
    b = cut.plan_part("p", d["pcm"], SR, d["units"], spans, CutConfig())
    assert dataclasses.asdict(a) == dataclasses.asdict(b)


def test_run_cut_is_byte_identical_on_rerun(tmp_path):
    write_work(tmp_path, {"pa": small_part("a", 1), "pb": small_part("b", 2, status={2: "low_conf"})})
    cut.run_cut(tmp_path, ["pa", "pb"], CutConfig())
    first = snapshot(tmp_path)
    cut.run_cut(tmp_path, ["pa", "pb"], CutConfig())
    assert first and snapshot(tmp_path) == first


CUT_KEYS = {"id", "units", "text", "boundary", "part", "start_sample", "end_sample", "pad_start_s", "pad_end_s",
            "gap_before_s", "gap_after_s", "flags", "sha256"}


def test_run_cut_round_trip(tmp_path):
    pa, pb = small_part("a", 1), small_part("b", 2, status={2: "low_conf"})
    write_work(tmp_path, {"pa": pa, "pb": pb})
    cut.run_cut(tmp_path, ["pa", "pb"], CutConfig())
    out, ids = tmp_path / "cut", set()
    for name, (pcm, units, _) in (("pa", pa), ("pb", pb)):
        lines = (out / f"{name}.cuts.jsonl").read_text().splitlines()
        recs = [json.loads(x) for x in lines]
        assert recs and [r["start_sample"] for r in recs] == sorted(r["start_sample"] for r in recs)
        for r in recs:
            assert set(r) == CUT_KEYS and list(r) == sorted(r) and r["part"] == name
            ids.add(r["id"])
            wav = out / "clips" / f"{r['id']}.wav"
            assert sf.info(str(wav)).subtype == "PCM_16" and sf.info(str(wav)).samplerate == SR
            data, sr = sf.read(str(wav), dtype="int16")
            assert sr == SR and data.ndim == 1
            assert np.array_equal(data, pcm[r["start_sample"]:r["end_sample"]])
            assert r["sha256"] == hashlib.sha256(pcm[r["start_sample"]:r["end_sample"]].tobytes()).hexdigest()
    assert {p.stem for p in (out / "clips").glob("*.wav")} == ids
    first = json.loads((out / "pa.cuts.jsonl").read_text().splitlines()[0])
    assert first["units"] == [pa[1][0]["id"], pa[1][1]["id"]] and first["id"] == pa[1][0]["id"]
    rej = [json.loads(x) for x in (out / "pb.rejects.jsonl").read_text().splitlines()]
    assert rej == [{"id": pb[1][2]["id"], "reason": "low_conf"}]
    assert (out / "pa.rejects.jsonl").read_text().strip() == ""
    pauses = json.loads((out / "pauses.json").read_text())
    assert {"comma", "sentence", "paragraph", "chapter", "frame_s", "pad_s"} <= set(pauses)
    assert pauses["frame_s"] == 0.01 and pauses["pad_s"] == 0.15
    assert all(set(pauses[k]) == {"n", "median_s", "p10_s", "p90_s"} for k in ("comma", "sentence", "paragraph", "chapter"))
    json.loads((out / "report.json").read_text())
    assert (out / "report.md").read_text().strip()


def test_run_cut_removes_the_clip_file_of_a_vanished_unit(tmp_path):
    pcm, units, spans = small_part("a", 1)
    write_work(tmp_path, {"pa": (pcm, units, spans)})
    cut.run_cut(tmp_path, ["pa"], CutConfig())
    gone = units[-1]["id"]
    assert (tmp_path / "cut" / "clips" / f"{gone}.wav").exists()
    write_work(tmp_path, {"pa": (pcm, units[:-1], spans[:-1])})
    cut.run_cut(tmp_path, ["pa"], CutConfig())
    assert not (tmp_path / "cut" / "clips" / f"{gone}.wav").exists()
    assert (tmp_path / "cut" / "clips" / f"{units[0]['id']}.wav").exists()


def test_run_cut_needs_every_part_of_the_run_in_one_call(tmp_path):
    write_work(tmp_path, {"pa": small_part("a", 1), "pb": small_part("b", 2)})
    cut.run_cut(tmp_path, ["pa", "pb"], CutConfig())
    out = tmp_path / "cut"
    before = {p.name: p.read_bytes() for p in out.glob("*.json*")}
    with pytest.raises(CutError, match="pb"):
        cut.run_cut(tmp_path, ["pa"], CutConfig())
    assert {p.name: p.read_bytes() for p in out.glob("*.json*")} == before
    cut.run_cut(tmp_path, ["pb", "pa"], CutConfig())
    assert sorted(json.loads((out / "report.json").read_text())["parts"]) == ["pa", "pb"]


def test_run_cut_rejects_identical_audio_in_two_parts(tmp_path):
    pcm, units_a, spans_a = small_part("a", 1)
    _, units_b, spans_b = small_part("a", 1)
    units_b = [{**u, "id": "b" + u["id"][1:]} for u in units_b]
    spans_b = [{**s, "id": "b" + s["id"][1:]} for s in spans_b]
    write_work(tmp_path, {"pa": (pcm, units_a, spans_a), "pb": (pcm, units_b, spans_b)})
    with pytest.raises(CutError):
        cut.run_cut(tmp_path, ["pa", "pb"], CutConfig())


@pytest.mark.parametrize("kind", ["stereo", "16k"])
def test_run_cut_rejects_wrong_wav_format(tmp_path, kind):
    pcm, units, spans = small_part("a", 1)
    write_work(tmp_path, {"pa": (pcm, units, spans)})
    wav = tmp_path / "audio" / "pa.24k.wav"
    if kind == "stereo":
        sf.write(str(wav), np.stack([pcm, pcm], axis=1), SR, subtype="PCM_16")
    else:
        sf.write(str(wav), pcm, 16000, subtype="PCM_16")
    with pytest.raises(CutError):
        cut.run_cut(tmp_path, ["pa"], CutConfig())


# ---------------------------------------------------------------- 11 property

@pytest.mark.parametrize("seed", range(25), ids=lambda s: f"seed{s:02d}")
def test_random_layouts_hold_the_invariants(seed):
    rng = np.random.default_rng(1000 + seed)
    n = int(rng.integers(5, 11))
    items = [pz(0.5)]
    for i in range(n):
        b = "mid" if rng.random() < 0.7 else ("para_end" if rng.random() < 0.65 else "chapter_end")
        items.append(sp(int(rng.integers(100, 300)) / 100, b, est_s=float(rng.uniform(1, 9)), heading=bool(rng.random() < 0.05)))
        items.append(pz(0.5) if i == n - 1 else pz(int(rng.integers(3, 201)) / 100))
    pcm, truth = make_part(items, 2000 + seed)
    units = make_units(items)
    spans = stub_spans(units, truth, float(rng.uniform(0, 0.06)), seed)
    plan = cut.plan_part("p", pcm, SR, units, spans, CutConfig())
    cl = plan.clips
    assert cl == sorted(cl, key=lambda c: c.start_sample)
    assert all(a.end_sample <= b.start_sample for a, b in zip(cl, cl[1:]))
    assert len({c.sha256 for c in cl}) == len(cl)
    assert all(c.sha256 == hashlib.sha256(pcm[c.start_sample:c.end_sample].tobytes()).hexdigest() for c in cl)
    assert_holds_true_speech(plan, truth, units)
    for c1, c2, _ in clip_pairs(plan, units):
        if c2.speech_start - c1.speech_end < 2 * cut.to_sample(PAD, SR):
            assert "short_pause" in c1.flags and "short_pause" in c2.flags, (c1.id, c2.id)


# ---------------------------------------------------------------- 12 pauses

def test_pause_stats_medians_and_counts_from_a_layout():
    gaps_after = [(0.5, "mid"), (0.7, "mid"), (0.9, "mid"), (1.5, "para_end"), (0.6, "mid"), (2.0, "para_end"),
                  (0.8, "mid"), (3.0, "chapter_end"), (1.0, "mid")]
    items = [pz(0.5)]
    for i, (g, b) in enumerate(gaps_after):
        items += [sp(1.5, b, est_s=3.0), pz(g)]
    items += [sp(1.5, "chapter_end", est_s=3.0), pz(0.5)]
    pcm, truth = make_part(items, 51)
    units = make_units(items)
    spans = stub_spans(units, truth, 0.02, 4)
    plan = cut.plan_part("p", pcm, SR, units, spans, CutConfig())
    stats = cut.pause_stats(plan.gaps, plan.commas)
    expect = {"sentence": [g for g, b in gaps_after if b == "mid"], "paragraph": [g for g, b in gaps_after if b == "para_end"],
              "chapter": [g for g, b in gaps_after if b == "chapter_end"]}
    for k, v in expect.items():
        assert stats[k]["n"] == len(v), k
        assert abs(stats[k]["median_s"] - float(np.median(v))) <= 0.02, k
        assert stats[k]["p10_s"] <= stats[k]["median_s"] <= stats[k]["p90_s"]
    exp_commas = []
    for u, s in zip(units, spans):
        exp_commas += _expected_commas(u, s)
    assert exp_commas and stats["comma"]["n"] == len(exp_commas)
    assert plan.commas == pytest.approx(exp_commas, abs=1e-3)


def _expected_commas(unit, span):
    words = span["words"]
    out = []
    for k in range(len(words) - 1):
        pos = [m.span() for m in WORD_RE.finditer(unit["text"])]
        if "," in unit["text"][pos[k][1]:pos[k + 1][0]]:
            out.append(words[k + 1]["start"] - words[k]["end"])
    return out


def _timed(texts, times):
    return [{"text": t, "start": a, "end": b, "score": 0.9} for t, (a, b) in zip(texts, times)]


def test_comma_gaps():
    unit = {"id": "u", "text": "Ráno, keď prišiel domov, otvoril okno."}
    times = [(0.0, 0.3), (0.45, 0.7), (0.8, 1.1), (1.2, 1.5), (1.9, 2.3), (2.4, 2.7)]
    span = {"id": "u", "start_s": 0.0, "end_s": 2.7, "words": _timed(WORD_RE.findall(unit["text"]), times)}
    assert cut.comma_gaps(unit, span) == pytest.approx([0.15, 0.4], abs=1e-9)
    assert cut.comma_gaps(unit, {**span, "words": span["words"][:-1]}) == []
    assert cut.comma_gaps(unit, {k: v for k, v in span.items() if k != "words"}) == []
    assert cut.comma_gaps(unit, {**span, "words": []}) == []
    plain = {"id": "u", "text": "Ráno keď prišiel domov otvoril okno."}
    assert cut.comma_gaps(plain, span) == []
    tail = {"id": "t", "text": "Hello, world,"}
    tspan = {"id": "t", "start_s": 0.0, "end_s": 1.0, "words": _timed(["Hello", "world"], [(0.0, 0.3), (0.5, 0.9)])}
    assert cut.comma_gaps(tail, tspan) == pytest.approx([0.2], abs=1e-9)


def test_pause_stats_values_and_empty():
    st = cut.pause_stats([("sentence", 0.5), ("sentence", 0.7), ("sentence", 0.9), ("paragraph", 1.5)], [0.1, 0.3])
    assert st["sentence"]["n"] == 3 and st["sentence"]["median_s"] == pytest.approx(0.7)
    assert st["paragraph"]["n"] == 1 and st["paragraph"]["median_s"] == st["paragraph"]["p10_s"] == st["paragraph"]["p90_s"] == 1.5
    assert st["comma"]["n"] == 2 and st["comma"]["median_s"] == pytest.approx(0.2)
    assert st["sentence"]["p10_s"] <= 0.7 <= st["sentence"]["p90_s"]
    assert st["chapter"] == {"n": 0, "median_s": None, "p10_s": None, "p90_s": None}
    empty = cut.pause_stats([], [])
    assert set(empty) == {"comma", "sentence", "paragraph", "chapter"}
    assert all(v == {"n": 0, "median_s": None, "p10_s": None, "p90_s": None} for v in empty.values())


# ---------------------------------------------------------------- 14 over_cap flag only

def test_over_cap_is_a_flag_only():
    items = [pz(0.5), sp(1.5), pz(0.8), sp(21.0, est_s=21.0), pz(0.8), sp(1.5), pz(0.5)]
    pcm, truth = make_part(items, 61)
    units = make_units(items)
    spans = stub_spans(units, truth, 0.02, 1, words=False)
    plan = cut.plan_part("p", pcm, SR, units, spans, CutConfig())
    assert len(plan.clips) == 3
    big = clip_of(plan, units, 1)
    assert (big.end_sample - big.start_sample) / SR > 20.0 and "over_cap" in big.flags
    assert [c.id for c in plan.clips if "over_cap" in c.flags] == [big.id]
    assert_holds_true_speech(plan, truth, units)


# ---------------------------------------------------------------- 15 over-cap re-split

def _timed_words(words, s, e):
    step = (e - s) / len(words)
    out = [{"text": w, "start": s + k * step, "end": s + (k + 0.95) * step, "score": 0.9} for k, w in enumerate(words)]
    out[-1]["end"] = e
    return out


def split_case(items, a, b, seed, status="ok"):
    """Speech units a..b of the part become ONE unit with one span and word timings that follow the audio."""
    pcm, truth = make_part(items, seed)
    units = make_units(items)
    spans = stub_spans(units, truth, 0.02, 1, words=False)
    group = units[a:b + 1]
    words = []
    for i in range(a, b + 1):
        words += _timed_words(units[i]["words"], truth["speech"][i][0] / SR, truth["speech"][i][1] / SR)
    merged = {**group[0], "text": " ".join(u["text"] for u in group), "words": [w for u in group for w in u["words"]],
              "boundary": group[-1]["boundary"], "est_s": sum(u["est_s"] for u in group)}
    span = {"id": merged["id"], "start_s": words[0]["start"], "end_s": words[-1]["end"], "status": status, "words": words}
    return pcm, truth, units[:a] + [merged] + units[b + 1:], spans[:a] + [span] + spans[b + 1:], group


def two_halves(pause, seed):
    items = [pz(0.5), sp(1.5), pz(0.8), sp(11.0), pz(pause), sp(11.0, "para_end"), pz(0.8), sp(1.5), pz(0.5)]
    return items, split_case(items, 1, 2, seed)


def test_over_cap_unit_splits_at_the_longest_pause():
    items, (pcm, truth, units, spans, group) = two_halves(0.6, 71)
    plan = cut.plan_part("p", pcm, SR, units, spans, CutConfig())
    parent = units[1]["id"]
    ids = [c.id for c in sorted(plan.clips, key=lambda c: c.start_sample)]
    assert ids == [units[0]["id"], f"{parent}.1", f"{parent}.2", units[2]["id"]] and not plan.rejects
    c1, c2 = (next(c for c in plan.clips if c.id == f"{parent}.{j}") for j in (1, 2))
    for c in (c1, c2):
        assert (c.end_sample - c.start_sample) / SR <= 20.0 and "over_cap" not in c.flags
    assert " ".join([c1.text, c2.text]) == units[1]["text"]
    assert [c1.boundary, c2.boundary] == ["mid", units[1]["boundary"]]
    for c, i in ((c1, 1), (c2, 2)):
        s, e = truth["speech"][i]
        assert c.start_sample <= s + F and c.end_sample >= e - F
    pause = truth["pause_after"][1]
    assert pause["start"] <= c1.end_sample <= pause["end"]
    assert c1.text == group[0]["text"] and c2.text == group[1]["text"]


def test_over_cap_pieces_are_not_merged_back_over_a_short_pause():
    items, (pcm, truth, units, spans, group) = two_halves(0.2, 72)
    plan = cut.plan_part("p", pcm, SR, units, spans, CutConfig())
    parent = units[1]["id"]
    pieces = [c for c in plan.clips if c.id.startswith(parent + ".")]
    assert [c.id for c in pieces] == [f"{parent}.1", f"{parent}.2"] and all(len(c.units) == 1 for c in pieces)
    assert all("short_pause" in c.flags and "over_cap" not in c.flags for c in pieces)


def test_over_cap_unit_splits_into_three_at_both_pauses():
    items = [pz(0.5), sp(15.0), pz(0.6), sp(15.0), pz(0.5), sp(15.0), pz(0.5)]
    pcm, truth, units, spans, group = split_case(items, 0, 2, 73)
    plan = cut.plan_part("p", pcm, SR, units, spans, CutConfig())
    parent = units[0]["id"]
    cl = sorted(plan.clips, key=lambda c: c.start_sample)
    assert [c.id for c in cl] == [f"{parent}.{j}" for j in (1, 2, 3)]
    assert [c.text for c in cl] == [u["text"] for u in group]
    for i, c in enumerate(cl):
        assert (c.end_sample - c.start_sample) / SR <= 20.0 and "over_cap" not in c.flags
        assert c.start_sample <= truth["speech"][i][0] + F and c.end_sample >= truth["speech"][i][1] - F
    for i in (0, 1):
        assert truth["pause_after"][i]["start"] <= cl[i].end_sample <= truth["pause_after"][i]["end"]


def _five_words(u, truth):
    s, e = truth["speech"][1]
    text = "Rain tapped on the roof"
    return {**u, "text": text, "words": WORD_RE.findall(text)}, _timed_words(WORD_RE.findall(text), s / SR, e / SR)


def _en_only(u, truth):
    s, e = truth["speech"][1]
    text = "<en>one two three four five six seven</en>"
    words = WORD_RE.findall("one two three four five six seven")
    return {**u, "text": text, "words": words}, _timed_words(words, s / SR, e / SR)


def _count_differs(u, truth):
    s, e = truth["speech"][1]
    return u, _timed_words(u["words"][:-1], s / SR, e / SR)


@pytest.mark.parametrize("variant", [_five_words, _en_only, _count_differs])
def test_over_cap_without_a_valid_split_stays_one_flagged_clip(variant):
    items = [pz(0.5), sp(1.5), pz(0.8), sp(21.0, est_s=21.0), pz(0.8), sp(1.5), pz(0.5)]
    pcm, truth = make_part(items, 74)
    units = make_units(items)
    spans = stub_spans(units, truth, 0.02, 1, words=False)
    units[1], spans[1]["words"] = variant(units[1], truth)
    out_units, out_spans = cut.split_over_cap(units, spans, CutConfig(), SR)
    assert out_units == units and out_spans == spans
    plan = cut.plan_part("p", pcm, SR, units, spans, CutConfig())
    big = clip_of(plan, units, 1)
    assert big.id == units[1]["id"] and "over_cap" in big.flags and len(plan.clips) == 3


def test_over_cap_low_conf_unit_is_rejected_not_split():
    items, _ = two_halves(0.6, 75)
    pcm, truth, units, spans, _ = split_case(items, 1, 2, 75, status="low_conf")
    plan = cut.plan_part("p", pcm, SR, units, spans, CutConfig())
    assert [(r["id"], r["reason"]) for r in plan.rejects] == [(units[1]["id"], "low_conf")]
    assert not any("." in c.id for c in plan.clips)


def test_unit_under_the_cap_is_untouched_by_the_split():
    d = std()
    spans = stub_spans(d["units"], d["truth"], 0.02, 5)
    before = copy.deepcopy((d["units"], spans))
    out_units, out_spans = cut.split_over_cap(d["units"], spans, CutConfig(), SR)
    assert (out_units, out_spans) == before
    assert all(a is b for a, b in zip(out_units, d["units"])) and all(a is b for a, b in zip(out_spans, spans))


def test_split_leaves_its_inputs_untouched():
    items, (pcm, truth, units, spans, group) = two_halves(0.6, 76)
    before = copy.deepcopy((units, spans))
    out_units, out_spans = cut.split_over_cap(units, spans, CutConfig(), SR)
    assert (units, spans) == before
    assert [u["id"] for u in out_units] == [units[0]["id"], units[1]["id"] + ".1", units[1]["id"] + ".2", units[2]["id"]]
    assert [s["id"] for s in out_spans] == [u["id"] for u in out_units]
    assert all(u.get("split_of") == units[1]["id"] for u in out_units[1:3])


def test_run_cut_cuts_the_pieces_of_a_split_unit(tmp_path):
    items, (pcm, truth, units, spans, group) = two_halves(0.6, 77)
    write_work(tmp_path, {"pa": (pcm, units, spans)})
    cut.run_cut(tmp_path, ["pa"], CutConfig())
    out, parent = tmp_path / "cut", units[1]["id"]
    assert (out / "clips" / f"{parent}.1.wav").is_file() and (out / "clips" / f"{parent}.2.wav").is_file()
    recs = [json.loads(x) for x in (out / "pa.cuts.jsonl").read_text().splitlines()]
    assert [r["units"] for r in recs] == [[units[0]["id"]], [f"{parent}.1"], [f"{parent}.2"], [units[2]["id"]]]
    assert len({r["sha256"] for r in recs}) == len(recs)


def test_words_of_an_en_span_count_without_the_tag_names():
    unit = {"id": "u", "text": "Ráno <en>hello world</en>, otvoril okno."}
    words = ["Ráno", "hello", "world", "otvoril", "okno"]
    times = [(0.0, 0.3), (0.4, 0.7), (0.8, 1.1), (1.6, 2.0), (2.1, 2.4)]
    span = {"id": "u", "start_s": 0.0, "end_s": 2.4, "words": _timed(words, times)}
    assert cut.comma_gaps(unit, span) == pytest.approx([0.5], abs=1e-9)


def test_over_cap_split_keeps_an_en_span_whole():
    text = "alpha beta gamma <en>one two three</en> delta epsilon zeta eta theta iota"
    words = ["alpha", "beta", "gamma", "one", "two", "three", "delta", "epsilon", "zeta", "eta", "theta", "iota"]
    times = [(0.0, 1.0), (1.2, 2.2), (2.4, 3.4), (3.6, 4.6), (7.0, 8.0), (8.2, 9.2), (9.5, 10.5), (10.7, 11.7), (11.9, 12.9),
             (13.1, 14.1), (14.3, 15.3), (23.0, 24.0)]
    unit = {"id": "u", "text": text, "boundary": "para_end", "est_s": 20.0, "paragraph": 0, "chapter": 0, "heading": False, "words": words}
    span = {"id": "u", "start_s": 0.0, "end_s": 24.0, "status": "ok", "words": _timed(words, times)}
    out_units, out_spans = cut.split_over_cap([unit], [span], CutConfig(), SR)
    assert [u["id"] for u in out_units] == ["u.1", "u.2"] and [s["id"] for s in out_spans] == ["u.1", "u.2"]
    assert [u["text"] for u in out_units] == ["alpha beta gamma <en>one two three</en>", "delta epsilon zeta eta theta iota"]
    assert [u["words"] for u in out_units] == [words[:6], words[6:]]
    assert [u["boundary"] for u in out_units] == ["mid", "para_end"]
    assert [(s["start_s"], s["end_s"]) for s in out_spans] == [(0.0, 9.2), (9.5, 24.0)]
