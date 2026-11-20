"""Coverage and split acceptance tests (spec: T4 spec.md "Coverage and split"). Seeded synthetic items only, no book text."""
import hashlib
import math
import os
import random
import subprocess
import sys
from collections import Counter

import pytest

from src.data.prepare.coverage import (SplitConfig, SplitError, check_split, clip_tags, coverage_markdown,
                                       coverage_table, group_key, make_split)

WORDS = ("amber basket candle dune ember field garden harbor island jungle kettle lantern meadow needle orchard pebble "
         "quiver ribbon saddle timber umbrella valley window yonder zephyr anchor bridge cobble dagger engine forest "
         "glacier hammer ivory jacket lagoon marble nectar oyster parcel quartz rocket").split()
COMMON = (("statement", 0.8), ("mid", 0.5), ("para_end", 0.3), ("quote_only", 0.2))
RARE = (("rare3", 3), ("rare5", 5), ("rare8", 8), ("rare14", 14))
NO_BLOCK = {"heldout_share": 2.0}


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def hsh(seed, first_id):
    return hashlib.sha1(f"{seed}:{first_id}".encode()).hexdigest()


def it(id, para=0, tags=(), dur=5.0, ok=True, text=None, part="p1", chapter=0):
    return {"id": id, "part": part, "chapter": chapter, "paragraph": para, "tags": sorted(tags), "duration_s": dur,
            "eligible": ok, "sha256": sha(id), "text": text if text is not None else f"plain line {id} of the story book"}


def make_corpus(seed=1):
    rng = random.Random(seed)
    items, texts, idx = [], set(), 0
    for part in range(6):
        for chapter in range(3):
            for para in range(rng.randint(8, 14)):
                for _ in range(rng.randint(1, 3)):
                    while True:
                        text = " ".join(rng.choice(WORDS) for _ in range(rng.randint(5, 12)))
                        if text not in texts:
                            texts.add(text)
                            break
                    tags = {t for t, p in COMMON if rng.random() < p}
                    items.append(it(f"p{part}-{idx:05d}", para, tags, rng.uniform(2, 10), rng.random() >= 0.08, text,
                                    f"part{part}", chapter))
                    idx += 1
    for name, n in RARE:
        for i in rng.sample(range(len(items)), n):
            items[i]["tags"] = sorted(set(items[i]["tags"]) | {name})
    return items


def groups_of(items):
    out = {}
    for x in items:
        out.setdefault(group_key(x), []).append(x)
    return out


def val_test_has(res, items, split, tag):
    return sum(1 for x in items if res.assignment.get(x["id"]) == split and tag in x["tags"])


class TestClipTags:
    UNITS = {"a1-00007": {"tags": ["mid", "quote_only"]}, "a1-00008": {"tags": ["mid", "joined"]}, "a1-00009": {"tags": ["number"]}}

    def test_single_unit(self):
        assert clip_tags({"units": ["a1-00009"]}, self.UNITS) == ["number"]

    def test_piece_takes_parent_tags(self):
        assert clip_tags({"id": "a1-00007.2", "units": ["a1-00007.2"]}, self.UNITS) == ["mid", "quote_only"]

    def test_merged_union_sorted(self):
        assert clip_tags({"units": ["a1-00007", "a1-00008"]}, self.UNITS) == ["joined", "mid", "quote_only"]

    def test_merged_piece_and_unit(self):
        assert clip_tags({"units": ["a1-00009", "a1-00007.1"]}, self.UNITS) == ["mid", "number", "quote_only"]


@pytest.mark.parametrize("n,quota", [(3, 1), (9, 1), (10, 1), (11, 2), (50, 5), (500, 5)])
def test_quota(n, quota):
    items = [it(f"x{i:04d}", i, ["t"], 2.0) for i in range(n)]
    res = make_split(items, SplitConfig(**NO_BLOCK))
    assert res.quotas == {"t": quota}
    assert res.skipped_tags == {}


def test_quota_skips_tags_under_three_clips():
    items = [it(f"x{i}", i, ["t"]) for i in range(2)] + [it("y", 9, ["u"])]
    res = make_split(items, SplitConfig(**NO_BLOCK))
    assert res.skipped_tags == {"t": 2, "u": 1}
    assert res.quotas == {}


@pytest.fixture(scope="module", params=[0, 7])
def run(request):
    items = make_corpus()
    cfg = SplitConfig(seed=request.param)
    return items, cfg, make_split(items, cfg)


def test_corpus_size():
    assert 350 <= len(make_corpus()) <= 450


def test_every_item_one_split_or_excluded(run):
    items, _, res = run
    excluded = {e["id"] for e in res.excluded}
    for x in items:
        assert (x["id"] in res.assignment) != (x["id"] in excluded)
    assert set(res.assignment.values()) <= {"train", "val", "test", "heldout"}
    assert {e["reason"] for e in res.excluded} <= {"text_overlap"}


def test_no_group_in_two_splits(run):
    items, _, res = run
    for g, xs in groups_of(items).items():
        assert len({res.assignment[x["id"]] for x in xs if x["id"] in res.assignment}) <= 1, g


def test_no_ineligible_in_val_test(run):
    items, _, res = run
    assert not [x["id"] for x in items if not x["eligible"] and res.assignment.get(x["id"]) in ("val", "test")]


def test_val_and_test_share_no_group(run):
    items, _, res = run
    seen = {s: {group_key(x) for x in items if res.assignment.get(x["id"]) == s} for s in ("val", "test", "heldout")}
    assert seen["val"] and seen["test"]
    assert not seen["val"] & seen["test"]
    assert not seen["heldout"] & (seen["val"] | seen["test"])


def test_quota_met_or_shortfall_recorded(run):
    items, _, res = run
    assert set(res.quotas) >= {"rare3", "rare5", "rare8", "rare14", "statement"}
    assert res.quotas["rare14"] == 2 and res.quotas["rare3"] == 1 and res.quotas["statement"] == 5
    groups = groups_of(items)
    placed = {s: {g for g, xs in groups.items() if any(res.assignment.get(x["id"]) == s for x in xs)} for s in ("val", "test", "heldout")}
    for split in ("val", "test"):
        for tag, q in res.quotas.items():
            have = val_test_has(res, items, split, tag)
            if have >= q:
                assert tag not in res.unmet[split]
                continue
            assert res.unmet[split][tag] == q - have
            blocked = placed["heldout"] | placed["val"] | (placed["test"] if split == "test" else set())
            left = [g for g, xs in groups.items() if g not in blocked and all(x["eligible"] for x in xs)
                    and any(tag in x["tags"] for x in xs)]
            assert not left, (split, tag)


def test_heldout_block(run):
    items, cfg, res = run
    ids = [x["id"] for x in items if res.assignment.get(x["id"]) == "heldout"]
    pos = [i for i, x in enumerate(items) if x["id"] in set(ids)]
    assert pos == list(range(pos[0], pos[-1] + 1))
    block = [items[i] for i in pos]
    assert len({(x["part"], x["chapter"]) for x in block}) == 1
    target = cfg.heldout_share * sum(x["duration_s"] for x in items)
    hours_s = sum(x["duration_s"] for x in block)
    assert 0.5 * target <= hours_s <= 2 * target
    assert res.heldout["clips"] == len(block)
    assert res.heldout["hours"] == pytest.approx(hours_s / 3600)
    assert (res.heldout["part"], res.heldout["chapter"]) == (block[0]["part"], block[0]["chapter"])
    assert res.heldout["paragraphs"] == len({group_key(x) for x in block})
    other = {group_key(x) for x in items if res.assignment.get(x["id"]) in ("val", "test", "train")}
    assert not {group_key(x) for x in block} & other


def test_check_split_passes_on_result(run):
    items, _, res = run
    check_split(items, res.assignment, {x["id"]: x["sha256"] for x in items}, excluded=res.excluded)


def test_deterministic():
    items = make_corpus()
    a, b = make_split(items, SplitConfig()), make_split(items, SplitConfig())
    assert a == b


def test_other_seed_changes_assignment():
    items = make_corpus()
    assert make_split(items, SplitConfig(seed=0)).assignment != make_split(items, SplitConfig(seed=7)).assignment


class TestHandBuilt:
    def test_ratio_rule_prefers_small_group(self):
        for seed in range(8):
            items = [it("g1-0", 0, ["r"]), *[it(f"g4-{i}", 1, ["r"] if i == 0 else []) for i in range(4)],
                     it("fill", 2, ["r"], ok=False)]
            res = make_split(items, SplitConfig(seed=seed, **NO_BLOCK))
            assert res.assignment["g1-0"] == "val", seed
            assert res.assignment["g4-0"] == "test", seed

    def test_tie_goes_to_lower_sha1(self):
        for seed in range(8):
            items = [it("ga", 0, ["r"]), it("gb", 1, ["r"]), it("fill", 2, ["r"], ok=False)]
            res = make_split(items, SplitConfig(seed=seed, **NO_BLOCK))
            low = min(("ga", "gb"), key=lambda i: hsh(seed, i))
            assert res.assignment[low] == "val", seed
        winners = {min(("ga", "gb"), key=lambda i: hsh(s, i)) for s in range(8)}
        assert winners == {"ga", "gb"}

    def test_rarest_tag_goes_first(self):
        for seed in range(8):
            items = [it("W", 0, ["mm", "zz"]), it("Z", 1, ["zz"]), it("V", 2, ["aa", "mm"]),
                     it("U1", 3, ["aa"]), it("U2", 3, []),
                     it("f1", 4, ["zz"], ok=False), it("f2", 5, ["mm"], ok=False), it("f3", 6, ["mm"], ok=False),
                     it("f4", 7, ["aa"], ok=False), it("f5", 8, ["aa"], ok=False), it("f6", 9, ["aa"], ok=False)]
            res = make_split(items, SplitConfig(seed=seed, **NO_BLOCK))
            assert res.quotas == {"zz": 1, "mm": 1, "aa": 1}
            val = {i for i, s in res.assignment.items() if s == "val"}
            assert val == {"W", "V"}, seed

    def test_group_with_ineligible_clip_never_val_test(self):
        items = [it("p0", 0, []), it("p1", 0, ["r"], ok=False), it("f1", 1, ["r"], ok=False), it("f2", 2, ["r"], ok=False)]
        res = make_split(items, SplitConfig(**NO_BLOCK))
        assert res.unmet == {"val": {"r": 1}, "test": {"r": 1}}
        assert {res.assignment[i] for i in ("p0", "p1")} == {"train"}

    def test_whole_chapter_run_beats_windows(self):
        items = ([it(f"a{i}", i, ["a"], 10, chapter=0) for i in range(3)]
                 + [it(f"b{i}", i, ["b"], 10, chapter=1) for i in range(10)]
                 + [it(f"c{i}", i, ["b"], 10, chapter=2) for i in range(17)])
        res = make_split(items, SplitConfig(heldout_share=0.1))
        assert {i for i, s in res.assignment.items() if s == "heldout"} == {"a0", "a1", "a2"}
        assert res.heldout == {"part": "p1", "chapter": 0, "clips": 3, "hours": pytest.approx(30 / 3600), "paragraphs": 3}

    def test_window_of_paragraphs_when_no_run_fits(self):
        tags = ["p"] * 10 + ["q"] * 10
        items = ([it(f"a{i}", i, ["p"], 10, chapter=0) for i in range(10)]
                 + [it(f"b{i}", i, [tags[i]], 10, chapter=1) for i in range(20)])
        res = make_split(items, SplitConfig(heldout_share=0.1))
        assert {i for i, s in res.assignment.items() if s == "heldout"} == {"b8", "b9", "b10"}
        assert res.heldout["chapter"] == 1 and res.heldout["paragraphs"] == 3

    def test_window_never_crosses_chapters(self):
        items = [it(f"c{c}", 0, [], 10, chapter=c) for c in range(4)]
        res = make_split(items, SplitConfig(heldout_share=0.75))
        assert res.heldout is None
        assert "heldout" not in res.assignment.values()

    def test_no_heldout_when_total_below_target(self):
        items = [it(f"x{i}", i, [], 10) for i in range(20)]
        res = make_split(items, SplitConfig(heldout_share=1.5))
        assert res.heldout is None
        assert set(res.assignment.values()) == {"train"}

    def test_text_overlap_excluded(self):
        v5, v3 = "one two three four five", "alpha beta gamma"
        items = [it("v5", 0, ["r"], text=v5), it("v3", 1, ["s"], text=v3),
                 it("t5", 2, text="One, two; THREE four... five!"), it("t3", 3, text="alpha beta gamma"),
                 it("fr1", 4, ["r"], ok=False), it("fr2", 5, ["r"], ok=False), it("fs1", 6, ["s"], ok=False), it("fs2", 7, ["s"], ok=False)]
        res = make_split(items, SplitConfig(**NO_BLOCK))
        assert res.assignment["v5"] == "val" and res.assignment["v3"] == "val"
        assert res.excluded == [{"id": "t5", "reason": "text_overlap"}]
        assert "t5" not in res.assignment
        assert res.assignment["t3"] == "train"

    def test_text_overlap_against_heldout(self):
        items = [it("h0", 0, text="one two three four five", chapter=0), it("t0", 0, text="one two three four five", chapter=1),
                 it("t1", 1, chapter=1)]
        res = make_split(items, SplitConfig(heldout_share=0.3))
        assert res.assignment["h0"] == "heldout"
        assert res.excluded == [{"id": "t0", "reason": "text_overlap"}]


class TestCheckSplit:
    ITEMS = [it("a", 0), it("b", 0), it("c", 1), it("d", 2)]
    GOOD = {"a": "train", "b": "train", "c": "val", "d": "heldout"}
    HASH = {x["id"]: x["sha256"] for x in ITEMS}

    def test_valid(self):
        check_split(self.ITEMS, self.GOOD, self.HASH)

    def test_shared_sha(self):
        with pytest.raises(SplitError):
            check_split(self.ITEMS, self.GOOD, {**self.HASH, "c": self.HASH["a"]})

    def test_same_split_same_sha_ok(self):
        check_split(self.ITEMS, self.GOOD, {**self.HASH, "b": self.HASH["a"]})

    def test_group_in_two_splits(self):
        with pytest.raises(SplitError):
            check_split(self.ITEMS, {**self.GOOD, "b": "val"}, self.HASH)

    def test_heldout_not_consecutive(self):
        with pytest.raises(SplitError):
            check_split(self.ITEMS, {"a": "heldout", "b": "heldout", "c": "train", "d": "heldout"}, self.HASH)

    def test_heldout_across_chapters(self):
        items = [*self.ITEMS[:3], it("d", 2, chapter=1)]
        with pytest.raises(SplitError):
            check_split(items, {"a": "train", "b": "train", "c": "heldout", "d": "heldout"}, self.HASH)

    def test_unassigned(self):
        with pytest.raises(SplitError):
            check_split(self.ITEMS, {k: v for k, v in self.GOOD.items() if k != "d"}, self.HASH)

    def test_excluded_is_not_unassigned(self):
        check_split(self.ITEMS, {k: v for k, v in self.GOOD.items() if k != "d"}, self.HASH, excluded=[{"id": "d", "reason": "text_overlap"}])


class TestCoverageTable:
    ITEMS = [it("a", 0, ["number", "statement"], 3.6), it("b", 0, ["statement"], 7.2), it("c", 1, ["date", "statement"], 3.6),
             it("d", 2, ["number"], 3.6), it("e", 0, ["roman", "statement"], 7.2, part="p2"), it("f", 0, ["statement"], 3.6, part="p2"),
             it("g", 3, ["statement"], 3.6)]
    ASSIGN = {"a": "train", "b": "train", "c": "val", "d": "test", "e": "heldout", "f": "heldout"}

    def test_tags(self):
        t = coverage_table(self.ITEMS, self.ASSIGN)["tags"]
        z = {"clips": 0, "hours": 0}
        assert t["statement"] == {"clips": 5, "hours": pytest.approx(0.007), "by_split": {
            "train": {"clips": 2, "hours": pytest.approx(0.003)}, "val": {"clips": 1, "hours": pytest.approx(0.001)},
            "test": z, "heldout": {"clips": 2, "hours": pytest.approx(0.003)}}}
        assert t["number"]["clips"] == 2 and t["number"]["hours"] == pytest.approx(0.002)
        assert t["number"]["by_split"]["train"]["clips"] == 1 and t["number"]["by_split"]["test"]["clips"] == 1
        assert t["date"]["clips"] == 1 and t["date"]["by_split"]["val"]["hours"] == pytest.approx(0.001)
        assert t["roman"]["by_split"]["heldout"] == {"clips": 1, "hours": pytest.approx(0.002)}
        assert set(t) == {"statement", "number", "date", "roman"}

    def test_splits(self):
        s = coverage_table(self.ITEMS, self.ASSIGN)["splits"]
        assert s["train"] == {"clips": 2, "hours": pytest.approx(0.003), "paragraphs": 1, "parts": {"p1": 2}}
        assert s["val"] == {"clips": 1, "hours": pytest.approx(0.001), "paragraphs": 1, "parts": {"p1": 1}}
        assert s["test"] == {"clips": 1, "hours": pytest.approx(0.001), "paragraphs": 1, "parts": {"p1": 1}}
        assert s["heldout"] == {"clips": 2, "hours": pytest.approx(0.003), "paragraphs": 1, "parts": {"p2": 2}}

    def test_structural_present_with_zero(self):
        st = coverage_table(self.ITEMS, self.ASSIGN)["structural"]
        assert list(st) == ["number", "date", "time", "roman", "abbreviation"]
        assert st["number"] == {"clips": 2, "hours": pytest.approx(0.002)}
        assert st["time"] == {"clips": 0, "hours": 0} and st["abbreviation"] == {"clips": 0, "hours": 0}

    def test_thin(self):
        assert coverage_table(self.ITEMS, self.ASSIGN)["thin"] == ["date", "number", "roman", "statement"]
        cfg = SplitConfig(thin_clips=3, thin_hours=0.0015)
        assert coverage_table(self.ITEMS, self.ASSIGN, cfg)["thin"] == ["date", "number", "roman"]

    def test_markdown_has_no_text(self):
        items = [{**x, "text": f"secret sentence {x['id']} zzq"} for x in self.ITEMS]
        md = coverage_markdown(coverage_table(items, self.ASSIGN))
        assert "statement" in md and "heldout" in md and "p1" in md
        for x in items:
            assert x["text"] not in md and "secret" not in md


def test_import_is_light():
    code = ("import sys, src.data.prepare.coverage; "
            "bad = [m for m in ('torch', 'transformers') if m in sys.modules]; "
            "sys.exit(1 if bad else 0)")
    env = {**os.environ, "PYTHONPATH": os.getcwd()}
    assert subprocess.run([sys.executable, "-c", code], env=env).returncode == 0
