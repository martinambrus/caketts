"""Manifest stage acceptance tests (spec: T4 spec.md "Manifest"). Synthetic tones and own sentences only, no book text."""
import hashlib
import itertools
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.data.prepare.manifest import ManifestConfig, clip_word_times, run_manifest, tempo_from_words
from src.data.tempo import tempo_from_alignment
from src.text.phonemizer import CzechSlovakPhonemizer

SR = 24000
PARTS = ("pa", "pb")
SUBJ = ["Starý mlýn", "Malý chlapec", "Naše babička", "Černý pes", "Velká loď", "Mladá učitelka", "Starý rybář", "Tichá řeka"]
VERB = ["stál", "čekal", "zpíval", "seděl", "hledal", "pracoval", "spal", "odpočíval"]
OBJ = ["u řeky", "na kopci", "před domem", "v lese", "za městem", "u cesty", "na louce", "pod mostem"]
SPLITS = ("train", "val", "test", "heldout")
G2P = CzechSlovakPhonemizer("cs")


def sentence(i, q=False):
    return f"{SUBJ[i % 8]} {VERB[i // 8 % 8]} {OBJ[i // 64 % 8]}{'?' if q else '.'}"


def jsonl(rows):
    return "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows)


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


class PartBuilder:
    def __init__(self, name, work, seeds):
        self.name, self.work, self.seeds = name, work, seeds
        self.cur, self.index = 0, 0
        self.units, self.cuts, self.aligns = [], [], {}

    def _unit_id(self):
        uid = f"{self.name}-{self.index:05d}"
        self.index += 1
        return uid

    def _write(self, cid, units, text, boundary, start, end, pad, flags):
        pcm = _tone(next(self.seeds), end - start)
        sf.write(str(self.work / "cut" / "clips" / f"{cid}.wav"), pcm, SR, subtype="PCM_16")
        self.cuts.append({"id": cid, "units": units, "text": text, "boundary": boundary, "part": self.name,
                          "start_sample": start, "end_sample": end, "pad_start_s": pad, "pad_end_s": pad,
                          "gap_before_s": 0.1, "gap_after_s": 0.1, "flags": sorted(flags),
                          "sha256": hashlib.sha256(pcm.tobytes()).hexdigest()})

    def _unit_row(self, uid, text, boundary, chapter, paragraph, tags):
        self.units.append({"id": uid, "part": self.name, "index": int(uid.split("-")[1].split(".")[0]), "text": text,
                           "boundary": boundary, "chapter": chapter, "paragraph": paragraph, "heading": False,
                           "est_s": 1.0, "words": text.split(), "tags": sorted(tags), "source": "synthetic"})

    def _words(self, text, t, wlen, wgap):
        words = []
        for w in text.split():
            words.append({"text": w, "start": t / SR, "end": (t + round(wlen * SR)) / SR, "score": 0.9})
            t += round(wlen * SR) + round(wgap * SR)
        return words, t

    def clip(self, units, chapter, paragraph, tags, pad=0.15, wlen=0.22, wgap=0.06, ugap=0.2, flags=()):
        start = self.cur
        t = start + round(pad * SR)
        ids, texts = [], []
        for text, boundary in units:
            uid = self._unit_id()
            words, t = self._words(text, t, wlen, wgap)
            t += round(ugap * SR) - round(wgap * SR)
            self._unit_row(uid, text, boundary, chapter, paragraph, tags)
            self.aligns[uid] = {"id": uid, "start_s": words[0]["start"], "end_s": words[-1]["end"], "words": words,
                                "score_mean": 0.9, "score_min": 0.8, "status": "ok", "flags": []}
            ids.append(uid)
            texts.append(text)
        end = t - round(ugap * SR) + round(pad * SR)
        self._write(ids[0], ids, " ".join(texts), units[-1][1], start, end, pad, flags)
        self.cur = end + round(0.1 * SR)

    def pieces(self, text, chapter, paragraph, tags):
        uid = self._unit_id()
        start = self.cur
        words, t = self._words(text, start + round(0.15 * SR), 0.22, 0.06)
        n = len(words) // 2
        mid = round((words[n - 1]["end"] + words[n]["start"]) / 2 * SR)
        end = round(words[-1]["end"] * SR) + round(0.15 * SR)
        self._unit_row(uid, text, "chapter_end", chapter, paragraph, tags)
        self.aligns[uid] = {"id": uid, "start_s": words[0]["start"], "end_s": words[-1]["end"], "words": words,
                            "score_mean": 0.9, "score_min": 0.8, "status": "ok", "flags": []}
        parts = text.split()
        self._write(f"{uid}.1", [f"{uid}.1"], " ".join(parts[:n]), "mid", start, mid, 0.15, ())
        self._write(f"{uid}.2", [f"{uid}.2"], " ".join(parts[n:]), "chapter_end", mid, end, 0.15, ())
        self.cur = end + round(0.1 * SR)


def _tone(seed, n):
    rng = np.random.RandomState(seed)
    t = np.arange(n) / SR
    x = 0.3 * np.sin(2 * np.pi * (120 + (seed * 37) % 300) * t) + 0.02 * rng.randn(n)
    return (x * 32767).astype(np.int16)


def tags_for(part, g):
    tags = {"statement"}
    if g % 3 == 1:
        tags |= {"question", "has_question"}
    if g % 4 == 2:
        tags.add("exclamation")
    if g % 5 == 3:
        tags.add("number")
    if part == "pa" and g == 4:
        tags.add("part_fragment")
    return tags


class Work:
    pass


def build_work(root, drops=(3, 9, 25), reviews=(4, 11), lie=(), bad_words=(), flagged=(), fragments=()):
    root = Path(root)
    work = root / "run"
    for sub in ("text", "cut/clips", "align", "audit"):
        (work / sub).mkdir(parents=True)
    seeds, sent = itertools.count(1), itertools.count(0)
    builders = []
    for name in PARTS:
        b = PartBuilder(name, work, seeds)
        for g in range(12):
            chapter, para = g // 6, g % 6
            tags = tags_for(name, g)
            q = "question" in tags
            end_b = "chapter_end" if para == 5 else "para_end"
            u1, u2 = (sentence(next(sent), q), "mid"), (sentence(next(sent), q), end_b)
            if g % 2 == 0:
                b.clip([u1, u2], chapter, para, tags)
            else:
                b.clip([u1], chapter, para, tags)
                b.clip([u2], chapter, para, tags)
        if name == "pa":
            b.pieces(sentence(next(sent)) + " " + sentence(next(sent)), 1, 6, {"statement"})
        else:
            three = " ".join(sentence(next(sent)) for _ in range(3))
            b.clip([("Ano.", "para_end")], 1, 6, {"statement"}, pad=0.05, wlen=0.2)
            b.clip([(three, "para_end")], 1, 7, {"statement"}, wlen=1.4, wgap=0.1, flags=("over_cap",))
            b.clip([(three.replace("Starý", "Mladý"), "chapter_end")], 1, 8, {"statement"}, pad=0.05, wlen=0.03,
                   wgap=0.01)
        builders.append(b)
    cuts = [c for b in builders for c in b.cuts]
    for b in builders:
        b.cuts[0]["gap_before_s"] = None
        b.cuts[-1]["gap_after_s"] = None
    for k in bad_words:
        pid = cuts[k]["units"][-1].split(".")[0]
        next(b for b in builders if pid in b.aligns).aligns[pid]["words"].pop()
    for k in lie:
        cuts[k]["end_sample"] += SR
    unit_rows = {u["id"]: u for b in builders for u in b.units}
    for k in (*flagged, *fragments):
        if k in flagged:
            cuts[k]["flags"] = ["no_quiet_cut"]
        for uid in cuts[k]["units"]:
            row = unit_rows[uid.split(".")[0]]
            row["tags"] = sorted(set(row["tags"]) | {"flagged_rare" if k in flagged else "part_fragment"})
    for b in builders:
        (work / "text" / f"{b.name}.units.jsonl").write_text(jsonl(b.units), encoding="utf-8")
        (work / "cut" / f"{b.name}.cuts.jsonl").write_text(jsonl(b.cuts), encoding="utf-8")
        (work / "cut" / f"{b.name}.rejects.jsonl").write_text(jsonl([{"id": f"{b.name}-99999", "reason": "low_conf"}]))
        (work / "align" / f"{b.name}.align.jsonl").write_text(jsonl(b.aligns.values()), encoding="utf-8")
    w = Work()
    w.work, w.cuts, w.builders = work, cuts, builders
    w.units = {u["id"]: u for b in builders for u in b.units}
    w.aligns = {uid: a for b in builders for uid, a in b.aligns.items()}
    w.audit = []
    for k, c in enumerate(cuts):
        verdict = "drop" if k in drops else ("review" if k in reviews else "ok")
        reasons = ["cer", "clipping"] if k == drops[-1] else (["cer"] if verdict == "drop" else
                                                              ["edge_noise_start"] if verdict == "review" else [])
        w.audit.append({"id": c["id"], "part": c["part"], "sha256": c["sha256"], "verdict": verdict, "reasons": reasons,
                        "metrics": {"duration_s": 1.0, "cer": 0.5 if verdict == "drop" else 0.01}, "hyp": ""})
    write_audit(w)
    return w


def write_audit(w):
    for part in PARTS:
        (w.work / "audit" / f"{part}.audit.jsonl").write_text(jsonl([a for a in w.audit if a["part"] == part]))


def manifest_rows(work):
    return {s: read_jsonl(work / "manifest" / f"{s}.jsonl") for s in SPLITS}


def all_rows(work):
    return [r for rows in manifest_rows(work).values() for r in rows]


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    w = build_work(tmp_path_factory.mktemp("base"), bad_words=(1,))
    w.report = run_manifest(w.work, PARTS)
    w.rows = {r["id"]: r for r in all_rows(w.work)}
    w.dropped = {r["id"]: r for r in read_jsonl(w.work / "manifest" / "dropped.jsonl")}
    return w


def test_tempo_from_words_matches_the_spec_and_tempo_from_alignment():
    d = [round(x * SR) for x in (0.30, 0.25, 0.40)]
    g1, g2 = round(0.05 * SR), round(0.30 * SR)
    words, t = [], 0
    for dur, gap in zip(d, (g1, g2, 0)):
        words.append((t, t + dur))
        t += dur + gap
    word_s = sum(d) / SR
    lab = tempo_from_words(words, 12, SR, 0.12)
    assert lab.articulation_rate == pytest.approx(12 / (word_s + 0.05), rel=1e-9)
    assert lab.pause_ratio == pytest.approx(0.30 / (word_s + 0.05 + 0.30), rel=1e-9)
    ref = tempo_from_alignment(["phone"] * 12 + ["boundary", "boundary"], [sum(d) / 12] * 12 + [g1, g2], 1 / SR, 0.12)
    assert (lab.articulation_rate, lab.pause_ratio) == pytest.approx((ref.articulation_rate, ref.pause_ratio), rel=1e-9)
    shifted = tempo_from_words([(a + 3600, b + 3600) for a, b in words], 12, SR, 0.12)
    assert (shifted.articulation_rate, shifted.pause_ratio) == pytest.approx((lab.articulation_rate, lab.pause_ratio))
    one = tempo_from_words([(0, 24000)], 10, SR, 0.12)
    assert one.pause_ratio == 0.0 and one.articulation_rate == pytest.approx(10.0)
    with pytest.raises(ValueError):
        tempo_from_words(words, 0, SR, 0.12)


def test_rates_exclude_the_pads(base):
    single = next(c for c in base.cuts if len(c["units"]) == 1 and "articulation_rate" in base.rows.get(c["id"], {}) and "." not in c["id"])
    row = base.rows[single["id"]]
    n_phones = sum(t.kind == "phone" for t in G2P.tokenize(row["text"]))
    assert row["articulation_rate"] == pytest.approx(n_phones / (5 * 0.22 + 4 * 0.06), rel=1e-3)
    assert row["pause_ratio"] == 0.0
    merged = next(c for c in base.cuts if len(c["units"]) == 2 and c["id"] in base.rows)
    row = base.rows[merged["id"]]
    n_phones = sum(t.kind == "phone" for t in G2P.tokenize(row["text"]))
    speech = 10 * 0.22 + 8 * 0.06
    assert row["articulation_rate"] == pytest.approx(n_phones / speech, rel=1e-3)
    assert row["pause_ratio"] == pytest.approx(0.2 / (speech + 0.2), rel=1e-3)


def test_clip_word_times_merged_piece_and_mismatch(base):
    sa = base.aligns
    merged = next(c for c in base.cuts if len(c["units"]) == 2)
    got = clip_word_times(merged, sa, SR)
    want = [(round(w["start"] * SR), round(w["end"] * SR)) for u in merged["units"] for w in sa[u]["words"]]
    assert got == want and len(got) == 10 and got == sorted(got)
    p1 = next(c for c in base.cuts if c["id"].endswith(".1"))
    p2 = next(c for c in base.cuts if c["id"].endswith(".2"))
    parent = sa[p1["id"].split(".")[0]]["words"]
    to_s = lambda ws: [(round(w["start"] * SR), round(w["end"] * SR)) for w in ws]
    assert clip_word_times(p1, sa, SR) == to_s(parent[:5])
    assert clip_word_times(p2, sa, SR) == to_s(parent[5:])
    assert clip_word_times({**merged, "text": merged["text"] + " navíc"}, sa, SR) is None


def test_rows_keys_types_and_values(base):
    units = base.units
    int_like = (int,)
    types = {"id": str, "audio": str, "text": str, "lang": str, "speaker": str, "session": str, "boundary": str,
             "duration_s": float, "gap_before_s": (float, type(None)), "gap_after_s": (float, type(None)),
             "tags": list, "split": str, "audit": str, "audit_reasons": list, "cer": (float, type(None)),
             "units": list, "paragraph": int_like, "chapter": int_like, "flags": list, "n_tokens": int, "sha256": str}
    cuts = {c["id"]: c for c in base.cuts}
    audit = {a["id"]: a for a in base.audit}
    assert len(base.rows) > 20
    for split, rows in manifest_rows(base.work).items():
        for r in rows:
            c = cuts[r["id"]]
            rate_keys = {"articulation_rate", "pause_ratio"}
            assert set(r) - rate_keys == set(types), r["id"]
            assert (set(r) & rate_keys) in (set(), rate_keys)
            for k, t in types.items():
                assert isinstance(r[k], t), (r["id"], k)
            assert r["split"] == split and r["lang"] == "cs" and r["speaker"] == "narrator"
            assert r["session"] == c["part"] and r["text"] == c["text"] and r["units"] == c["units"]
            assert r["boundary"] == c["boundary"] and r["boundary"] in {"mid", "para_end", "chapter_end"}
            assert re.fullmatch(r"[A-Za-z0-9._-]+", r["id"])
            assert Path(r["audio"]).is_absolute() and Path(r["audio"]).exists()
            assert Path(r["audio"]).resolve() == (base.work / "cut" / "clips" / f"{r['id']}.wav").resolve()
            info = sf.info(r["audio"])
            assert r["duration_s"] == info.frames / info.samplerate
            assert r["gap_before_s"] == c["gap_before_s"] and r["gap_after_s"] == c["gap_after_s"]
            assert r["flags"] == c["flags"] and r["sha256"] == c["sha256"]
            assert r["audit"] in ("ok", "review") and r["audit"] == audit[r["id"]]["verdict"]
            assert r["audit_reasons"] == audit[r["id"]]["reasons"] and r["cer"] == audit[r["id"]]["metrics"]["cer"]
            first = units[c["units"][0].split(".")[0]]
            assert (r["chapter"], r["paragraph"]) == (first["chapter"], first["paragraph"])
            assert r["tags"] == sorted({t for u in c["units"] for t in units[u.split(".")[0]]["tags"]})
            assert r["n_tokens"] == len(G2P.tokenize(r["text"]))
    assert len(base.rows) == sum(len(v) for v in manifest_rows(base.work).values())


def test_duration_comes_from_the_written_wav(tmp_path):
    w = build_work(tmp_path, lie=(2,))
    run_manifest(w.work, PARTS)
    row = next(r for r in all_rows(w.work) if r["id"] == w.cuts[2]["id"])
    claimed = (w.cuts[2]["end_sample"] - w.cuts[2]["start_sample"]) / SR
    assert row["duration_s"] == sf.info(row["audio"]).frames / SR and row["duration_s"] != claimed


def test_waterfall_dropped_and_filter(base):
    drops = [a for a in base.audit if a["verdict"] == "drop"]
    assert len(drops) == 3
    in_manifest = set(base.rows)
    for a in drops:
        assert a["id"] not in in_manifest
        assert base.dropped[a["id"]]["stage"] == "audit" and base.dropped[a["id"]]["reasons"] == a["reasons"]
    short, long_, impossible = (base.cuts[-3]["id"], base.cuts[-2]["id"], base.cuts[-1]["id"])
    for cid, reason in ((short, "too_short"), (long_, "too_long"), (impossible, "impossible_alignment")):
        assert cid not in in_manifest
        assert base.dropped[cid]["stage"] == "filter" and base.dropped[cid]["reasons"] == [reason]
    assert all(set(d) == {"id", "stage", "reasons"} for d in base.dropped.values())
    rep = base.report
    total = rep["total"]
    assert total["clips"] == len(base.cuts) == 41 and total["audit_dropped"] == 3 and total["g2p_errors"] == 0
    assert total["filtered"] == {"too_long": 1, "too_short": 1, "impossible_alignment": 1}
    assert total["clips"] == (total["audit_dropped"] + total["g2p_errors"] + sum(total["filtered"].values())
                              + total["text_overlap"] + sum(total[s] for s in SPLITS))
    assert total["units"] == len(base.units) and total["cut_rejected_units"] == {"low_conf": 2}
    for key in ("units", "clips", "audit_dropped", "g2p_errors", "text_overlap", *SPLITS):
        assert total[key] == sum(rep["parts"][p][key] for p in PARTS)
    for p in PARTS:
        part = rep["parts"][p]
        assert part["clips"] == (part["audit_dropped"] + part["g2p_errors"] + sum(part["filtered"].values())
                                 + part["text_overlap"] + sum(part[s] for s in SPLITS))
    assert all(total[s] > 0 for s in SPLITS)
    assert {s: len(rows) for s, rows in manifest_rows(base.work).items()} == {s: total[s] for s in SPLITS}


def test_round_trip_through_dataset_and_collator(base):
    from src.data.audio import AudioProcessor
    from src.data.dataset import BOUNDARY_IDS, TTSCollator, TTSDataset, filter_manifest, load_manifest
    from src.text.phonemizer import build_vocabulary

    rows = [r for r in load_manifest(str(base.work / "manifest" / "train.jsonl")) if "articulation_rate" in r][:4]
    assert len(rows) == 4 and all("audit" in r and "sha256" in r and "units" in r for r in rows)
    n_tokens = {r["id"]: len(G2P.tokenize(r["text"])) for r in rows}
    kept, report = filter_manifest(rows, n_tokens, 256 / SR)
    assert len(kept) == 4 and not (report.too_long or report.too_short or report.impossible_alignment)
    vocab = build_vocabulary({"cs": [r["text"] for r in rows]})
    ds = TTSDataset(kept, AudioProcessor(), vocab, {"cs": G2P}, speaker_ids={"narrator": 0})
    batch = TTSCollator()([ds[i] for i in range(4)])
    assert batch["ids"] == [r["id"] for r in rows]
    assert batch["boundary_ids"].tolist() == [BOUNDARY_IDS[r["boundary"]] for r in rows]
    assert batch["rates"].flatten().tolist() == pytest.approx([x for r in rows for x in (r["articulation_rate"], r["pause_ratio"])])
    assert batch["phoneme_lengths"].tolist() == [r["n_tokens"] for r in rows]


def test_split_invariants_and_book_order(base):
    manifests = manifest_rows(base.work)
    ids = {s: {r["id"] for r in rows} for s, rows in manifests.items()}
    shas = {s: {r["sha256"] for r in rows} for s, rows in manifests.items()}
    groups = {s: {(r["session"], r["chapter"], r["paragraph"]) for r in rows} for s, rows in manifests.items()}
    for a, b in itertools.combinations(SPLITS, 2):
        assert not ids[a] & ids[b] and not shas[a] & shas[b] and not groups[a] & groups[b]
    order = {c["id"]: i for i, c in enumerate(base.cuts)}
    for rows in list(manifests.values()) + [list(base.dropped.values())]:
        positions = [order[r["id"]] for r in rows]
        assert positions == sorted(positions)
    kept = sorted(order[r["id"]] for r in base.rows.values())
    pos = [kept.index(order[r["id"]]) for r in manifests["heldout"]]
    assert pos and pos == list(range(pos[0], pos[-1] + 1))
    assert base.report["split_checks"] == {"no_group_in_two_splits": True, "no_shared_audio": True,
                                           "heldout_contiguous": True}
    assert len({(r["session"], r["chapter"]) for r in manifests["heldout"]}) == 1


def test_hash_recheck_stale_audit_and_missing_audit_raise(tmp_path):
    w = build_work(tmp_path / "a")
    victim = w.cuts[0]["id"]
    wav = w.work / "cut" / "clips" / f"{victim}.wav"
    pcm, _ = sf.read(str(wav), dtype="int16")
    sf.write(str(wav), np.roll(pcm, 7), SR, subtype="PCM_16")
    with pytest.raises(ValueError, match=victim):
        run_manifest(w.work, PARTS)
    w = build_work(tmp_path / "b")
    w.audit[0]["sha256"] = "0" * 64
    write_audit(w)
    with pytest.raises(ValueError, match=w.cuts[0]["id"]):
        run_manifest(w.work, PARTS)
    w = build_work(tmp_path / "c")
    gone = w.audit.pop(5)
    write_audit(w)
    with pytest.raises(ValueError, match=gone["id"]):
        run_manifest(w.work, PARTS)


def test_reports_have_no_text_and_runs_are_byte_identical(base):
    out = base.work / "manifest"
    blobs = "".join((out / n).read_text(encoding="utf-8") for n in ("report.json", "report.md", "coverage.json",
                                                                      "coverage.md"))
    for c in base.cuts:
        assert c["text"] not in blobs
    names = [f"{s}.jsonl" for s in SPLITS] + ["dropped.jsonl", "coverage.json"]
    before = {n: (out / n).read_bytes() for n in names}
    run_manifest(base.work, PARTS)
    assert before == {n: (out / n).read_bytes() for n in names}


def test_rows_without_word_times_omit_the_rates(base):
    bad = base.cuts[1]["id"]
    assert bad in base.rows and "articulation_rate" not in base.rows[bad] and "pause_ratio" not in base.rows[bad]
    assert base.report["rows_without_word_times"] == 1
    assert all("articulation_rate" in r for i, r in base.rows.items() if i != bad)
    for s in SPLITS:
        med = base.report["rate_medians"][s]
        assert set(med) == {"articulation_rate", "pause_ratio"}


def test_heldout_holes_count_the_dropped_clips_inside_its_span(base, tmp_path):
    held = manifest_rows(base.work)["heldout"]
    order = [c["id"] for c in base.cuts]
    at = sorted(order.index(r["id"]) for r in held)
    holes = base.report["heldout_holes"]
    assert holes["span_clips"] == at[-1] - at[0] + 1
    assert holes["dropped_clips"] == holes["span_clips"] - len(held) > 0
    assert holes["cut_rejected_units"] == 0
    from src.data.prepare.coverage import SplitConfig

    none = run_manifest(build_work(tmp_path).work, PARTS, ManifestConfig(split=SplitConfig(heldout_share=2.0)))
    assert none["heldout"] is None and none["heldout_holes"] is None


def test_report_blocks(base):
    rep = base.report
    assert set(rep["hours"]) == set(SPLITS) and rep["hours"]["train"] > 0
    assert rep["config"]["language"] == "cs" and "unmet" in rep and "heldout" in rep and "seconds" in rep
    assert rep["heldout"]["clips"] == len(manifest_rows(base.work)["heldout"])
    assert json.loads((base.work / "manifest" / "report.json").read_text())["total"] == rep["total"]


def test_flagged_and_fragment_clips_never_enter_val_or_test(tmp_path):
    w = build_work(tmp_path, flagged=(10, 11, 12), fragments=(13, 14, 15))
    rep = run_manifest(w.work, PARTS)
    rows = manifest_rows(w.work)
    for r in rows["val"] + rows["test"]:
        assert not set(r["flags"]) & {"short_pause", "no_quiet_cut", "over_cap"}, r["id"]
        assert not set(r["tags"]) & {"part_fragment", "over_cap"}, r["id"]
    assert {w.cuts[k]["id"] for k in (10, 11, 12, 13, 14, 15)} <= {r["id"] for r in all_rows(w.work)}
    for split in ("val", "test"):
        assert {"flagged_rare", "part_fragment"} <= set(rep["unmet"][split])


def test_text_overlap_exclusions_are_listed_and_left_out(tmp_path, monkeypatch):
    from src.data.prepare import manifest as mf

    real, seen = mf.make_split, {}

    def patched(items, cfg):
        res = real(items, cfg)
        seen["victim"] = next(i for i, split in res.assignment.items() if split == "train")
        del res.assignment[seen["victim"]]
        res.excluded.append({"id": seen["victim"], "reason": "text_overlap"})
        return res

    monkeypatch.setattr(mf, "make_split", patched)
    w = build_work(tmp_path)
    rep = mf.run_manifest(w.work, PARTS)
    assert seen["victim"] not in {r["id"] for r in all_rows(w.work)}
    dropped = {r["id"]: r for r in read_jsonl(w.work / "manifest" / "dropped.jsonl")}
    assert dropped[seen["victim"]] == {"id": seen["victim"], "stage": "split", "reasons": ["text_overlap"]}
    total = rep["total"]
    assert total["text_overlap"] == 1
    assert total["clips"] == (total["audit_dropped"] + total["g2p_errors"] + sum(total["filtered"].values())
                              + total["text_overlap"] + sum(total[s] for s in SPLITS))


def test_a_leaking_split_stops_the_stage(tmp_path, monkeypatch):
    from src.data.prepare import manifest as mf
    from src.data.prepare.coverage import SplitError

    real = mf.make_split

    def leaky(items, cfg):
        res = real(items, cfg)
        groups = {}
        for x in items:
            if res.assignment.get(x["id"]) == "train":
                groups.setdefault((x["part"], x["chapter"], x["paragraph"]), []).append(x["id"])
        res.assignment[next(ids for ids in groups.values() if len(ids) > 1)[0]] = "val"
        return res

    monkeypatch.setattr(mf, "make_split", leaky)
    with pytest.raises(SplitError):
        mf.run_manifest(build_work(tmp_path).work, PARTS)


def test_light_import():
    code = "import sys, src.data.prepare.manifest; print(int('torch' in sys.modules or 'transformers' in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(Path(__file__).resolve().parents[1]))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "0"
