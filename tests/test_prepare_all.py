"""End-to-end test of scripts/prepare_corpus.py: every stage on a tiny synthetic book, stub aligner and stub judges."""
import importlib.util
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.data.prepare.align.types import WordSpan
from test_cut import SR, make_part, pz, sp
from test_prepare_source import make_epub, xhtml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_corpus.py"
STAGES = ("source", "lexicon", "text", "audio", "align", "cut", "audit", "manifest", "gold")
DROP_KEYS = {"seconds", "timing", "peak_rss_mib", "peak_rss_mb", "normalise_s", "transcripts_new", "transcripts_cached",
             "by_judge"}
BOOK = {
    "a1": [["Starý rybář seděl u tiché řeky celé ráno.", "Malý chlapec přinesl z lesa košík hub."],
           ["Naše babička chodila do Wellness centra každý týden.", "Černý pes spal pod starým dubovým stolem.",
            "Mladá učitelka vysvětlovala dětem zákony přírody."],
           ["Večer zazvonil zvon na věži kostela.", "Všichni lidé šli domů po dlouhém dni."]],
    "a2": [["Ráno přišla mlha a zahalila celé údolí.", "Velká loď pomalu plula po klidné hladině.",
            "Kapitán zvedl dalekohled a hledal ostrov."],
           ["Sestra uvařila silný čaj s medem a citronem.", "Ve Wellness salonu voněla levandule a máta.",
            "Tichá řeka tekla dál k velkému jezeru."]]}
WRONG_A = "Dlouhá cesta vedla přes hory a doliny do vzdáleného města."
WRONG_BOTH = "Zítra odpoledne pojedeme vlakem k babičce na chalupu za horami."


def load_script():
    spec = importlib.util.spec_from_file_location("prepare_corpus_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sentences(part):
    return [s for para in BOOK[part] for s in para]


def build_book(root):
    raw, cfg = root / "raw", root / "cfg"
    raw.mkdir()
    cfg.mkdir()
    truth, gold_parts, pauses, blocks = {}, {}, [], []
    for n, part in enumerate(BOOK):
        items, k = [pz(0.5)], 0
        for para in BOOK[part]:
            for i, _ in enumerate(para):
                last = i == len(para) - 1
                items += [sp(1.6 + 0.2 * (k % 4), "para_end" if last else "mid")]
                items += [pz(1.5) if last else pz(0.6 + 0.1 * (k % 5))]
                k += 1
        items[-1] = pz(0.5)
        pcm, tr = make_part(items, 100 + n)
        sf.write(str(raw / f"{part}.wav"), np.column_stack((pcm, pcm)), SR, subtype="PCM_16")
        (raw / f"{part}.txt.orig").write_text(
            "\n\n".join(" ".join(p) for p in BOOK[part]).replace("Wellness", "velnes") + "\n", encoding="utf-8")
        truth[part] = tr["speech"]
        gold_parts[part] = [{"clip": f"{part}-{j:05d}", "speech_start": int(a), "speech_end": int(b), "end_reliable": True}
                            for j, (a, b) in enumerate(tr["speech"])]
        pauses += [{"part": part, "start": int(tr["speech"][j][1]), "end": int(tr["speech"][j + 1][0])}
                   for j in range(len(tr["speech"]) - 1)]
        blocks += [("p", " ".join(p)) for p in BOOK[part]]
    assert all(len(truth[p]) == len(sentences(p)) for p in BOOK)
    (root / "book.epub").write_bytes(make_epub({"c1.xhtml": xhtml(*blocks)}))
    (cfg / "gold.json").write_text(json.dumps({"sample_rate": SR, "parts": gold_parts, "pauses": pauses}), encoding="utf-8")
    return truth


def write_yaml(root, name="work", extra="", lexicon="rewrite_dir: ../raw", with_source=True):
    path = root / "cfg" / f"{name}.yaml"
    source = "source: {epub: {path: ../book.epub}, orig_dir: ../raw}\n" if with_source else ""
    lex = f"lexicon: {{{lexicon}}}\n" if with_source else ""
    path.write_text(f"raw_dir: ../raw\nwork_dir: ../{name}\n{source}{lex}align: {{backend: xls-r-cs}}\n"
                    f"gold: {{file: gold.json}}\n{extra}", encoding="utf-8")
    return str(path)


class StubBackend:
    model_id = "stub/aligner"
    stride_s = 0.02

    def __init__(self, truth, shift=0.0):
        self.truth, self.shift = truth, shift

    def align_block(self, audio, t0, ids, words_per_unit):
        out = []
        for uid, words in zip(ids, words_per_unit):
            part, k = uid.rsplit("-", 1)
            s, e = self.truth[part][int(k)]
            lo, hi = s / SR + 0.04, e / SR - 0.055
            step = (hi - lo) / len(words)
            out.append([WordSpan(w, lo + j * step + self.shift, lo + (j + 0.9) * step + self.shift, 0.9)
                        for j, w in enumerate(words)])
        return out


class StubAsr:
    model_id = "stub/asr"

    def transcribe_words(self, audio):
        raise AssertionError("the ASR stub must not be called for short parts")


class StubJudge:
    def __init__(self, work, model_id, wrong):
        self.work, self.model_id, self.wrong, self.answers, self.pos, self.calls = work, model_id, wrong, None, 0, 0

    def transcribe(self, arrays):
        if self.answers is None:
            self.answers = []
            for part in BOOK:
                path = os.path.join(self.work, "cut", f"{part}.cuts.jsonl")
                if os.path.exists(path):
                    with open(path, encoding="utf-8") as f:
                        self.answers += [json.loads(line)["text"] for line in f if line.strip()]
            for idx, text in self.wrong.items():
                self.answers[idx] = text
        got = self.answers[self.pos:self.pos + len(arrays)]
        self.pos += len(arrays)
        self.calls += 1
        return got


class Run:
    def __init__(self, root, truth, name="work", wrong_a=None, wrong_b=None, shift=0.0, id_a="stub/a", id_b="stub/b",
                 extra="", lexicon="rewrite_dir: ../raw", with_source=True):
        self.root, self.name = root, name
        self.yaml = write_yaml(root, name, extra, lexicon, with_source)
        self.work = str((root / name).resolve())
        self.hooks = {
            "align_backend": StubBackend(truth, shift), "align_asr": StubAsr(),
            "audit_runners": {"parakeet-tdt-0.6b-v3": StubJudge(self.work, id_a, {2: WRONG_A, 6: WRONG_BOTH}
                                                                if wrong_a is None else wrong_a),
                              "canary-1b-v2": StubJudge(self.work, id_b, {4: WRONG_A, 6: WRONG_BOTH}
                                                        if wrong_b is None else wrong_b)}}

    def main(self, stage="all"):
        return load_script().main([stage, "--config", self.yaml], **self.hooks)

    def read(self, rel):
        return Path(self.work, rel).read_text(encoding="utf-8")

    def rows(self, rel):
        return [json.loads(x) for x in self.read(rel).splitlines() if x.strip()]

    def report(self, stage):
        return json.loads(self.read(f"{stage}/report.json"))


def drop_keys(value):
    if isinstance(value, dict):
        return {k: drop_keys(v) for k, v in value.items() if k not in DROP_KEYS}
    if isinstance(value, list):
        return [drop_keys(v) for v in value]
    return value


def assert_sorted_keys(text):
    def check(pairs):
        keys = [k for k, _ in pairs]
        assert keys == sorted(keys)
        return dict(pairs)

    for doc in (text.splitlines() if text.lstrip().startswith('{"') else [text]):
        json.loads(doc, object_pairs_hook=check)


def snapshot(run):
    files, reports = {}, {}
    base = Path(run.work)
    for path in sorted(p for p in base.rglob("*") if p.is_file()):
        rel = str(path.relative_to(base))
        data = path.read_bytes().replace(run.work.encode(), b"<WORK>")
        if path.name == "report.json":
            reports[rel] = drop_keys(json.loads(data))
        elif path.name != "report.md":
            files[rel] = data
    return files, reports


def assert_same(a, b):
    assert sorted(a[0]) == sorted(b[0]) and sorted(a[1]) == sorted(b[1])
    assert [r for r in a[0] if a[0][r] != b[0][r]] == []
    assert [r for r in a[1] if a[1][r] != b[1][r]] == []


@pytest.fixture(scope="module")
def book(tmp_path_factory):
    root = tmp_path_factory.mktemp("book")
    return root, build_book(root)


@pytest.fixture(scope="module")
def good(book):
    root, truth = book
    run = Run(root, truth, "work_all")
    assert run.main() == 0
    return run, snapshot(run)


def test_all_runs_every_stage_and_the_outputs_make_sense(good):
    run, _ = good
    for stage in (s for s in STAGES if s != "lexicon"):
        assert Path(run.work, stage, "report.json").is_file(), stage
    assert Path(run.work, "lexicon.tsv").is_file()
    counts = {"a1": len(sentences("a1")), "a2": len(sentences("a2"))}
    for part, n in counts.items():
        units = run.rows(f"text/{part}.units.jsonl")
        assert len(units) == n and [u["id"] for u in units] == [f"{part}-{i:05d}" for i in range(n)]
    units = run.rows("text/a1.units.jsonl") + run.rows("text/a2.units.jsonl")
    assert sum("velnes" in u["text"].lower() for u in units) == 2 and not any("wellness" in u["text"].lower() for u in units)
    lex = run.read("lexicon.tsv")
    assert lex.splitlines()[1].split("\t") == ["Wellness", "velnes", "respelling", "2", "y", "y"]
    for part in counts:
        assert_sorted_keys(run.read(f"align/{part}.align.jsonl"))
        assert_sorted_keys(run.read(f"align/{part}.meta.json"))
    assert run.report("source")["parts"].keys() == counts.keys()
    cut = run.report("cut")
    assert set(cut["parts"]) == {"a1", "a2"} and sum(c["merged_clips"] for c in cut["parts"].values()) == 0
    assert {p: len(run.rows(f"cut/{p}.cuts.jsonl")) for p in counts} == counts
    audit = run.report("audit")
    assert set(audit["parts"]) == {"a1", "a2"} and audit["total"]["clips"] == 13
    rows = run.rows("audit/a1.audit.jsonl") + run.rows("audit/a2.audit.jsonl")
    verdict = {r["id"]: (r["verdict"], r["reasons"]) for r in rows}
    assert verdict["a1-00006"] == ("drop", ["cer"])
    assert verdict["a1-00002"] == ("review", ["cer_disputed"]) and verdict["a1-00004"] == ("review", ["cer_disputed"])
    assert [i for i, v in verdict.items() if v[0] == "ok"] == [i for i in verdict if i not in ("a1-00002", "a1-00004", "a1-00006")]
    for part in counts:
        for judge in ("parakeet-tdt-0.6b-v3", "canary-1b-v2"):
            assert len(run.rows(f"audit/{part}.asr.{judge}.jsonl")) == counts[part]
    ids = {j: {r["model_id"] for r in run.rows(f"audit/a1.asr.{j}.jsonl")} for j in ("parakeet-tdt-0.6b-v3", "canary-1b-v2")}
    assert ids == {"parakeet-tdt-0.6b-v3": {"stub/a"}, "canary-1b-v2": {"stub/b"}}
    manifest = [r for s in ("train", "val", "test", "heldout") for r in run.rows(f"manifest/{s}.jsonl")]
    dropped = {r["id"]: r for r in run.rows("manifest/dropped.jsonl")}
    assert dropped["a1-00006"]["stage"] == "audit" and "a1-00006" not in {r["id"] for r in manifest}
    assert len({r["sha256"] for r in manifest}) == len(manifest) == 13 - len(dropped)
    assert {r["id"] for r in manifest}.isdisjoint(dropped)
    for r in manifest:
        info = sf.info(r["audio"])
        assert r["duration_s"] == info.frames / info.samplerate
    gold = run.report("gold")["total"]
    assert gold["unsafe_cut_rate"] == 0 and gold["matched"] == gold["gold_spans"] == 13
    assert gold["start_error"]["within_50ms"] == 1.0 and gold["end_error"]["within_50ms"] == 1.0


def test_all_twice_is_byte_identical(good, tmp_path, monkeypatch):
    run, first = good
    monkeypatch.chdir(tmp_path)
    assert run.main() == 0
    assert_same(first, snapshot(run))


def test_stages_one_by_one_equal_all(book, good, tmp_path, monkeypatch):
    root, truth = book
    run = Run(root, truth, "work_steps")
    monkeypatch.chdir(tmp_path)
    for stage in STAGES:
        assert run.main(stage) == 0, stage
    expected = good[1]
    assert_same(expected, snapshot(run))
    for stage in STAGES:
        assert run.main(stage) == 0, stage
        assert_same(expected, snapshot(run))


def test_unknown_yaml_key_and_relative_paths(book, good, tmp_path, monkeypatch, capsys):
    root, truth = book
    bad = Run(root, truth, "work_bad", extra="audit: {judgez: []}\n")
    monkeypatch.chdir(tmp_path)
    capsys.readouterr()
    assert bad.main() == 2
    assert "audit.judgez" in capsys.readouterr().err
    assert not Path(bad.work).exists()
    run = Run(root, truth, "work_rel")
    assert run.main() == 0
    assert Path(root, "work_rel", "manifest", "train.jsonl").is_file()
    assert list(tmp_path.iterdir()) == []
    assert_same(good[1], snapshot(run))


def clean_manifest(run):
    return [r for s in ("train", "val", "test", "heldout") for r in run.rows(f"manifest/{s}.jsonl")]


def test_negative_control_judges_decide_the_audit_outcome(book, good):
    root, truth = book
    minority = {i: WRONG_BOTH for i in (0, 1, 2)}
    run = Run(root, truth, "work_nc_a", wrong_a=minority, wrong_b=minority)
    assert run.main() == 0
    rows = run.rows("audit/a1.audit.jsonl")
    assert [r["verdict"] for r in rows[:3]] == ["drop"] * 3
    dropped = {r["id"] for r in run.rows("manifest/dropped.jsonl")}
    assert {"a1-00000", "a1-00001", "a1-00002"} <= dropped
    assert dropped != {r["id"] for r in good[0].rows("manifest/dropped.jsonl")}
    assert len(clean_manifest(run)) < len(clean_manifest(good[0]))


def test_negative_control_shifted_aligner_degrades_the_gold_report(book, good):
    root, truth = book
    run = Run(root, truth, "work_nc_b", shift=0.4)
    run.main()
    good_gold, bad_gold = good[0].report("gold")["total"], run.report("gold")["total"]
    assert good_gold["matched"] == 13
    assert good_gold["end_error"]["within_50ms"] == 1.0
    assert bad_gold["end_error"]["within_50ms"] < 1.0 and bad_gold["end_error"]["median_ms"] > 100


def test_negative_control_transcripts_are_cached_and_a_new_model_id_retranscribes_that_judge_only(book, good):
    root, truth = book
    run = Run(root, truth, "work_nc_c")
    assert run.main() == 0
    first = snapshot(run)
    assert first[0] == good[1][0] and first[1] == good[1][1]
    every = {i: WRONG_BOTH for i in range(13)}
    run.hooks["audit_runners"] = {"parakeet-tdt-0.6b-v3": StubJudge(run.work, "stub/a", every),
                                  "canary-1b-v2": StubJudge(run.work, "stub/b", every)}
    assert run.main() == 0
    assert all(j.calls == 0 for j in run.hooks["audit_runners"].values())
    assert_same(first, snapshot(run))
    assert run.report("audit")["total"]["by_judge"] == {"canary-1b-v2": {"cached": 13, "new": 0},
                                                       "parakeet-tdt-0.6b-v3": {"cached": 13, "new": 0}}
    run.hooks["audit_runners"] = {"parakeet-tdt-0.6b-v3": StubJudge(run.work, "stub/a2", {2: WRONG_A, 6: WRONG_BOTH}),
                                  "canary-1b-v2": StubJudge(run.work, "stub/b", every)}
    assert run.main() == 0
    audit = run.report("audit")
    assert audit["total"]["by_judge"] == {"canary-1b-v2": {"cached": 13, "new": 0},
                                          "parakeet-tdt-0.6b-v3": {"cached": 0, "new": 13}}
    assert audit["judges"]["parakeet-tdt-0.6b-v3"] == "stub/a2" and run.hooks["audit_runners"]["canary-1b-v2"].calls == 0


def test_all_stops_at_the_first_failing_stage(book, tmp_path, monkeypatch, capsys):
    root, truth = book
    run = Run(root, truth, "work_stop", lexicon="rewrite_dir: ../raw, book_dir: ../cfg")
    monkeypatch.chdir(tmp_path)
    capsys.readouterr()
    assert run.main() == 2
    assert "run the source stage first" in capsys.readouterr().err
    assert sorted(p.name for p in Path(run.work).iterdir()) == ["source"]


def test_top_level_parts_limit_every_stage(book, tmp_path, monkeypatch):
    root, truth = book
    run = Run(root, truth, "work_parts", extra="parts: [a2]\n", wrong_a={}, wrong_b={})
    monkeypatch.chdir(tmp_path)
    assert run.main() == 0
    for stage, suffix in (("text", ".units.jsonl"), ("align", ".align.jsonl"), ("cut", ".cuts.jsonl"),
                          ("audit", ".audit.jsonl")):
        assert sorted(p.name for p in Path(run.work, stage).glob(f"*{suffix}")) == [f"a2{suffix}"], stage
    assert [r["id"] for r in run.rows("text/a2.units.jsonl")] == [f"a2-{i:05d}" for i in range(6)]
    assert run.report("gold")["total"]["gold_spans"] == 6


def test_text_parts_follow_the_recordings_when_no_top_level_parts(book, tmp_path, monkeypatch):
    root, truth = book
    other = tmp_path / "other"
    shutil.copytree(root / "raw", other / "raw")
    shutil.copytree(root / "cfg", other / "cfg", ignore=shutil.ignore_patterns("*.yaml"))
    shutil.copy(other / "raw" / "a2.txt.orig", other / "raw" / "a3.txt.orig")
    run = Run(other, truth, "work_plain", with_source=False)
    monkeypatch.chdir(tmp_path)
    assert run.main() == 0
    assert sorted(p.name for p in Path(run.work, "text").glob("*.units.jsonl")) == ["a1.units.jsonl", "a2.units.jsonl"]
