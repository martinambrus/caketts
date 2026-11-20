import gc
import hashlib
import importlib.util
import json
import os
import weakref
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

from src.data.prepare.align.ctc import BACKENDS
from src.data.prepare.align.types import WordSpan
from src.data.prepare.asr import ASR_BACKENDS, AsrWord
from src.data.prepare.text import TextStageError

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_corpus.py"
spec = importlib.util.spec_from_file_location("prepare_corpus", SCRIPT)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)

SR = 16000
STAGES = ["source", "lexicon", "text", "audio", "align", "cut", "audit", "manifest", "gold"]


def write_cfg(tmp_path, **data):
    data.setdefault("parts", ["pA", "pB"])
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


def recording(monkeypatch, names=STAGES, fail=None):
    calls = []

    def make(name):
        def run(cfg, hooks):
            calls.append(name)
            if fail and (name in fail):
                raise fail[name]
        return run

    for n in names:
        monkeypatch.setitem(prep.RUNNERS, n, make(n))
    return calls


def test_cli_rejects_unknown_stage_and_missing_config(tmp_path):
    with pytest.raises(SystemExit) as e:
        prep.main(["nonsense", "--config", str(write_cfg(tmp_path))])
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        prep.main(["text"])
    assert e.value.code == 2


def test_cli_help_lists_the_stages(capsys):
    with pytest.raises(SystemExit) as e:
        prep.main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert all(s in out for s in STAGES + ["all"])


def test_unknown_yaml_key_returns_2_and_runs_nothing(tmp_path, monkeypatch, capsys):
    calls = recording(monkeypatch)
    cfg = write_cfg(tmp_path, align={"nonsense_key": 1})
    assert prep.main(["all", "--config", str(cfg)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "align.nonsense_key" in err
    assert calls == []


def test_all_runs_the_stages_in_order_and_skips_absent_sections(tmp_path, monkeypatch):
    calls = recording(monkeypatch)
    assert prep.main(["all", "--config", str(write_cfg(tmp_path))]) == 0
    assert calls == ["text", "audio", "align", "cut", "audit", "manifest"]


def test_all_runs_the_present_optional_sections(tmp_path, monkeypatch, capsys):
    calls = recording(monkeypatch)
    cfg = write_cfg(tmp_path, source={"epub": {"path": "b.epub"}, "orig_dir": "raw"},
                    lexicon={"rewrite_dir": "raw"}, gold={"file": "g.json"})
    assert prep.main(["all", "--config", str(cfg)]) == 0
    assert calls == STAGES
    out = capsys.readouterr().out.splitlines()
    assert [ln.split(":")[0] for ln in out] == STAGES
    assert all(ln.endswith(" s") and "done in" in ln for ln in out)


def test_a_pipeline_error_stops_all(tmp_path, monkeypatch, capsys):
    calls = recording(monkeypatch, fail={"audio": prep.PipelineError("boom")})
    assert prep.main(["all", "--config", str(write_cfg(tmp_path))]) == 2
    assert calls == ["text", "audio"]
    assert "error: boom" in capsys.readouterr().err


def test_single_stage_runs_only_that_stage(tmp_path, monkeypatch):
    calls = recording(monkeypatch)
    assert prep.main(["cut", "--config", str(write_cfg(tmp_path))]) == 0
    assert calls == ["cut"]


def touch(path, text=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_each_input_check_names_the_missing_file_and_the_stage(tmp_path, capsys):
    cfg = write_cfg(tmp_path, parts=["pA"])
    work = tmp_path / "work"
    cases = [
        ("align", work / "text" / "pA.units.jsonl", "text"),
        ("align", work / "audio" / "pA.16k.wav", "audio"),
        ("cut", work / "text" / "pA.units.jsonl", "text"),
        ("cut", work / "align" / "pA.align.jsonl", "align"),
        ("cut", work / "audio" / "pA.24k.wav", "audio"),
        ("audit", work / "cut" / "pA.cuts.jsonl", "cut"),
        ("manifest", work / "text" / "pA.units.jsonl", "text"),
        ("manifest", work / "cut" / "pA.cuts.jsonl", "cut"),
        ("manifest", work / "align" / "pA.align.jsonl", "align"),
        ("manifest", work / "audit" / "pA.audit.jsonl", "audit"),
    ]
    for stage, missing, run_first in cases:
        needs = {"align": ["text/pA.units.jsonl", "audio/pA.16k.wav"],
                 "cut": ["text/pA.units.jsonl", "align/pA.align.jsonl", "audio/pA.24k.wav"],
                 "audit": ["cut/pA.cuts.jsonl"],
                 "manifest": ["text/pA.units.jsonl", "cut/pA.cuts.jsonl", "align/pA.align.jsonl",
                              "audit/pA.audit.jsonl"]}[stage]
        for rel in needs:
            target = work / rel
            if target != missing:
                touch(target)
            elif target.exists():
                target.unlink()
        assert prep.main([stage, "--config", str(cfg)]) == 2
        err = capsys.readouterr().err
        assert err.startswith("error: ") and str(missing) in err and f"{run_first} stage" in err, (stage, err)


def test_stages_without_their_section(tmp_path, capsys):
    cfg = write_cfg(tmp_path)
    for stage in ("gold", "source", "lexicon"):
        assert prep.main([stage, "--config", str(cfg)]) == 2
        assert capsys.readouterr().err == f"error: {stage}: the config has no {stage} section\n"


def test_text_stage_error_returns_1_with_the_failures(tmp_path, monkeypatch, capsys):
    failures = [{"part": "pA", "line": n, "block": 1, "codepoints": ["U+0001"]} for n in range(1, 31)]
    recording(monkeypatch, fail={"text": TextStageError("text stage failed: 30", failures)})
    assert prep.main(["text", "--config", str(write_cfg(tmp_path))]) == 1
    lines = capsys.readouterr().err.splitlines()
    assert lines[0] == "text stage failed: 30"
    assert lines[1:] == [f"pA {n} U+0001" for n in range(1, 21)]


def scene(name, n_words, per_unit, step=0.5, dur=0.4, tail=0.4):
    words = [f"slovo{name}{k}" for k in range(n_words)]
    truth = [WordSpan(w, 0.3 + k * step, 0.3 + k * step + dur, 0.9) for k, w in enumerate(words)]
    units = [{"id": f"{name}-{u:05d}", "text": "x", "words": words[u * per_unit:(u + 1) * per_unit]}
             for u in range(n_words // per_unit)]
    spans = {u["id"]: truth[i * per_unit:(i + 1) * per_unit] for i, u in enumerate(units)}
    duration = truth[-1].end + tail
    return units, spans, [AsrWord(s.text, s.start + 0.02, s.end - 0.02) for s in truth], duration


def write_wav(path, duration, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    sf.write(str(path), (rng.standard_normal(int(duration * SR)) * 0.05).astype(np.float32), SR,
             subtype="PCM_16")


class World:
    def __init__(self, tmp_path):
        self.work = tmp_path / "work"
        self.events = []
        self.units, self.spans, self.asr_words, self.durations = {}, {}, {}, {}
        for name, n_words in (("pA", 4), ("pB", 9)):
            u, s, a, d = scene(name, n_words, *((3,) if name == "pB" else (2, 0.3, 0.25)))
            self.units[name], self.asr_words[name], self.durations[name] = u, a, d
            self.spans.update(s)
            write_wav(self.work / "audio" / f"{name}.16k.wav", d, seed=len(name) + n_words)
            touch(self.work / "text" / f"{name}.units.jsonl",
                  "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in u))
        self.by_len = {int(round(d * SR)): self.asr_words[n] for n, d in self.durations.items()}
        self.align_calls = 0
        self.asr_calls = 0

    def backend(self):
        world = self

        class Backend:
            model_id = "stub/aligner"
            stride_s = 0.02

            def align_block(self, audio, t0, ids, words_per_unit):
                world.events.append("align_block")
                world.align_calls += 1
                return [list(world.spans[i]) for i in ids]

        return Backend()

    def asr(self, model_id="stub/asr"):
        world = self

        class Asr:
            def transcribe_words(self, audio):
                world.events.append("asr")
                world.asr_calls += 1
                return world.by_len[len(audio)]

        a = Asr()
        a.model_id = model_id
        return a


def align_cfg(tmp_path, **align):
    return write_cfg(tmp_path, align={"block_max_s": 2.0, "margin_s": 0.5, **align})


def run_align(tmp_path, world, cfg, asr=None, backend=None):
    return prep.main(["align", "--config", str(cfg)], align_backend=backend or world.backend(),
                     align_asr=asr or world.asr())


def read_tree(directory):
    return {p.name: p.read_bytes() for p in sorted(Path(directory).iterdir())}


def test_align_writes_the_contract_files_and_is_byte_identical(tmp_path):
    world = World(tmp_path)
    cfg = align_cfg(tmp_path)
    assert run_align(tmp_path, world, cfg) == 0
    out = world.work / "align"
    assert {p.name for p in out.iterdir()} == {"pA.align.jsonl", "pA.meta.json", "pB.align.jsonl", "pB.meta.json",
                                               "pB.asr.json", "report.json", "report.md"}
    rows = [json.loads(ln) for ln in (out / "pB.align.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == [u["id"] for u in world.units["pB"]]
    assert all(set(r) == {"id", "start_s", "end_s", "words", "score_mean", "score_min", "status", "flags"}
               for r in rows)
    assert all(r["status"] == "ok" for r in rows)
    assert (out / "pB.align.jsonl").read_text().splitlines()[0] == json.dumps(rows[0], ensure_ascii=False,
                                                                               sort_keys=True)
    meta = json.loads((out / "pA.meta.json").read_text())
    assert set(meta) == {"gaps", "meta"} and meta["meta"]["model_id"] == "stub/aligner"
    assert (out / "pA.meta.json").read_text() == json.dumps(meta, sort_keys=True, indent=1, ensure_ascii=False) + "\n"
    report_json = json.loads((out / "report.json").read_text())
    assert report_json["config"]["block_max_s"] == 2.0
    assert report_json["parts"]["pA"]["units"] == 2 and report_json["parts"]["pA"]["asr_words"] is None
    assert report_json["parts"]["pA"]["asr_cache"] is None
    assert report_json["parts"]["pB"]["asr_cache"] == "miss" and report_json["parts"]["pB"]["asr_words"] == 9
    assert report_json["parts"]["pB"]["status"] == {"ok": 3}
    assert report_json["total"]["units"] == 5
    for text in ("report.json", "report.md"):
        assert "slovo" not in (out / text).read_text()
    first = {k: v for k, v in read_tree(out).items() if not k.startswith("report")}
    assert run_align(tmp_path, world, cfg) == 0
    assert {k: v for k, v in read_tree(out).items() if not k.startswith("report")} == first


def test_align_asr_cache_content_and_invalidation(tmp_path):
    world = World(tmp_path)
    cfg = align_cfg(tmp_path)
    run_align(tmp_path, world, cfg)
    assert world.asr_calls == 1
    cache = json.loads((world.work / "align" / "pB.asr.json").read_text())
    wav = (world.work / "audio" / "pB.16k.wav").read_bytes()
    assert cache == {"audio_sha256": hashlib.sha256(wav).hexdigest(),
                     "config": {"window_s": 35.0, "overlap_s": 1.0, "cut_search_s": 3.0},
                     "model_id": "stub/asr",
                     "words": [[w.text, w.start, w.end] for w in world.asr_words["pB"]]}
    run_align(tmp_path, world, cfg)
    assert world.asr_calls == 1
    assert json.loads((world.work / "align" / "report.json").read_text())["parts"]["pB"]["asr_cache"] == "hit"
    run_align(tmp_path, world, cfg, asr=world.asr("stub/asr2"))
    assert world.asr_calls == 2
    run_align(tmp_path, world, align_cfg(tmp_path, asr={"window_s": 20.0}))
    assert world.asr_calls == 3
    run_align(tmp_path, world, align_cfg(tmp_path, asr={"window_s": 20.0}))
    assert world.asr_calls == 3
    write_wav(world.work / "audio" / "pB.16k.wav", world.durations["pB"], seed=99)
    run_align(tmp_path, world, align_cfg(tmp_path, asr={"window_s": 20.0}))
    assert world.asr_calls == 4


def test_align_asr_pass_comes_before_any_alignment(tmp_path):
    world = World(tmp_path)
    cfg = align_cfg(tmp_path, longform="always")
    run_align(tmp_path, world, cfg)
    assert world.asr_calls == 2 and world.align_calls >= 2
    first_align = world.events.index("align_block")
    assert world.events[:first_align] == ["asr", "asr"] and "asr" not in world.events[first_align:]


def test_align_a_short_part_makes_no_asr_call(tmp_path):
    world = World(tmp_path)
    run_align(tmp_path, world, align_cfg(tmp_path))
    assert world.asr_calls == 1
    assert not (world.work / "align" / "pA.asr.json").exists()


def test_align_stage_created_runner_is_made_once_and_closed_before_ctc(tmp_path, monkeypatch):
    import src.data.prepare.asr as asr_module

    world = World(tmp_path)
    made = []

    class FakeRunner:
        model_id = ASR_BACKENDS["parakeet-tdt-0.6b-v3"].model_id

        def __init__(self, cfg):
            made.append(cfg)
            world.events.append("init")

        def transcribe_words(self, audio):
            world.events.append("asr")
            return world.by_len[len(audio)]

        def close(self):
            world.events.append("close")

    monkeypatch.setattr(asr_module, "AsrRunner", FakeRunner)
    cfg = align_cfg(tmp_path, longform="always")
    assert prep.main(["align", "--config", str(cfg)], align_backend=world.backend()) == 0
    assert len(made) == 1
    first_align = world.events.index("align_block")
    assert world.events[:first_align] == ["init", "asr", "asr", "close"]
    assert world.events.count("close") == 1
    meta = json.loads((world.work / "align" / "pA.meta.json").read_text())["meta"]
    assert meta["asr_model_id"] == FakeRunner.model_id


def test_align_hook_asr_is_never_closed(tmp_path):
    world = World(tmp_path)
    asr = world.asr()
    asr.close = lambda: world.events.append("close")
    run_align(tmp_path, world, align_cfg(tmp_path), asr=asr)
    assert "close" not in world.events


def test_align_creates_the_ctc_aligner_once_and_drops_it(tmp_path, monkeypatch):
    import src.data.prepare.align.ctc as ctc_module

    world = World(tmp_path)
    made, refs = [], []
    inner = world.backend()

    class FakeCtc:
        model_id = inner.model_id
        stride_s = inner.stride_s

        def __init__(self, *args):
            made.append(args)
            refs.append(weakref.ref(self))

        def align_block(self, *args):
            return inner.align_block(*args)

    monkeypatch.setattr(ctc_module, "CtcAligner", FakeCtc)
    assert prep.main(["align", "--config", str(align_cfg(tmp_path))], align_asr=world.asr()) == 0
    assert made == [(BACKENDS["mms-300m"], "cs", "auto", 30.0, 2.0)]
    gc.collect()
    assert refs[0]() is None


def audit_world(tmp_path, model_id="stub/aligner"):
    work = tmp_path / "work"
    touch(work / "cut" / "pA.cuts.jsonl")
    touch(work / "align" / "pA.meta.json", json.dumps({"gaps": [], "meta": {"model_id": model_id}}))
    return write_cfg(tmp_path, parts=["pA"], audit={"judges": ["parakeet-tdt-0.6b-v3"]})


class Judge:
    def __init__(self, model_id):
        self.model_id, self.calls = model_id, 0

    def transcribe(self, clips):
        self.calls += 1
        return ["" for _ in clips]


def test_audit_refuses_a_judge_equal_to_the_aligner(tmp_path):
    cfg = audit_world(tmp_path)
    judge = Judge("stub/aligner")
    with pytest.raises((prep.PipelineError, ValueError), match="different models"):
        prep.main(["audit", "--config", str(cfg)], audit_runners={"parakeet-tdt-0.6b-v3": judge})
    assert judge.calls == 0


def test_audit_gets_the_aligner_id_from_the_align_meta(tmp_path, monkeypatch):
    import src.data.prepare.audit as audit_module

    cfg = audit_world(tmp_path)
    seen = {}
    monkeypatch.setattr(audit_module, "run_audit", lambda *a, **k: seen.update(args=a, kwargs=k) or {})
    judge = Judge("stub/other")
    assert prep.main(["audit", "--config", str(cfg)], audit_runners={"parakeet-tdt-0.6b-v3": judge}) == 0
    assert seen["kwargs"]["aligner_model_id"] == "stub/aligner"
    assert seen["kwargs"]["runner"] == {"parakeet-tdt-0.6b-v3": judge}


def test_audit_falls_back_to_the_configured_backend_without_meta(tmp_path, monkeypatch):
    import src.data.prepare.audit as audit_module

    cfg = audit_world(tmp_path)
    (tmp_path / "work" / "align" / "pA.meta.json").unlink()
    seen = {}
    monkeypatch.setattr(audit_module, "run_audit", lambda *a, **k: seen.update(kwargs=k) or {})
    assert prep.main(["audit", "--config", str(cfg)]) == 0
    assert seen["kwargs"]["aligner_model_id"] == BACKENDS["mms-300m"].model_id
    assert seen["kwargs"]["runner"] is None


def test_audit_refuses_differing_aligner_ids_between_parts(tmp_path, capsys):
    work = tmp_path / "work"
    for name, mid in (("pA", "one"), ("pB", "two")):
        touch(work / "cut" / f"{name}.cuts.jsonl")
        touch(work / "align" / f"{name}.meta.json", json.dumps({"gaps": [], "meta": {"model_id": mid}}))
    assert prep.main(["audit", "--config", str(write_cfg(tmp_path))]) == 2
    assert capsys.readouterr().err.startswith("error: ")


def test_lexicon_stage_builds_the_normalizer_for_the_language(tmp_path, monkeypatch):
    import src.data.prepare.lexicon as lexicon_module

    book = tmp_path / "book"
    touch(book / "pA.txt", "Dobrý den.\n")
    seen = {}
    monkeypatch.setattr(lexicon_module, "derive_to_file", lambda *a, **k: seen.update(args=a, kwargs=k) or [])
    cfg = write_cfg(tmp_path, language="sk", lexicon={"rewrite_dir": "rw", "book_dir": "book",
                                                      "path": "deep/er/lex.tsv"})
    assert prep.main(["lexicon", "--config", str(cfg)]) == 0
    assert not seen["args"]
    normalizer = seen["kwargs"]["normalizer"]
    assert normalizer is not None and normalizer.language == "sk"
    assert seen["kwargs"]["path"] == str(tmp_path / "deep" / "er" / "lex.tsv")
    assert (tmp_path / "deep" / "er").is_dir()


def test_lexicon_stage_needs_the_book_texts(tmp_path, capsys):
    (tmp_path / "book").mkdir()
    cfg = write_cfg(tmp_path, lexicon={"rewrite_dir": "rw", "book_dir": "book"})
    assert prep.main(["lexicon", "--config", str(cfg)]) == 2
    err = capsys.readouterr().err
    assert str(tmp_path / "book") in err and "source stage" in err


def test_relative_paths_resolve_against_the_yaml_not_the_cwd(tmp_path, monkeypatch):
    (tmp_path / "conf" / "raw").mkdir(parents=True)
    (tmp_path / "conf" / "raw" / "a1.txt.orig").write_text("Dobrý den. Jak se máš?\n", encoding="utf-8")
    (tmp_path / "conf" / "cfg.yaml").write_text(yaml.safe_dump({"raw_dir": "raw", "work_dir": "work",
                                                                "parts": ["a1"]}))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert prep.main(["text", "--config", str(tmp_path / "conf" / "cfg.yaml")]) == 0
    units = tmp_path / "conf" / "work" / "text" / "a1.units.jsonl"
    assert units.is_file() and json.loads(units.read_text().splitlines()[0])["part"] == "a1"
    assert os.listdir(elsewhere) == []


@pytest.mark.parametrize("stage,data", [("audio", {"parts": ["a1"]}), ("text", {"parts": ["a1"]}),
                                        ("source", {"source": {"epub": {"path": "missing.epub"}, "orig_dir": "raw"}})])
def test_a_missing_input_file_is_an_error_line(tmp_path, capsys, stage, data):
    (tmp_path / "raw").mkdir()
    cfg = write_cfg(tmp_path, **data)
    assert prep.main([stage, "--config", str(cfg)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "Traceback" not in err
