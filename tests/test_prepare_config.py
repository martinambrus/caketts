import dataclasses
import os
import typing
from pathlib import Path

import pytest
import yaml

from src.data.prepare.audit import AuditConfig
from src.data.prepare.config import ConfigError, PipelineConfig, from_mapping, load_config, validate
from src.data.prepare.manifest import ManifestConfig

EXAMPLE = Path(__file__).resolve().parents[1] / "configs" / "data" / "example.yaml"
SECTIONS = ("source", "lexicon", "text", "audio", "align", "cut", "audit", "manifest", "gold")
FED = {"language", "speaker", "use_part_fragments", "work_dir"}
DERIVED = {
    "text.parts", "text.text_dir", "text.suffix", "text.placement", "text.lexicon", "text.compare_dir",
    "text.compare_suffix", "lexicon.path", "lexicon.book_dir", "audit.edge_db",
}
NO_DEFAULT = {"source.orig_dir", "source.epub.path", "lexicon.rewrite_dir"}


def write(tmp_path, data, name="cfg.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(data) if not isinstance(data, str) else data)
    return p


def norm(v):
    if isinstance(v, (list, tuple)):
        return [norm(x) for x in v]
    if isinstance(v, dict):
        return {k: norm(x) for k, x in v.items()}
    return v


def default_of(f):
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:
        return f.default_factory()
    return dataclasses.MISSING


def dataclass_of(hint):
    args = [a for a in typing.get_args(hint) if a is not type(None)] or [hint]
    return args[0] if dataclasses.is_dataclass(args[0]) else None


def walk(cls, node, path, problems):
    hints = typing.get_type_hints(cls)
    for f in dataclasses.fields(cls):
        p = f"{path}.{f.name}"
        if f.name in FED:
            if f.name in node:
                problems.append(f"{p} must not be in the example")
            continue
        if f.name not in node:
            problems.append(f"{p} missing")
            continue
        sub = dataclass_of(hints[f.name])
        if sub is not None:
            if not isinstance(node[f.name], dict):
                problems.append(f"{p} not a mapping")
            else:
                walk(sub, node[f.name], p, problems)
            continue
        if p in DERIVED:
            if node[f.name] is not None:
                problems.append(f"{p} must be null, is {node[f.name]!r}")
        elif p in NO_DEFAULT:
            pass
        elif norm(node[f.name]) != norm(default_of(f)):
            problems.append(f"{p}: {node[f.name]!r} != default {default_of(f)!r}")
    extra = set(node) - {f.name for f in dataclasses.fields(cls)}
    if extra:
        problems.append(f"{path}: keys not in dataclass {sorted(extra)}")


def test_example_loads_and_is_complete():
    raw = yaml.safe_load(EXAMPLE.read_text())
    cfg = load_config(EXAMPLE)
    assert isinstance(cfg, PipelineConfig)
    assert cfg.source is not None and cfg.lexicon is not None and cfg.gold is not None
    top = {"language": "cs", "speaker": "narrator", "use_part_fragments": False, "raw_dir": "raw", "work_dir": "work", "parts": None}
    assert {k: raw[k] for k in top} == top
    assert set(raw) == set(top) | set(SECTIONS)
    problems = []
    for name in SECTIONS:
        cls = type(getattr(cfg, name))
        walk(cls, raw[name], name, problems)
    assert problems == []


def test_unknown_keys_name_the_dotted_path(tmp_path):
    cases = [
        ({"bogus": 1}, "bogus"),
        ({"cut": {"pad": 1}}, "cut.pad"),
        ({"align": {"asr": {"windows_s": 3.0}}}, "align.asr.windows_s"),
        ({"manifest": {"split": {"sed": 1}}}, "manifest.split.sed"),
        ({"source": {"epub": {"path": "b.epub"}, "parts": [{"name": "a", "start": 1}, {"name": "b", "strat": 2}]}}, "source.parts[1].strat"),
    ]
    for data, path in cases:
        with pytest.raises(ConfigError) as e:
            load_config(write(tmp_path, data))
        msg = str(e.value)
        assert msg.startswith(path), msg
        assert "valid" in msg
    with pytest.raises(ConfigError) as e:
        from_mapping({"cut": {"pad": 1}}, str(tmp_path))
    assert "pad_s" in str(e.value)


@pytest.mark.parametrize("data,path", [
    ({"text": {"language": "sk"}}, "text.language"),
    ({"align": {"language": "sk"}}, "align.language"),
    ({"align": {"asr": {"language": "sk"}}}, "align.asr.language"),
    ({"audit": {"language": "sk"}}, "audit.language"),
    ({"manifest": {"language": "sk"}}, "manifest.language"),
    ({"manifest": {"speaker": "x"}}, "manifest.speaker"),
    ({"manifest": {"use_part_fragments": True}}, "manifest.use_part_fragments"),
    ({"text": {"use_part_fragments": True}}, "text.use_part_fragments"),
    ({"text": {"work_dir": "w"}}, "text.work_dir"),
    ({"source": {"epub": {"path": "b.epub"}, "work_dir": "w"}}, "source.work_dir"),
])
def test_fed_keys_in_a_section_raise(tmp_path, data, path):
    with pytest.raises(ConfigError) as e:
        from_mapping(data, str(tmp_path))
    assert str(e.value) == f"{path}: set it once at the top level"


def test_fed_keys_reach_stages(tmp_path):
    cfg = from_mapping({"language": "sk", "speaker": "bob", "use_part_fragments": True}, str(tmp_path))
    assert cfg.text.use_part_fragments is True and cfg.manifest.use_part_fragments is True
    assert cfg.text.language == cfg.align.language == cfg.align.asr.language == cfg.audit.language == cfg.manifest.language == "sk"
    assert cfg.manifest.speaker == "bob"
    assert cfg.language == "sk" and cfg.speaker == "bob" and cfg.use_part_fragments is True
    d = from_mapping({}, str(tmp_path))
    assert d.text.use_part_fragments is False and d.manifest.use_part_fragments is False
    assert d.text.language == d.align.asr.language == d.manifest.language == "cs"
    assert d.manifest.speaker == "narrator"


def test_paths_resolve_against_yaml_dir(tmp_path, monkeypatch):
    yml_dir = tmp_path / "conf"
    yml_dir.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    p = write(yml_dir, {
        "raw_dir": "rawdir", "work_dir": "../w",
        "source": {"epub": {"path": "b.epub"}, "orig_dir": "/abs/orig"},
        "lexicon": {"rewrite_dir": "~/rew"},
        "text": {"compare_dir": "cmp"},
        "gold": {"file": "gold.json", "audacity": {"a1": "labels/a1.txt", "a2": "/abs/a2.txt"}},
    })
    cfg = load_config(p)
    assert cfg.raw_dir == str(yml_dir / "rawdir")
    assert cfg.work_dir == str(tmp_path / "w")
    assert cfg.source.epub.path == str(yml_dir / "b.epub")
    assert cfg.source.orig_dir == "/abs/orig"
    assert cfg.lexicon.rewrite_dir == str(tmp_path / "home" / "rew")
    assert cfg.text.compare_dir == str(yml_dir / "cmp")
    assert cfg.gold.file == str(yml_dir / "gold.json")
    assert cfg.gold.audacity == {"a1": str(yml_dir / "labels" / "a1.txt"), "a2": "/abs/a2.txt"}
    assert cfg.source.work_dir == cfg.text.work_dir == cfg.work_dir


def test_derived_defaults(tmp_path):
    base = str(tmp_path)
    plain = from_mapping({"parts": ["a", "b"]}, base)
    assert plain.text.parts == ["a", "b"]
    assert plain.text.text_dir == os.path.join(base, "raw")
    assert plain.text.suffix == ".txt.orig"
    assert plain.text.placement is None and plain.text.lexicon is None and plain.text.compare_dir is None
    assert plain.source is None and plain.lexicon is None and plain.gold is None
    assert plain.text.work_dir == os.path.join(base, "work")

    src = from_mapping({"source": {"epub": {"path": "b.epub"}}}, base)
    epub = os.path.join(base, "work", "source", "epub")
    assert src.text.text_dir == epub and src.text.suffix == ".txt"
    assert src.text.placement == os.path.join(epub, "placement.json")
    assert src.text.lexicon is None

    lex = from_mapping({"source": {"epub": {"path": "b.epub"}}, "lexicon": {"rewrite_dir": "rw", "rewrite_suffix": ".x"}}, base)
    assert lex.lexicon.path == os.path.join(base, "work", "lexicon.tsv")
    assert lex.lexicon.book_dir == epub
    assert lex.text.lexicon == lex.lexicon.path
    assert lex.text.compare_dir == os.path.join(base, "rw") and lex.text.compare_suffix == ".x"

    only_lex = from_mapping({"lexicon": {"rewrite_dir": "rw"}}, base)
    assert only_lex.text.text_dir == os.path.join(base, "raw") and only_lex.lexicon.book_dir == epub

    win = from_mapping({
        "source": {"epub": {"path": "b.epub"}}, "parts": ["p"],
        "text": {"text_dir": "td", "suffix": ".t", "placement": "pl.json", "parts": ["q"]},
    }, base)
    assert win.text.text_dir == os.path.join(base, "td") and win.text.suffix == ".t"
    assert win.text.placement == os.path.join(base, "pl.json") and win.text.parts == ["q"]


def test_type_errors_name_the_path(tmp_path):
    cases = [
        ({"cut": {"pad_s": "x"}}, "cut.pad_s"),
        ({"cut": {"pad_s": "1e-3"}}, "cut.pad_s"),
        ({"cut": {"min_split_words": True}}, "cut.min_split_words"),
        ({"cut": {"min_split_words": 2.5}}, "cut.min_split_words"),
        ({"audio": {"clip_sr": "no"}}, "audio.clip_sr"),
        ({"text": {"skip_unreadable": 1}}, "text.skip_unreadable"),
        ({"text": {"skip_unreadable": "off"}}, "text.skip_unreadable"),
        ({"manifest": {"exclude_flags": "short_pause"}}, "manifest.exclude_flags"),
        ({"manifest": {"split": {"seed": "1"}}}, "manifest.split.seed"),
        ({"align": {"asr": {"batch_size": "4"}}}, "align.asr.batch_size"),
        ({"align": []}, "align"),
        ({"language": 3}, "language"),
        ({"text": {"headings": {"a": [1, "x"]}}}, "text.headings.a[1]"),
    ]
    for data, path in cases:
        with pytest.raises(ConfigError) as e:
            from_mapping(data, str(tmp_path))
        assert str(e.value).startswith(f"{path}: expected"), str(e.value)
    assert from_mapping({"cut": {"pad_s": 1}}, str(tmp_path)).cut.pad_s == 1.0
    assert from_mapping({"cut": {"pad_s": 0.001}}, str(tmp_path)).cut.pad_s == 0.001
    assert from_mapping({"text": {"skip_unreadable": True}}, str(tmp_path)).text.skip_unreadable is True


def test_lists_tuples_and_nested(tmp_path):
    cfg = from_mapping({
        "manifest": {"exclude_flags": ["a", "b"], "split": {"seed": 7, "heldout_share": 0.1}},
        "audit": {"drop_reasons": ["cer", "x"]},
        "source": {"epub": {"path": "b.epub", "heading_tags": ["h1"], "replacements": {"a": "b"}},
                   "parts": [{"name": "p1", "start": 3}, {"name": "p2", "start": "Kapitola 4"}]},
        "text": {"book_config": {"x": [1, {"y": 2}]}, "headings": {"p1": [1, 2]}, "estimate": {"a": 1}},
    }, str(tmp_path))
    assert cfg.manifest.exclude_flags == ("a", "b")
    assert cfg.audit.drop_reasons == ("cer", "x")
    assert cfg.manifest.split.seed == 7 and cfg.manifest.split.heldout_share == 0.1
    assert cfg.source.epub.heading_tags == ("h1",) and cfg.source.epub.replacements == {"a": "b"}
    assert [(p.name, p.start) for p in cfg.source.parts] == [("p1", 3), ("p2", "Kapitola 4")]
    assert cfg.text.book_config == {"x": [1, {"y": 2}]}
    assert cfg.text.headings == {"p1": [1, 2]} and cfg.text.estimate == {"a": 1.0}
    if "judges" in AuditConfig.__dataclass_fields__:
        j = from_mapping({"audit": {"judges": ["m1", "m2"]}}, str(tmp_path))
        assert j.audit.judges == ("m1", "m2")


def test_part_names(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for n in ("a10", "a2", "a1", "b"):
        (raw / f"{n}.wav").write_bytes(b"")
    (raw / "a3.txt").write_text("x")
    cfg = from_mapping({}, str(tmp_path))
    assert cfg.part_names() == ["a1", "a2", "a10", "b"]
    assert from_mapping({"parts": ["z", "y"]}, str(tmp_path)).part_names() == ["z", "y"]
    empty = tmp_path / "e"
    empty.mkdir()
    with pytest.raises(ConfigError):
        from_mapping({}, str(empty)).part_names()


def test_required_keys(tmp_path):
    with pytest.raises(ConfigError, match=r"^source\.epub"):
        from_mapping({"source": {}}, str(tmp_path))
    with pytest.raises(ConfigError, match=r"^source\.epub\.path"):
        from_mapping({"source": {"epub": {}}}, str(tmp_path))
    with pytest.raises(ConfigError, match=r"^lexicon\.rewrite_dir"):
        from_mapping({"lexicon": {}}, str(tmp_path))
    with pytest.raises(ConfigError, match=r"^source\.parts\[0\]\.start"):
        from_mapping({"source": {"epub": {"path": "b"}, "parts": [{"name": "a"}]}}, str(tmp_path))


def test_validate_and_empty_yaml(tmp_path):
    cfg = load_config(write(tmp_path, ""))
    assert cfg.align.backend == "mms-300m" and cfg.cut.pad_s == 0.15 and cfg.audit.drop_reasons == ("cer",)
    assert cfg.manifest == ManifestConfig(language="cs", speaker="narrator", use_part_fragments=False)
    assert cfg.parts is None and cfg.raw_dir == str(tmp_path / "raw") and cfg.work_dir == str(tmp_path / "work")
    validate(cfg)
    bad = from_mapping({"align": {"backend": "nope"}}, str(tmp_path))
    with pytest.raises(ConfigError, match=r"^align: "):
        validate(bad)
    with pytest.raises(ConfigError, match=r"^align: "):
        load_config(write(tmp_path, {"align": {"longform": "nope"}}, "b.yaml"))


def test_audit_edge_level_follows_the_cut(tmp_path):
    base = str(tmp_path)
    plain = from_mapping({}, base)
    assert plain.audit.edge_db == plain.cut.quiet_db
    assert from_mapping({"cut": {"quiet_db": 18.0}}, base).audit.edge_db == 18.0
    assert from_mapping({"cut": {"quiet_db": 18.0}, "audit": {"edge_db": 12.0}}, base).audit.edge_db == 12.0


def test_language_the_normalizer_cannot_read_fails_at_load(tmp_path):
    with pytest.raises(ConfigError, match=r"^language: "):
        load_config(write(tmp_path, {"language": "en"}))
    assert load_config(write(tmp_path, {"language": "sk"}, "sk.yaml")).language == "sk"


@pytest.mark.parametrize("judges", [[], ["parakeet-tdt-0.6b-v3", "parakeet-tdt-0.6b-v3"], ["parakeet-tdt-0.6b-v3", "whisper"]])
def test_validate_judges(tmp_path, judges):
    with pytest.raises(ConfigError, match=r"^audit\.judges: "):
        load_config(write(tmp_path, {"audit": {"judges": judges}}))
