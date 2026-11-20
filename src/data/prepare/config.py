"""YAML configuration of the whole corpus pipeline: one file, mapped onto the stage dataclasses."""
from __future__ import annotations

import dataclasses
import os
import re
import typing
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import yaml

from src.data.prepare.align.interface import validate_config as validate_align
from src.data.prepare.align.types import AlignConfig
from src.data.prepare.audio import AudioConfig
from src.data.prepare.audit import AuditConfig, validate_judges
from src.data.prepare.cut import CutConfig
from src.data.prepare.gold import GoldConfig
from src.data.prepare.manifest import ManifestConfig
from src.data.prepare.source import SourceConfig
from src.data.prepare.text import TextConfig


class ConfigError(ValueError):
    pass


@dataclass
class LexiconConfig:
    rewrite_dir: str
    path: str
    book_dir: str
    rewrite_suffix: str = ".txt.orig"
    book_suffix: str = ".txt"


@dataclass
class PipelineConfig:
    work_dir: str
    raw_dir: str
    parts: Optional[List[str]]
    language: str
    speaker: str
    use_part_fragments: bool
    source: Optional[SourceConfig]
    lexicon: Optional[LexiconConfig]
    text: TextConfig
    audio: AudioConfig
    align: AlignConfig
    cut: CutConfig
    audit: AuditConfig
    manifest: ManifestConfig
    gold: Optional[GoldConfig] = None

    def part_names(self) -> List[str]:
        if self.parts is not None:
            return list(self.parts)
        try:
            names = [n[:-4] for n in os.listdir(self.raw_dir) if n.endswith(".wav")]
        except OSError:
            names = []
        if not names:
            raise ConfigError(f"parts: not set and no *.wav files found in {self.raw_dir}")
        return sorted(names, key=_natural)


_TOP = {"language": str, "speaker": str, "use_part_fragments": bool, "raw_dir": str, "work_dir": str,
        "parts": Optional[List[str]]}
_SECTIONS = {"source": SourceConfig, "lexicon": LexiconConfig, "text": TextConfig, "audio": AudioConfig,
             "align": AlignConfig, "cut": CutConfig, "audit": AuditConfig, "manifest": ManifestConfig,
             "gold": GoldConfig}
_OPTIONAL = ("source", "lexicon", "gold")
_PATHS = {"raw_dir", "work_dir", "source.orig_dir", "source.epub.path", "lexicon.path", "lexicon.rewrite_dir",
          "lexicon.book_dir", "text.text_dir", "text.placement", "text.lexicon", "text.compare_dir", "gold.file"}
_FED_SECTIONS = {"language": ("text", "align", "align.asr", "audit", "manifest"), "speaker": ("manifest",),
                 "use_part_fragments": ("text", "manifest")}


def _natural(name: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def _resolve(value: str, base: str) -> str:
    return os.path.abspath(os.path.join(base, os.path.expanduser(value)))


def _describe(hint) -> str:
    return getattr(hint, "__name__", None) or str(hint).replace("typing.", "")


def _fail(path: str, hint, value) -> ConfigError:
    return ConfigError(f"{path}: expected {_describe(hint)}, got {type(value).__name__} {value!r}")


class _Ctx:
    def __init__(self, base: str, fixed: Dict[str, Any], derived: Dict[str, Any]):
        self.base, self.fixed, self.derived = base, fixed, derived


def _convert(value, hint, path: str, ctx: _Ctx):
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if origin is Union:
        rest = [a for a in args if a is not type(None)]
        if value is None and len(rest) < len(args):
            return None
        if len(rest) == 1:
            return _convert(value, rest[0], path, ctx)
        for a in rest:
            try:
                return _convert(value, a, path, ctx)
            except ConfigError:
                pass
        raise _fail(path, hint, value)
    if dataclasses.is_dataclass(hint):
        if not isinstance(value, dict):
            raise _fail(path, "a mapping", value)
        return _build(hint, value, path, ctx)
    if hint is bool:
        if not isinstance(value, bool):
            raise _fail(path, hint, value)
        return value
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise _fail(path, hint, value)
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _fail(path, "int or float", value)
        return float(value)
    if hint is str:
        if not isinstance(value, str):
            raise _fail(path, hint, value)
        return _resolve(value, ctx.base) if path in _PATHS else value
    if hint is dict:
        if not isinstance(value, dict):
            raise _fail(path, hint, value)
        return value
    if hint is tuple or origin is tuple:
        if not isinstance(value, list):
            raise _fail(path, "a list", value)
        item = args[0] if args else Any
        return tuple(v if item is Any else _convert(v, item, f"{path}[{i}]", ctx) for i, v in enumerate(value))
    if origin is list:
        if not isinstance(value, list):
            raise _fail(path, "a list", value)
        return [_convert(v, args[0], f"{path}[{i}]", ctx) for i, v in enumerate(value)]
    if origin is dict:
        if not isinstance(value, dict):
            raise _fail(path, "a mapping", value)
        out = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise _fail(f"{path}.{k}", "a string key", k)
            sub = f"{path}.{k}"
            if path == "gold.audacity":
                out[k] = _resolve(v, ctx.base) if isinstance(v, str) else _convert(v, args[1], sub, ctx)
            else:
                out[k] = _convert(v, args[1], sub, ctx)
        return out
    raise _fail(path, hint, value)


def _build(cls, mapping: dict, path: str, ctx: _Ctx):
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    valid = sorted(n for n in fields if f"{path}.{n}" not in ctx.fixed)
    for key in mapping:
        if f"{path}.{key}" in ctx.fixed:
            raise ConfigError(f"{path}.{key}: set it once at the top level")
        if key not in fields:
            raise ConfigError(f"{path}.{key}: unknown key; valid keys: {', '.join(valid)}")
    kwargs = {}
    for name, f in fields.items():
        p = f"{path}.{name}"
        if p in ctx.fixed:
            kwargs[name] = ctx.fixed[p]
        elif mapping.get(name) is not None:
            kwargs[name] = _convert(mapping[name], hints[name], p, ctx)
        elif ctx.derived.get(p) is not None:
            kwargs[name] = ctx.derived[p]
        elif dataclasses.is_dataclass(hints[name]) and f.default_factory is not dataclasses.MISSING:
            kwargs[name] = _build(hints[name], {}, p, ctx)
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            raise ConfigError(f"{p}: required")
    return cls(**kwargs)


def from_mapping(mapping: Optional[dict], base_dir: str) -> PipelineConfig:
    mapping = {} if mapping is None else mapping
    if not isinstance(mapping, dict):
        raise _fail("<top level>", "a mapping", mapping)
    for key in mapping:
        if key not in _TOP and key not in _SECTIONS:
            raise ConfigError(f"{key}: unknown key; valid keys: {', '.join(sorted(set(_TOP) | set(_SECTIONS)))}")
    ctx = _Ctx(base_dir, {}, {})
    top = {}
    for key, hint in _TOP.items():
        value = mapping.get(key)
        default = {"language": "cs", "speaker": "narrator", "use_part_fragments": False, "raw_dir": "raw",
                   "work_dir": "work", "parts": None}[key]
        top[key] = default if value is None else _convert(value, hint, key, ctx)
    top["raw_dir"] = _resolve(top["raw_dir"], base_dir)
    top["work_dir"] = _resolve(top["work_dir"], base_dir)
    work, epub = top["work_dir"], os.path.join(top["work_dir"], "source", "epub")

    fixed = {"source.work_dir": work, "text.work_dir": work}
    for key, sections in _FED_SECTIONS.items():
        for s in sections:
            fixed[f"{s}.{key}"] = top[key]
    present = {s: mapping.get(s) for s in _SECTIONS}
    for s, v in present.items():
        if v is not None and not isinstance(v, dict):
            raise _fail(s, "a mapping", v)
    derived = {"text.parts": top["parts"], "text.text_dir": top["raw_dir"]}
    if present["source"] is not None:
        derived.update({"text.text_dir": epub, "text.suffix": ".txt", "text.placement": os.path.join(epub, "placement.json")})
    lex = present["lexicon"]
    if lex is not None:
        lex_path = os.path.join(work, "lexicon.tsv")
        derived.update({"lexicon.path": lex_path, "lexicon.book_dir": epub, "text.lexicon": lex_path})
    ctx = _Ctx(base_dir, fixed, derived)
    if lex is not None:
        built_lex = _build(LexiconConfig, lex, "lexicon", ctx)
        derived.update({"text.compare_dir": built_lex.rewrite_dir, "text.compare_suffix": built_lex.rewrite_suffix})
    else:
        built_lex = None
    sections = {}
    for name, cls in _SECTIONS.items():
        if name == "lexicon":
            sections[name] = built_lex
        elif present[name] is None and name in _OPTIONAL:
            sections[name] = None
        else:
            sections[name] = _build(cls, present[name] or {}, name, ctx)
        if name == "cut":
            derived["audit.edge_db"] = sections["cut"].quiet_db
    return PipelineConfig(**top, **sections)


def validate(cfg: PipelineConfig) -> None:
    try:
        validate_align(cfg.align)
    except ValueError as e:
        raise ConfigError(f"align: {e}") from e
    try:
        validate_judges(cfg.audit)
    except ValueError as e:
        raise ConfigError(f"audit.judges: {e}") from e


def load_config(path) -> PipelineConfig:
    path = os.path.abspath(os.path.expanduser(str(path)))
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    cfg = from_mapping(data, os.path.dirname(path))
    validate(cfg)
    return cfg
