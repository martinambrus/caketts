"""
Text stage of the corpus preparation (plan Step 4.2): the book text of each part -> normalised text -> sentences ->
training units, with a review list of every logged reading and a report.

The input is a directory of per-part book text files (blank-line paragraphs, optional "# " heading lines). Per part the
stage reads the bytes, converts the line endings, checks the format (reported, never repaired), applies the headings of
the config and the chapter starts of a placement.json, normalises the whole part in one call, segments it and builds the
units. The stage writes <work_dir>/text/<part>.norm.txt, <part>.units.jsonl, review_readings.csv, report.json and
report.md; the book text appears only in the first three and in the review list.

With a lexicon (TextConfig.lexicon, see lexicon.py) the applied book forms are replaced before the normalisation. With
compare_dir (the rewrite of the book that matches the narration, one <part><compare_suffix> file per part) the stage also
compares the normalised words with the rewrite's and writes review_disagreements.csv.

Usage:
    run_text_stage(TextConfig(text_dir="raw", work_dir="work/orig", placement="work/source/orig/placement.json"))
"""
from __future__ import annotations

import ast
import csv
import difflib
import functools
import io
import json
import logging
import math
import os
import re
import resource
import time
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from typing import Dict, List, Optional, Tuple

from src.data.prepare.lexicon import apply_lexicon, compare, read_lexicon
from src.text.normalizer import TextNormalizer
from src.text.phonemizer import _SPAN_RE, _WORD_RE
from src.text.segmenter import segment
from src.text.units import Unit, build_units, estimate_s, quote_spans

NORMALIZER_LOGGER = "src.text.normalizer"
_LITERAL_AT_START = re.compile(r"""'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*\"""")  # volatile: the message opens with repr(char)
_LINE_END = re.compile("\r\n|[\r\n  \v\f\x1c\x1d\x1e\x85]")
_KIND_NAMES = {"\r\n": "CRLF", "\n": "LF", "\r": "CR"}
_QUOTES = "\"'„“”‚‘’»«‹›"
_SENTENCE_MARKS = ".!?…\"»"
_UNFINISHED = ".!?…\"“”»"
_CLOSING = "\"'“”‘’»«‹›)] "
_LANG_TAG = re.compile(r"</?(?:cs|sk|en)>")
_LIST_START = re.compile(r"(?:Za|Po)\s+[^\W\d_]*[éí],")
_LIST_INSIDE = re.compile(r"[,;:]\s+(?:za|po)\s+[^\W\d_]*[éí],")
_DATE = re.compile(r"\d{1,2}\.\d{1,2}\.(?:\d{4})?(?!\d)")
_TIME = re.compile(r"\d{1,2}:\d{2}(?!\d)")
_ROMAN = re.compile(r"[IVXLCDM]{2,}\.?|[IVXLCDM]\.")
_ABBREVIATION = re.compile(r"[^\W\d_]{1,6}\.")
_EDGE = "\"'„“”‚‘’»«‹›()[],;:!?…"
_ABBREVIATION_TOKEN = re.compile(r"(?<![^\W\d_])([^\W\d_]{1,6})\.\s+(?=[^\W\d_])")
_ESTIMATE_KEYS = {"a", "b", "c", "e"}
_PAUSE = {"mid": "sentence", "para_end": "paragraph", "chapter_end": "chapter"}
_DIFF_TAGS = ("number", "date", "time", "roman", "abbreviation")
_READING_COLUMNS = ["part", "paragraph", "line", "reason", "source", "reading", "context", "matches"]
_DISAGREEMENT_COLUMNS = ["part", "paragraph", "category", "book_form", "normalizer_reading", "rewrite_form"]


@dataclass
class TextConfig:
    text_dir: str
    work_dir: str
    suffix: str = ".txt.orig"
    parts: Optional[List[str]] = None
    language: str = "cs"
    book_config: dict = field(default_factory=dict)
    headings: Dict[str, List[int]] = field(default_factory=dict)
    placement: Optional[str] = None
    max_est_s: float = 17.0
    estimate: Dict[str, float] = field(default_factory=dict)
    heading_short_chars: int = 40
    use_part_fragments: bool = False
    lexicon: Optional[str] = None
    compare_dir: Optional[str] = None
    compare_suffix: str = ".txt.orig"


class TextStageError(Exception):
    def __init__(self, message: str, failures: Optional[List[dict]] = None):
        super().__init__(message)
        self.failures = failures or []


# ---- reading and format ------------------------------------------------------------------------------------

def _natural(name: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def _part_names(cfg: TextConfig) -> List[str]:
    if cfg.parts is not None:
        return list(cfg.parts)
    names = [f[:-len(cfg.suffix)] for f in os.listdir(cfg.text_dir) if f.endswith(cfg.suffix)]
    if not names:
        raise TextStageError(f"no <part>{cfg.suffix} file in {cfg.text_dir}")
    return sorted(names, key=_natural)


def _blank(line: str) -> bool:
    return not line.strip()


def _blocks(lines: List[str]) -> List[Tuple[bool, int, int]]:
    """The blocks of lines as (is_heading, first, last) 0-based: runs of non-blank lines and "# " heading lines."""
    blocks, first = [], None
    for i, line in enumerate([*lines, ""]):
        if line.startswith("# ") or _blank(line):
            if first is not None:
                blocks.append((False, first, i - 1))
                first = None
            if line.startswith("# "):
                blocks.append((True, i, i))
        elif first is None:
            first = i
    return blocks


def _shape(line: str) -> str:
    return "".join("A" if c.isupper() else "a" if c.islower() else "9" if c.isdigit() else c for c in line[:40])


def _convert_endings(text: str) -> Tuple[str, dict]:
    kinds, lines_of = [], []
    for number, m in enumerate(_LINE_END.finditer(text), 1):
        kinds.append(_KIND_NAMES.get(m.group()) or f"U+{ord(m.group()):04X}")
        lines_of.append(number)
    counts = Counter(kinds)
    major = min(counts, key=lambda k: (-counts[k], k)) if counts else None
    minority: Dict[str, List[int]] = {}
    for kind, number in zip(kinds, lines_of):
        if kind != major:
            minority.setdefault(kind, []).append(number)
    return _LINE_END.sub("\n", text), {"counts": dict(counts), "minority_lines": minority}


def _format_check(text: str, endings: dict, bom: bool, nfc_before: List[int], cfg: TextConfig) -> dict:
    lines = text.splitlines()
    numbered = list(enumerate(lines, 1))
    runs, start = [], None
    for n, line in [*numbered, (len(lines) + 1, "x")]:
        if _blank(line) and start is None:
            start = n
        elif not _blank(line) and start is not None:
            if n - start > 1:
                runs.append({"line": start, "length": n - start})
            start = None
    pairs = []
    for n, line in numbered[:-1]:
        nxt = lines[n]
        if _blank(line) or _blank(nxt) or line.startswith("# ") or nxt.startswith("# "):
            continue
        head = nxt.lstrip()[0]
        pairs.append({"line": n, "ends_sentence": line.rstrip()[-1] in _SENTENCE_MARKS,
                      "next_starts_upper_or_quote": head.isupper() or head in _QUOTES})
    candidates = []
    first_text = next((n for n, line in numbered if not _blank(line)), None)
    for n, line in numbered:
        stripped = line.strip()
        reasons = []
        if n == first_text:
            reasons.append("first_line")
        if stripped and len(stripped) < cfg.heading_short_chars and stripped[-1] not in _UNFINISHED:
            reasons.append("short_unterminated")
        letters = [c for c in stripped if c.isalpha()]
        if len(letters) >= 4 and not any(c.islower() for c in letters):
            reasons.append("all_caps")
        if reasons:
            candidates.append({"line": n, "reasons": reasons, "shape": _shape(stripped)})
    return {
        "bom": bom,
        "line_count": len(lines),
        "line_endings": endings,
        "missing_final_newline": bool(text) and not text.endswith("\n"),
        "non_nfc_lines": nfc_before,
        "tab_lines": [n for n, line in numbered if "\t" in line],
        "nbsp_lines": [n for n, line in numbered if " " in line],
        "edge_space_lines": [n for n, line in numbered if line.strip() and (line[0] == " " or line[-1] == " ")],
        "whitespace_only_lines": [n for n, line in numbered if line and _blank(line)],
        "blank_runs": runs,
        "heading_lines": [n for n, line in numbered if line.startswith("# ")],
        "consecutive_pairs": pairs,
        "heading_candidates": candidates,
    }


def _apply_headings(lines: List[str], wanted: List[int], starts: List[int]) -> Tuple[dict, dict]:
    lines_in = list(lines)
    config = {"applied": [], "not_applied": []}
    for n in sorted(set(wanted)):
        if not 1 <= n <= len(lines_in):
            config["not_applied"].append({"line": n, "reason": "line_out_of_range"})
        elif _blank(lines_in[n - 1]):
            config["not_applied"].append({"line": n, "reason": "line_blank"})
        elif lines_in[n - 1].startswith("# "):
            config["not_applied"].append({"line": n, "reason": "already_a_heading"})
        else:
            lines[n - 1] = "# " + lines_in[n - 1]
            config["applied"].append(n)
    chapters = {"applied": 0, "not_applied": []}
    for n in sorted(set(starts)):
        if not 1 <= n <= len(lines_in):
            reason = "line_out_of_range"
        elif n == 1:
            reason = "first_line"
        elif not _blank(lines_in[n - 2]):
            reason = "previous_line_not_blank"
        else:
            lines[n - 2] = "# "
            chapters["applied"] += 1
            continue
        chapters["not_applied"].append({"line": n, "reason": reason})
    return config, chapters


# ---- normalisation -----------------------------------------------------------------------------------------

class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.records: List[Tuple[str, str, str, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        args = record.args
        if not (isinstance(args, tuple) and len(args) == 4 and all(isinstance(a, str) for a in args)):
            raise TextStageError(f"unexpected log record of {NORMALIZER_LOGGER}: args {args!r}, "
                                 f"wanted (reason, source, reading, context) as four strings")
        self.records.append(args)


def _normalise_logged(normalizer, text: str) -> Tuple[str, List[Tuple[str, str, str, str]]]:
    logger = logging.getLogger(NORMALIZER_LOGGER)
    handler, level = _Capture(), logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        return normalizer.normalize(text), handler.records
    finally:
        logger.setLevel(level)
        logger.removeHandler(handler)


def _failing_character(message: str) -> Optional[str]:
    m = _LITERAL_AT_START.search(message)
    if not m:
        return None
    try:
        value = ast.literal_eval(m.group())
    except (ValueError, SyntaxError):
        return None
    return value if isinstance(value, str) and len(value) == 1 else None


def _failures_by_block(normalizer, part: str, lines: List[str]) -> List[dict]:
    failures = []
    for number, (_, first, last) in enumerate(_blocks(lines), 1):
        block, found = "\n".join(lines[first:last + 1]), []
        while True:
            try:
                _normalise_logged(normalizer, block)
                break
            except ValueError as e:
                ch = _failing_character(str(e))
                if ch is None or ch not in block:
                    found.append("unparsed" if ch is None else f"U+{ord(ch):04X}")
                    break
                found.append(f"U+{ord(ch):04X}")
                block = block.replace(ch, "")
        if found:
            failures.append({"part": part, "line": first + 1, "block": number, "codepoints": found})
    return failures


# ---- units -------------------------------------------------------------------------------------------------

def _words(text: str) -> Tuple[List[str], List[List[int]]]:
    words, spans, pos = [], [], 0

    def scan(start: int, end: int) -> None:
        for m in _WORD_RE.finditer(text, start, end):
            words.append(m.group())
            spans.append([m.start(), m.end()])

    for s in _SPAN_RE.finditer(text):
        scan(pos, s.start())
        scan(s.start(2), s.end(2))
        pos = s.end()
    scan(pos, len(text))
    return words, spans


def _has_letter(text: str) -> bool:
    return any(c.isalpha() for c in _LANG_TAG.sub("", text))


def _end_mark(text: str) -> str:
    tail = text.rstrip(_CLOSING)
    if tail.endswith("...") or tail.endswith("…"):
        return "ellipsis"
    return {".": "statement", "?": "question", "!": "exclamation"}.get(tail[-1:], "unterminated")


def _quote_tags(unit: Unit) -> set:
    text, tags = unit.text, set()
    spans = quote_spans(text, unit.open_before)
    quotes = [s for s in spans if text[s[0]] not in "(["]
    if len(spans) > len(quotes):
        tags.add("parenthesis")
    if any(s == (0, len(text)) and text[s[0]] in "([" for s in spans):
        tags.add("parenthesis_whole")
    if quotes:
        before, after = _has_letter(text[:quotes[0][0]]), _has_letter(text[quotes[-1][1]:])
        between = any(_has_letter(text[a[1]:b[0]]) for a, b in zip(quotes, quotes[1:]))
        if len(quotes) == 1:
            tags.add({(False, False): "quote_only", (False, True): "quote_then_narration",
                      (True, False): "narration_then_quote", (True, True): "narration_quote_narration"}[(before, after)])
        else:
            tags.add("multi_quote")
            if between and not before and not after:
                tags.add("quote_narration_quote")
        if unit.kind == "joined":
            tags.add("multi_sentence_quote")
    return tags


def _unit_tags(unit: Unit, words: List[str]) -> set:
    text = unit.text
    tags = {_end_mark(text), unit.boundary} | _quote_tags(unit)
    for mark, name in (("?", "has_question"), ("!", "has_exclamation")):
        if mark in text:
            tags.add(name)
    if unit.open_after > 0:
        tags.add("open_cut" if unit.boundary == "mid" else "open_para_end")
    if text[:2] in ("— ", "– ", "- "):
        tags.add("dash_dialogue")
    if _LIST_START.match(text) or _LIST_INSIDE.search(text):
        tags.add("list_item")
    tags |= {f"{m.group(1)}_span" for m in _SPAN_RE.finditer(text)}
    for flag, name in ((unit.heading, "heading"), (len(words) <= 2, "very_short"), (unit.kind == "joined", "joined"),
                       (unit.kind == "split", "clause_cut"), (unit.over_cap, "over_cap")):
        if flag:
            tags.add(name)
    return tags


def _classify(token: str, abbreviations) -> Optional[str]:
    core = token.strip(_EDGE)
    if not core or not any(c.isalnum() for c in token):
        return None
    if any(c.isdigit() for c in core):
        if _DATE.fullmatch(core):
            return "date"
        return "time" if _TIME.fullmatch(core) else "number"
    if _ROMAN.fullmatch(core):
        return "roman"
    return "abbreviation" if (_ABBREVIATION.fullmatch(core) and core.lower() in abbreviations) else None


def _diff_tags(raw_lines: List[str], norm_lines: List[str], units: List[Unit], sentences, blocks, abbreviations) -> Tuple[List[set], int]:
    extra, unmapped = [set() for _ in units], 0
    by_line: Dict[int, List[int]] = {}
    for k, u in enumerate(units):
        by_line.setdefault(sentences[u.first].line, []).append(k)
    for heading, first, last in blocks:
        cut = 2 if heading else 0
        raw = " ".join(raw_lines[first:last + 1])[cut:].split()
        norm = " ".join(norm_lines[first:last + 1])[cut:].split()
        members = by_line.get(first, [])
        counts = [len(units[k].text.split()) for k in members]
        if sum(counts) != len(norm):
            unmapped += 1
            continue
        if not members:
            continue
        starts = [sum(counts[:i]) for i in range(len(counts))]
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, raw, norm, autojunk=False).get_opcodes():
            if op == "equal":
                continue
            names = {c for c in (_classify(t, abbreviations) for t in raw[i1:i2]) if c}
            if not names:
                continue
            reach = range(j1, max(j2, j1 + 1))
            for i, start in enumerate(starts):
                stop = start + counts[i]
                if any(start <= j < stop or (j >= len(norm) and i == len(starts) - 1) for j in reach):
                    extra[members[i]] |= names
    return extra, unmapped


def _abbreviation_matches(text: str, known):
    text = _SPAN_RE.sub("\x00", text)
    return [m for m in _ABBREVIATION_TOKEN.finditer(text)
            if text[m.end()].islower() and m.group(1).lower() + "." in known]


def _rank(values: List[float], q: float) -> float:
    return values[math.ceil(q * len(values)) - 1] if values else 0.0


def _end_unit(unit: Unit, end: dict) -> Unit:
    if end["inside_paragraph"]:
        boundary = "mid"
    else:
        boundary = "chapter_end" if end["next"] in ("heading", "none") else "para_end"
    return replace(unit, boundary=boundary, pause=_PAUSE[boundary])


def _sentence_entries(unit: Unit, sentences) -> List[dict]:
    if unit.kind == "split":
        return [{"text": unit.text, "boundary": unit.boundary}]
    entries = [{"text": s.text, "boundary": s.boundary} for s in sentences[unit.first:unit.last + 1]]
    entries[-1]["boundary"] = unit.boundary
    return entries


def _locate(records, blocks, lines: List[str]) -> List[list]:
    texts = [" ".join(lines[first:last + 1]) for _, first, last in blocks]
    rows = []
    for reason, source, reading, context in records:
        hits = [i for i, t in enumerate(texts) if context in t]
        where = ["", ""] if not hits else [hits[0] + 1, blocks[hits[0]][1] + 1]
        rows.append([where[0], where[1], reason, source, reading, context, len(hits)])
    return rows


def _paragraph_of(blocks, line: int):
    return next((i for i, (_, first, last) in enumerate(blocks, 1) if first <= line - 1 <= last), "")


def _compare_part(cfg, normalizer, name: str, raw: str, given: str, normalised: str, blocks) -> dict:
    path = os.path.join(cfg.compare_dir, name + cfg.compare_suffix)
    if not os.path.isfile(path):
        raise TextStageError(f"no rewrite file {path} to compare part {name} with")
    with open(path, "rb") as f:
        rewrite = f.read().decode("utf-8-sig")
    result = compare(raw, given, normalised, _convert_endings(rewrite)[0], normalizer.abbreviations)
    rows = [[name, _paragraph_of(blocks, line), *rest] for line, *rest in result.pop("rows")]
    return {**result, "disagreements": dict(sorted(Counter(r[2] for r in rows).items())), "csv": rows}


def _process_part(cfg, normalizer, placement, name: str, estimate, entries=None) -> dict:
    path = os.path.join(cfg.text_dir, name + cfg.suffix)
    with open(path, "rb") as f:
        data = f.read()
    try:
        decoded = data.decode("utf-8")
    except UnicodeDecodeError as e:
        return {"failed": [{"part": name, "error": "utf-8", "byte_offset": e.start}]}
    bom = decoded.startswith("﻿")
    converted, endings = _convert_endings(decoded[1:] if bom else decoded)
    nfc_before = [n for n, line in enumerate(converted.splitlines(), 1) if line != unicodedata.normalize("NFC", line)]
    fmt = _format_check(converted, endings, bom, nfc_before, cfg)
    text = unicodedata.normalize("NFC", converted)
    raw, replaced = text, 0
    if entries is not None:
        text, replaced = apply_lexicon(text, entries)
    plan = (placement.get("parts") or {}).get(name)
    lines = text.splitlines()
    config_headings, chapter_starts = _apply_headings(lines, cfg.headings.get(name, []),
                                                      plan["chapter_starts"] if plan else [])
    given = "\n".join(lines) + ("\n" if text.endswith("\n") else "")
    given_lines = given.splitlines()
    blocks = _blocks(given_lines)
    started = time.perf_counter()
    try:
        normalised, records = _normalise_logged(normalizer, given)
    except ValueError:
        failures = _failures_by_block(normalizer, name, given_lines)
        if not failures:
            failures = [{"part": name, "line": 1, "block": 0, "codepoints": []}]
        return {"failed": failures, "format": fmt}
    seconds = time.perf_counter() - started
    norm_lines = normalised.splitlines()
    if len(norm_lines) != len(given_lines):
        raise TextStageError(f"{name}: the normaliser changed the line count from {len(given_lines)} to {len(norm_lines)}")
    sentences = segment(normalised)
    units = build_units(sentences, cfg.max_est_s, estimate)
    start, finish = (plan["start"], plan["end"]) if plan else ({"mid_sentence": False}, {
        "inside_paragraph": False, "mid_sentence": False, "next": "paragraph"})
    if units:
        units[-1] = _end_unit(units[-1], finish)
    fragments = {0 if start["mid_sentence"] else None, len(units) - 1 if finish["mid_sentence"] else None} - {None}
    extra, unmapped = _diff_tags(given_lines, norm_lines, units, sentences, _blocks(norm_lines), normalizer.abbreviations)
    block_end = {first: last for _, first, last in _blocks(norm_lines)}
    rows = []
    for k, u in enumerate(units):
        words, spans = _words(u.text)
        tags = _unit_tags(u, words) | extra[k]
        if k in fragments:
            tags.add("part_fragment")
        first_line = sentences[u.first].line
        rows.append({
            "id": f"{name}-{k:05d}", "part": name, "index": k, "text": u.text, "sentences": _sentence_entries(u, sentences),
            "boundary": u.boundary, "chapter": u.chapter, "paragraph": u.paragraph, "heading": u.heading,
            "est_s": round(estimate(u.text), 3), "words": words, "word_spans": spans, "tags": tags,
            "source": {"line_start": first_line + 1, "line_end": block_end[first_line] + 1}})
    abbreviations = Counter(m.group(1).lower() for m in _abbreviation_matches(normalised, normalizer.abbreviations))
    comparison = _compare_part(cfg, normalizer, name, raw, given, normalised, blocks) if cfg.compare_dir else None
    stats_sentences = [s for s in sentences if not s.heading]
    return {
        "norm": normalised, "rows": rows, "units": units, "format": fmt, "readings": _locate(records, blocks, given_lines),
        "readings_by_reason": dict(sorted(Counter(r[0] for r in records).items())),
        "abbreviation_tokens": dict(sorted(abbreviations.items())), "diff_unmapped_blocks": unmapped,
        "paragraphs": len({(s.chapter, s.paragraph) for s in stats_sentences}),
        "headings": len({(s.chapter, s.paragraph) for s in sentences if s.heading}), "sentences": len(sentences),
        "clause_cuts": sum(u.pause == "clause" for u in units),
        "chapter_starts": chapter_starts, "config_headings": config_headings, "normalise_s": seconds,
        "lexicon_replacements": replaced if entries is not None else None, "comparison": comparison}


# ---- report ------------------------------------------------------------------------------------------------

def _rss() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def _stats(rows: List[dict], extras: dict) -> dict:
    est = sorted(r["est_s"] for r in rows)
    tags = Counter(t for r in rows for t in r["tags"])
    return {
        "paragraphs": extras["paragraphs"], "headings": extras["headings"], "sentences": extras["sentences"],
        "units": len(rows), "joined_units": sum("joined" in r["tags"] for r in rows),
        "clause_cuts": extras["clause_cuts"],
        "est_s": {"p50": _rank(est, 0.5), "p95": _rank(est, 0.95), "max": est[-1] if est else 0.0},
        "units_over_cap": tags["over_cap"], "words": sum(len(r["words"]) for r in rows), "fragments": tags["part_fragment"],
        "tags": dict(sorted(tags.items())), "readings_by_reason": extras["readings_by_reason"],
        "abbreviation_tokens": extras["abbreviation_tokens"], "diff_unmapped_blocks": extras["diff_unmapped_blocks"],
        "units_not_nfc": extras["units_not_nfc"]}


def _format_counts(fmt: dict) -> dict:
    out = {}
    for key, value in fmt.items():
        if key == "line_endings":
            out[key] = sum(len(v) for v in value["minority_lines"].values())
        elif isinstance(value, (list, dict)):
            out[key] = len(value)
        else:
            out[key] = int(value)
    return out


def _comparison_stats(result: Optional[dict], parts: Optional[List[dict]] = None) -> dict:
    if result is not None:
        words, before, after = result["rewrite_words"], result["matched_before"], result["matched_after"]
        disagreements = result["disagreements"]
    else:
        words, before, after = (sum(p[k] for p in parts) for k in ("rewrite_words", "matched_before", "matched_after"))
        disagreements = _sum_counts([p["disagreements"] for p in parts])
    return {"rewrite_words": words, "matched_before": before, "matched_after": after,
            "share_before": round(before / words, 4) if words else 0.0,
            "share_after": round(after / words, 4) if words else 0.0, "disagreements": disagreements}


def _sum_counts(counts: List[dict]) -> dict:
    total: Counter = Counter()
    for c in counts:
        total.update(c)
    return dict(sorted(total.items()))


def _markdown(report: dict) -> str:
    out = [f"# Text stage report ({report['status']})", ""]
    cfg = report["config"]
    out += [f"- text_dir: {cfg['text_dir']}", f"- work_dir: {cfg['work_dir']}", f"- placement: {cfg['placement']}",
            f"- use_part_fragments: {cfg['use_part_fragments']}", ""]
    for failure in report["failures"]:
        out.append(f"- FAILED {json.dumps(failure, ensure_ascii=False)}")
    for name, p in [*report["parts"].items(), ("total", report["total"])]:
        out += [f"## {name}", ""]
        if p.get("status") == "failed":
            out.append("failed")
        else:
            for key in ("paragraphs", "headings", "sentences", "units", "joined_units", "clause_cuts", "units_over_cap",
                        "words", "fragments", "diff_unmapped_blocks", "units_not_nfc"):
                out.append(f"- {key}: {p[key]}")
            for key in ("lexicon_replacements", "comparison"):
                if key in p:
                    out.append(f"- {key}: {p[key]}")
            out += [f"- est_s: {p['est_s']}", f"- tags: {p['tags']}", f"- readings_by_reason: {p['readings_by_reason']}",
                    f"- chapter_starts applied: {p['chapter_starts']['applied']}, "
                    f"not applied: {json.dumps(p['chapter_starts']['not_applied'])}",
                    f"- abbreviation_tokens: {dict(sorted(p['abbreviation_tokens'].items(), key=lambda kv: -kv[1])[:20])}",
                    f"- timing: {p['timing']}"]
        fmt = p.get("format")
        if fmt is None:
            out.append("")
            continue
        out.append("- format:")
        if name == "total":
            out.append(f"  - counts: {fmt}")
        else:
            for key, value in fmt.items():
                if key == "line_endings":
                    shown = {k: v[:10] for k, v in value["minority_lines"].items()}
                    out.append(f"  - line_endings: {value['counts']}, minority lines (first 10): {shown}")
                elif isinstance(value, list):
                    first = [json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v for v in value[:10]]
                    out.append(f"  - {key}: {len(value)}, first 10: {first}")
                else:
                    out.append(f"  - {key}: {value}")
        out.append("")
    return "\n".join(out) + "\n"


def _write(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)


def run_text_stage(cfg: TextConfig) -> dict:
    bad = set(cfg.estimate) - _ESTIMATE_KEYS
    if bad:
        raise ValueError(f"estimate has keys {sorted(bad)}; allowed: {sorted(_ESTIMATE_KEYS)}")
    estimate = functools.partial(estimate_s, **cfg.estimate)
    placement = {}
    if cfg.placement:
        with open(cfg.placement, encoding="utf-8") as f:
            placement = json.load(f)
    names = _part_names(cfg)
    out = os.path.join(cfg.work_dir, "text")
    os.makedirs(out, exist_ok=True)
    normalizer = TextNormalizer(cfg.language, cfg.book_config)
    entries = read_lexicon(cfg.lexicon) if cfg.lexicon else None
    done, failures, failed_reports = {}, [], {}
    for name in names:
        result = _process_part(cfg, normalizer, placement, name, estimate, entries)
        if "failed" in result:
            failures += result["failed"]
            failed_reports[name] = {"status": "failed", **({"format": result["format"]} if "format" in result else {})}
            for suffix in (".units.jsonl", ".norm.txt"):
                if os.path.exists(os.path.join(out, name + suffix)):
                    os.remove(os.path.join(out, name + suffix))
        else:
            done[name] = result
    all_rows = [r for d in done.values() for r in d["rows"]]
    est = sorted(r["est_s"] for r in all_rows)
    p90 = _rank(est, 0.9)
    for r in all_rows:
        if r["est_s"] > p90:
            r["tags"].add("long")
        r["tags"] = sorted(r["tags"])
    parts, csv_rows, disagreement_rows = {}, [], []
    for name in names:
        if name in failed_reports:
            parts[name] = failed_reports[name]
            continue
        d = done[name]
        d["units_not_nfc"] = sum(u.text != unicodedata.normalize("NFC", u.text) for u in d["units"])
        parts[name] = {**_stats(d["rows"], d), "format": d["format"], "chapter_starts": d["chapter_starts"],
                       "config_headings": d["config_headings"],
                       "timing": {"normalise_s": round(d["normalise_s"], 3), "peak_rss_mib": _rss()}}
        if entries is not None:
            parts[name]["lexicon_replacements"] = d["lexicon_replacements"]
        if d["comparison"]:
            parts[name]["comparison"] = _comparison_stats(d["comparison"])
            disagreement_rows += d["comparison"]["csv"]
        csv_rows += [[name, *r] for r in d["readings"]]
        _write(os.path.join(out, name + ".norm.txt"), d["norm"])
        if not failures:
            _write(os.path.join(out, name + ".units.jsonl"),
                   "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in d["rows"]))
    ok = [p for p in parts.values() if p.get("status") != "failed"]
    total = _stats(all_rows, {
        "paragraphs": sum(d["paragraphs"] for d in done.values()), "headings": sum(d["headings"] for d in done.values()),
        "sentences": sum(d["sentences"] for d in done.values()),
        "clause_cuts": sum(d["clause_cuts"] for d in done.values()),
        "readings_by_reason": _sum_counts([d["readings_by_reason"] for d in done.values()]),
        "abbreviation_tokens": _sum_counts([d["abbreviation_tokens"] for d in done.values()]),
        "diff_unmapped_blocks": sum(d["diff_unmapped_blocks"] for d in done.values()),
        "units_not_nfc": sum(p["units_not_nfc"] for p in ok)})
    total["format"] = _sum_counts([_format_counts(p["format"]) for p in parts.values() if "format" in p])
    total["chapter_starts"] = {"applied": sum(p["chapter_starts"]["applied"] for p in ok), "not_applied": [
        {"part": n, **x} for n, p in parts.items() if p.get("status") != "failed" for x in p["chapter_starts"]["not_applied"]]}
    if entries is not None:
        total["lexicon_replacements"] = sum(p["lexicon_replacements"] for p in ok)
    if cfg.compare_dir:
        total["comparison"] = _comparison_stats(None, [p["comparison"] for p in ok])
    total["timing"] = {"normalise_s": round(sum(p["timing"]["normalise_s"] for p in ok), 3), "peak_rss_mib": _rss()}
    report = {"status": "failed" if failures else "ok", "failures": failures, "config": asdict(cfg), "parts": parts,
              "total": total}
    _write(os.path.join(out, "review_readings.csv"), _csv(csv_rows))
    if cfg.compare_dir:
        _write(os.path.join(out, "review_disagreements.csv"), _csv(disagreement_rows, _DISAGREEMENT_COLUMNS))
    _write(os.path.join(out, "report.json"), json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    _write(os.path.join(out, "report.md"), _markdown(report))
    if failures:
        raise TextStageError(f"text stage failed: {len(failures)} failure(s) in {sorted({f['part'] for f in failures})}",
                             failures)
    return report


def _csv(rows: List[list], header: List[str] = _READING_COLUMNS) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buf.getvalue()
