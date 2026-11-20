import bisect
import difflib
import json
import os
import posixpath
import re
import unicodedata
import zipfile
from dataclasses import dataclass, field, replace
from html.parser import HTMLParser
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import unquote
from xml.etree import ElementTree as ET

_WORD = re.compile(r"[^\W\d_]+")
_SENT_END = re.compile("[.!?…][\"'”’»)\\]›]*$")
_HTML_TYPES = ("application/xhtml+xml", "text/html")
_BLOCK_TAGS = ("p", "h1", "h2", "h3", "h4", "h5", "h6")
_MAX_ANCHOR_DISTANCE = 5
_MAX_LINE_RANK = 3


@dataclass
class EpubConfig:
    path: str
    heading_tags: Tuple[str, ...] = ("h1", "h2", "h3", "h4", "h5", "h6")
    chapter_regex: str = r"^(?P<num>\d+)\s*/\s*(?P<title>\S.*)$"
    chapter_format: str = "{num}. {title}"
    keep_tables: bool = False
    replacements: Dict[str, str] = field(default_factory=dict)


@dataclass
class Item:
    index: int
    kind: str
    text: str
    tag: str
    file: str
    table: bool
    excluded: bool


@dataclass
class PartSpec:
    name: str
    start: Union[int, str]


@dataclass
class SourceConfig:
    epub: EpubConfig
    work_dir: str
    orig_dir: Optional[str] = None
    orig_suffix: str = ".txt.orig"
    parts: Optional[List[PartSpec]] = None
    solid: int = 3


@dataclass
class EpubBook:
    items: List[Item]
    report: dict


class _BlockParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.blocks = []
        self.skip = 0
        self.outside = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1
        elif tag == "br":
            if self.stack:
                self.stack[-1][1].append(" ")
        elif tag in _BLOCK_TAGS:
            self.stack.append((tag, []))

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = max(0, self.skip - 1)
        elif self.stack and self.stack[-1][0] == tag:
            name, parts = self.stack.pop()
            self.blocks.append((name, "".join(parts)))

    def handle_data(self, data):
        if self.skip:
            return
        if self.stack:
            self.stack[-1][1].append(data)
        else:
            self.outside += len("".join(data.split()))

    def finish(self):
        self.close()
        while self.stack:
            name, parts = self.stack.pop()
            self.blocks.append((name, "".join(parts)))


def _decode(data):
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace")


def _spine_files(z):
    opf_path = None
    if "META-INF/container.xml" in z.namelist():
        root = ET.fromstring(z.read("META-INF/container.xml"))
        for el in root.iter():
            if el.tag.rsplit("}", 1)[-1] == "rootfile" and el.get("full-path"):
                opf_path = el.get("full-path")
                break
    if opf_path is None:
        opfs = [n for n in z.namelist() if n.lower().endswith(".opf")]
        if len(opfs) != 1:
            raise ValueError("cannot locate the package document: %d .opf files" % len(opfs))
        opf_path = opfs[0]
    opf = ET.fromstring(z.read(opf_path))
    base = posixpath.dirname(opf_path)
    manifest = {}
    spine = []
    for el in opf.iter():
        name = el.tag.rsplit("}", 1)[-1]
        if name == "item":
            manifest[el.get("id")] = (el.get("href") or "", el.get("media-type") or "")
        elif name == "itemref":
            spine.append(el.get("idref"))
    files = []
    for idref in spine:
        if idref not in manifest:
            continue
        href, media = manifest[idref]
        if media in _HTML_TYPES:
            files.append(posixpath.normpath(posixpath.join(base, unquote(href.split("#")[0]))))
    return files


def read_epub(cfg: EpubConfig) -> EpubBook:
    chapter = re.compile(cfg.chapter_regex)
    items: List[Item] = []
    report = {
        "files": 0,
        "missing_files": [],
        "empty_skipped": 0,
        "outside_text_chars": {},
        "chapter_numbers": [],
    }
    with zipfile.ZipFile(cfg.path) as z:
        names = set(z.namelist())
        for fname in _spine_files(z):
            if fname not in names:
                report["missing_files"].append(fname)
                continue
            report["files"] += 1
            parser = _BlockParser()
            parser.feed(_decode(z.read(fname)))
            parser.finish()
            if parser.outside:
                report["outside_text_chars"][fname] = parser.outside
            for tag, raw in parser.blocks:
                text = unicodedata.normalize("NFC", raw)
                if not text.strip():
                    report["empty_skipped"] += 1
                    continue
                table = "\t" in text
                text = re.sub(r"[ \n\r\f\v]+", " ", text).strip(" ")
                kind = "body"
                if tag in cfg.heading_tags:
                    kind = "heading"
                else:
                    m = chapter.match(text) if tag == "p" else None
                    if m:
                        kind = "chapter"
                        groups = m.groupdict()
                        text = cfg.chapter_format.format(**groups)
                        report["chapter_numbers"].append([len(items), groups.get("num")])
                items.append(Item(len(items), kind, text, tag, fname, table, table and not cfg.keep_tables))
    counts = {}
    for it in items:
        if it.excluded:
            continue
        for key, value in cfg.replacements.items():
            n = it.text.count(key) if key else 0
            if n:
                it.text = it.text.replace(key, value)
                entry = counts.setdefault(key, {"count": 0, "items": []})
                entry["count"] += n
                entry["items"].append(it.index)
    report["replacements"] = counts
    report["items"] = {k: sum(1 for i in items if i.kind == k) for k in ("heading", "chapter", "body")}
    report["items"]["total"] = len(items)
    report["table_items"] = [i.index for i in items if i.table]
    report["tables_excluded"] = not cfg.keep_tables
    return EpubBook(items, report)


def to_book_text(items: List[Item]) -> str:
    blocks = [("# " + i.text) if i.kind != "body" else i.text for i in items if not i.excluded]
    return "\n\n".join(blocks) + "\n" if blocks else ""


def _words(text, pattern=_WORD):
    return [(m.group().lower(), m.start(), m.end()) for m in pattern.finditer(text)]


def _natural(name):
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name)]


def _mapper(blocks, src):
    dst = 1 - src
    keys = [b[src] for b in blocks]

    def f(i, prefer):
        j = bisect.bisect_right(keys, i) - 1
        if j >= 0 and i < keys[j] + blocks[j][2]:
            return blocks[j][dst] + i - keys[j], 0
        before = after = None
        if j >= 0:
            last = keys[j] + blocks[j][2] - 1
            before = (blocks[j][dst] + blocks[j][2] - 1 + i - last, i - last)
        if j + 1 < len(blocks):
            after = (blocks[j + 1][dst] - (keys[j + 1] - i), keys[j + 1] - i)
        if prefer == "nearest":
            cands = [c for c in (before, after) if c]
            return min(cands, key=lambda c: c[1]) if cands else None
        order = (after, before) if prefer == "after" else (before, after)
        return next((c for c in order if c), None)

    return f


def _mid_sentence(left, right):
    first = next((c for c in right if c.isalpha()), "")
    return not (_SENT_END.search(left.rstrip()) and first.isupper())


def _trim_left(text, pos):
    while pos > 0 and not text[pos - 1].isspace() and not text[pos - 1].isalnum():
        pos -= 1
    return pos


def _trim_right(text, pos):
    while pos < len(text) and not text[pos].isspace() and not text[pos].isalnum():
        pos += 1
    return pos


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _dump(obj):
    return json.dumps(obj, indent=1, ensure_ascii=True) + "\n"


class _Stream:
    def __init__(self, items):
        self.inc = [i for i in items if not i.excluded]
        self.spans = [_words(i.text) for i in self.inc]
        self.words = []
        self.item_of = []
        self.offset_of = []
        self.first_word = []
        for p, sp in enumerate(self.spans):
            self.first_word.append(len(self.words) if sp else None)
            for o, (w, _, _) in enumerate(sp):
                self.words.append(w)
                self.item_of.append(p)
                self.offset_of.append(o)

    def next_kind(self, p):
        if p + 1 >= len(self.inc):
            return "none"
        return "paragraph" if self.inc[p + 1].kind == "body" else "heading"


def _cut(stream, ps, os_, pe, oe):
    inc = stream.inc
    out = []
    start = end = None
    for p in range(ps, pe + 1):
        it = inc[p]
        sp = stream.spans[p]
        lo, hi = 0, len(it.text)
        if p == ps and os_ > 0:
            lo = _trim_left(it.text, sp[os_][1])
            start = {
                "inside_paragraph": True,
                "mid_sentence": _mid_sentence(it.text[:lo], it.text[lo:]),
            }
        if p == pe and oe < len(sp) - 1:
            hi = _trim_right(it.text, sp[oe][2])
            end = {
                "inside_paragraph": True,
                "mid_sentence": _mid_sentence(it.text[:hi], it.text[hi:]),
            }
        out.append(replace(it, text=it.text[lo:hi]))
    start = start or {"inside_paragraph": False, "mid_sentence": False}
    end = end or {"inside_paragraph": False, "mid_sentence": False}
    end["next"] = stream.next_kind(pe)
    return to_book_text(out), start, end


def _gap_items(items, lo, hi):
    return [
        {"index": i.index, "kind": i.kind, "excluded": i.excluded, "words": len(_words(i.text))}
        for i in items
        if (lo is None or i.index > lo) and (hi is None or i.index < hi)
    ]


def _gap(after, before, words_between, items):
    if not items and not words_between:
        return None
    return {"after": after, "before": before, "words_between": words_between, "items": items}


def _placement(source, names, flags, chapter_starts):
    parts = {}
    for n in names:
        start, end = flags[n]
        parts[n] = {"chapter_starts": chapter_starts.get(n, []), "start": start, "end": end}
    return {"source": source, "parts": parts}


def _read_orig(cfg):
    suffix = cfg.orig_suffix
    stems = [f[: -len(suffix)] for f in os.listdir(cfg.orig_dir) if f.endswith(suffix)]
    stems.sort(key=_natural)
    if not stems:
        raise ValueError("no %s files in %s" % (suffix, cfg.orig_dir))
    words, lines_tbl, line_of, part_first, part_last = [], [], [], [], []
    for k, stem in enumerate(stems):
        with open(os.path.join(cfg.orig_dir, stem + suffix), "rb") as f:
            text = unicodedata.normalize("NFC", f.read().decode("utf-8"))
        first = None
        for ln, line in enumerate(re.split(r"\r\n|\n|\r", text), 1):
            ws = _words(line)
            if not ws:
                continue
            lines_tbl.append((k, ln, len(words), len(ws)))
            for w, _, _ in ws:
                line_of.append(len(lines_tbl) - 1)
                words.append(w)
            if first is None:
                first = len(words) - len(ws)
        if first is None:
            raise ValueError("part %s has no words" % stem)
        part_first.append(first)
        part_last.append(len(words) - 1)
    return stems, words, lines_tbl, line_of, part_first, part_last


def _stage_orig(cfg, book):
    stream = _Stream(book.items)
    names, ow, lines_tbl, line_of, part_first, part_last = _read_orig(cfg)
    ew = stream.words
    blocks = [
        b for b in difflib.SequenceMatcher(None, ow, ew, autojunk=True).get_matching_blocks()
        if b[2] >= cfg.solid
    ]
    o_in = bytearray(len(ow))
    e_in = bytearray(len(ew))
    for a, b, s in blocks:
        o_in[a:a + s] = b"\x01" * s
        e_in[b:b + s] = b"\x01" * s
    to_epub = _mapper(blocks, 0)
    to_orig = _mapper(blocks, 1)
    if not blocks:
        raise ValueError("no solid matching block between the .orig parts and the EPUB")

    decisions = []
    bounds = []
    prev_end = -1
    for k, name in enumerate(names):
        s_raw = to_epub(part_first[k], "after")[0]
        e_raw = to_epub(part_last[k], "before")[0]
        s = min(max(s_raw, 0), len(ew) - 1)
        if s <= prev_end:
            decisions.append({"part": name, "what": "start_moved_after_previous_end", "from": s, "to": prev_end + 1})
            s = prev_end + 1
        if s > len(ew) - 1:
            raise ValueError("EPUB words exhausted before part %s" % name)
        e = min(max(e_raw, 0), len(ew) - 1)
        if e < s:
            decisions.append({"part": name, "what": "end_moved_to_start", "from": e, "to": s})
            e = s
        bounds.append((s, e))
        prev_end = e

    def edge(g, w):
        li = line_of[g]
        _, ln, first, cnt = lines_tbl[li]
        return {
            "orig_line": ln,
            "epub_word": w,
            "item": stream.inc[stream.item_of[w]].index,
            "word_offset": stream.offset_of[w],
            "score": round(sum(o_in[first:first + cnt]) / cnt, 4),
            "exact": bool(o_in[g]),
        }

    texts, flags, parts_report = {}, {}, {}
    for k, name in enumerate(names):
        s, e = bounds[k]
        text, start, end = _cut(stream, stream.item_of[s], stream.offset_of[s], stream.item_of[e], stream.offset_of[e])
        texts[name] = text
        flags[name] = (start, end)
        parts_report[name] = {
            "items": [stream.inc[stream.item_of[s]].index, stream.inc[stream.item_of[e]].index],
            "epub_words": [s, e],
            "orig_words": part_last[k] - part_first[k] + 1,
            "start": dict(edge(part_first[k], s), **start),
            "end": dict(edge(part_last[k], e), **end),
        }

    gaps = []
    for k in range(len(names) + 1):
        after = names[k - 1] if k else None
        before = names[k] if k < len(names) else None
        lo = stream.inc[stream.item_of[bounds[k - 1][1]]].index if k else None
        hi = stream.inc[stream.item_of[bounds[k][0]]].index if before else None
        words = (bounds[k][0] if before else len(ew)) - (bounds[k - 1][1] + 1 if k else 0)
        g = _gap(after, before, words, _gap_items(book.items, lo, hi))
        if g:
            gaps.append(g)

    numbers = dict((i, n) for i, n in book.report["chapter_numbers"])
    rows, starts, counts = [], {n: set() for n in names}, {"placed": 0, "outside_parts": 0, "no_match": 0, "inside_line": 0}
    for p, it in enumerate(stream.inc):
        if it.kind == "body":
            continue
        fw = stream.first_word[p]
        if fw is not None and e_in[fw]:
            anchor = fw
        else:
            anchor = next(
                (stream.first_word[q] for q in range(p + 1, len(stream.inc))
                 if stream.inc[q].kind == "body" and stream.first_word[q] is not None),
                None,
            )
        placed, reason = None, None
        if anchor is None or not any(s <= anchor <= e for s, e in bounds):
            reason = "outside_parts" if anchor is not None else "no_match"
        else:
            hit = to_orig(anchor, "nearest")
            if hit is None or hit[1] > _MAX_ANCHOR_DISTANCE:
                reason = "no_match"
            else:
                g = min(max(hit[0], 0), len(ow) - 1)
                k, ln, first, _ = lines_tbl[line_of[g]]
                if g - first < _MAX_LINE_RANK:
                    placed = {"part": names[k], "line": ln}
                    starts[names[k]].add(ln)
                else:
                    reason = "inside_line"
        counts["placed" if placed else reason] += 1
        epub_part = next(
            (names[k] for k, v in enumerate(parts_report.values()) if v["items"][0] <= it.index <= v["items"][1]),
            None,
        )
        rows.append({
            "index": it.index,
            "kind": it.tag if it.kind == "heading" else "chapter",
            "number": numbers.get(it.index),
            "epub_part": epub_part,
            "placed": placed,
            "reason": reason,
        })

    report = {
        "mode": "orig",
        "epub": book.report,
        "words": {
            "orig": len(ow),
            "epub": len(ew),
            "solid": cfg.solid,
            "orig_in_solid": sum(o_in),
            "epub_in_solid": sum(e_in),
        },
        "parts": parts_report,
        "gaps": gaps,
        "decisions": decisions,
        "chapters": {
            "rows": rows,
            "counts": counts,
            "excluded_tab_headings": [i.index for i in book.items if i.kind == "heading" and i.excluded],
        },
    }
    out = os.path.join(cfg.work_dir, "source")
    for n in names:
        _write(os.path.join(out, "epub", n + ".txt"), texts[n])
    _write(os.path.join(out, "epub", "placement.json"), _dump(_placement("epub", names, flags, {})))
    _write(
        os.path.join(out, "orig", "placement.json"),
        _dump(_placement("orig", names, flags, {n: sorted(v) for n, v in starts.items()})),
    )
    return report


def _stage_specs(cfg, book):
    stream = _Stream(book.items)
    inc = stream.inc
    pos = []
    for spec in cfg.parts:
        if isinstance(spec.start, int):
            p = next((q for q, it in enumerate(inc) if it.index >= spec.start), None)
            if p is None:
                raise ValueError("part %s: no included item at or after index %d" % (spec.name, spec.start))
        else:
            want = [w for w, _, _ in _words(unicodedata.normalize("NFC", spec.start))]
            cands = [
                q for q in range(len(inc))
                if want and [w for w, _, _ in stream.spans[q][: len(want)]] == want
            ]
            if len(cands) != 1:
                raise ValueError(
                    "part %s: first words match %d items: %s" % (spec.name, len(cands), [inc[q].index for q in cands])
                )
            p = cands[0]
        if pos and p <= pos[-1]:
            raise ValueError("part %s does not start after the previous part" % spec.name)
        pos.append(p)
    names = [s.name for s in cfg.parts]
    texts, flags, parts_report, gaps = {}, {}, {}, []
    for k, name in enumerate(names):
        pe = (pos[k + 1] if k + 1 < len(pos) else len(inc)) - 1
        block = inc[pos[k]:pe + 1]
        texts[name] = to_book_text(block)
        flags[name] = (
            {"inside_paragraph": False, "mid_sentence": False},
            {"inside_paragraph": False, "mid_sentence": False, "next": stream.next_kind(pe)},
        )
        parts_report[name] = {"items": [block[0].index, block[-1].index], "included_items": len(block)}
    first_idx = inc[pos[0]].index
    gap = _gap(None, names[0], sum(len(s) for s in stream.spans[:pos[0]]), _gap_items(book.items, None, first_idx))
    if gap:
        gaps.append(gap)
    report = {
        "mode": "specs",
        "epub": book.report,
        "parts": parts_report,
        "gaps": gaps,
        "decisions": [],
    }
    out = os.path.join(cfg.work_dir, "source")
    for n in names:
        _write(os.path.join(out, "epub", n + ".txt"), texts[n])
    _write(os.path.join(out, "epub", "placement.json"), _dump(_placement("epub", names, flags, {})))
    return report


def _render_md(report):
    lines = ["# Source stage report", "", "mode: %s" % report["mode"], ""]
    e = report["epub"]
    lines += [
        "## EPUB", "",
        "- files read: %d, items: %s" % (e["files"], json.dumps(e["items"])),
        "- empty blocks skipped: %d" % e["empty_skipped"],
        "- table items (excluded: %s): %s" % (e["tables_excluded"], e["table_items"]),
        "- text outside blocks, chars per file: %s" % json.dumps(e["outside_text_chars"]),
        "- replacements: %s" % json.dumps(e["replacements"], ensure_ascii=True),
        "",
    ]
    if "words" in report:
        lines += ["## Words", "", "- " + json.dumps(report["words"]), ""]
    lines += ["## Parts", ""]
    for name, p in report["parts"].items():
        lines.append("- %s: %s" % (name, json.dumps(p)))
    lines += ["", "## Gaps", ""]
    lines += ["- " + json.dumps(g) for g in report["gaps"]] or ["- none"]
    lines += ["", "## Decisions", ""]
    lines += ["- " + json.dumps(d) for d in report["decisions"]] or ["- none"]
    if "chapters" in report:
        lines += ["", "## Chapters", "", "- counts: " + json.dumps(report["chapters"]["counts"]),
                  "- headings left out because they contain a TAB: %s" % report["chapters"]["excluded_tab_headings"], ""]
        lines += ["- " + json.dumps(r) for r in report["chapters"]["rows"]]
    return "\n".join(lines) + "\n"


def run_source_stage(cfg: SourceConfig) -> dict:
    book = read_epub(cfg.epub)
    if cfg.orig_dir:
        report = _stage_orig(cfg, book)
    elif cfg.parts:
        report = _stage_specs(cfg, book)
    else:
        raise ValueError("either orig_dir or parts is required")
    out = os.path.join(cfg.work_dir, "source")
    _write(os.path.join(out, "report.json"), _dump(report))
    _write(os.path.join(out, "report.md"), _render_md(report))
    return report
