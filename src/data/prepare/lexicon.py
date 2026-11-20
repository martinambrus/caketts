"""
Pronunciation lexicon of a book (plan Step 4.2): the words the book prints differently from how the narrator says them.

derive aligns the book text of each part with the rewrite of it that matches the narration, word by word, and lists every
difference of at most six words per side (thirty when the book side holds a symbol the normalizer cannot read) as a
pair (book form, spoken form) with a category. The lexicon file is a TSV with the header book_form, spoken_form, category,
count, consistent, apply; a hand-written file needs book_form, spoken_form and apply, and an empty file is an empty lexicon.
apply_lexicon replaces each applied book form as a whole word or whole phrase (a lower-case form also matches its capitalised start of a sentence), and a symbol the normalizer cannot read wherever it
stands, except between two digits. compare lists where the normalizer's reading of the book text differs from the rewrite.

Usage:
    derive_to_file("work/source/epub", "raw", "work/lexicon.tsv")
"""
from __future__ import annotations

import difflib
import os
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from src.data.prepare.source import _natural, _words
from src.text.normalizer import TextNormalizer

_TOKEN = re.compile(r"[^\W\d_]+|\d+|[^\w\s]")
_HEADING = re.compile(r"(?m)^# ")
_WORD_RUN = re.compile(r"\w+")
_OPENING = " \t\"'„“”‚‘’»«‹›()[]—–-#"
_SENTENCE_END = ".!?…\n"
_COLUMNS = ["book_form", "spoken_form", "category", "count", "consistent", "apply"]
_REQUIRED = ("book_form", "spoken_form", "apply")
_CATEGORIES = ["symbol", "respelling", "number", "abbreviation", "insertion", "deletion", "other"]
_MAX_SIDE = 6
_MAX_SYMBOL_SIDE = 30
_REACH = 12
_SHOWN = 80


def _prepare(text: str) -> str:
    return unicodedata.normalize("NFC", re.sub(r"\r\n|\r", "\n", text))


def _fails(normalizer, text: str) -> bool:
    try:
        normalizer.normalize(text)
    except ValueError:
        return True
    return False


def _unreadable(normalizer, text: str, start: int, cache: dict) -> bool:
    ch = text[start]
    if ch not in cache:
        cache[ch] = _fails(normalizer, f"a {ch} b")
    if not cache[ch]:
        return False
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", start)
    line_end = len(text) if line_end < 0 else line_end
    lo, hi = max(line_start, start - _REACH), min(line_end, start + 1 + _REACH)
    whole = text[lo:hi]
    cut = whole
    if lo > line_start and not text[lo - 1].isspace():
        cut = cut.partition(" ")[2]
    if hi < line_end and not text[hi].isspace():
        cut = cut.rpartition(" ")[0]
    return _fails(normalizer, cut if ch in cut else whole)


def _tokens(text: str, normalizer=None, cache: Optional[dict] = None) -> List[Tuple[str, int, int]]:
    text = _HEADING.sub("  ", text)
    cache = {} if cache is None else cache
    return [t for t in _words(text, _TOKEN)
            if t[0][0].isalnum() or (normalizer is not None and _unreadable(normalizer, text, t[1], cache))]


def _opcodes(a, b):
    return difflib.SequenceMatcher(None, [t[0] for t in a], [t[0] for t in b], autojunk=False).get_opcodes()


def _surface(text: str, tokens) -> str:
    return " ".join(text[tokens[0][1]:tokens[-1][2]].split()) if tokens else ""


def _spoken(text: str, tokens) -> str:
    return " ".join(text[s:e] for _, s, e in tokens)


def _category(book: List[str], spoken: List[str], abbreviation: bool) -> str:
    if any(not w[0].isalnum() for w in book):
        return "symbol"
    if any(w[0].isdigit() for w in book):
        return "number"
    if abbreviation:
        return "abbreviation"
    if not book:
        return "insertion"
    if not spoken:
        return "deletion"
    return "respelling" if all(w.isalpha() for w in book + spoken) else "other"


def _is_abbreviation(text: str, tokens, abbreviations) -> Tuple[bool, str]:
    surface = _surface(text, tokens)
    if not surface:
        return False, surface
    dotted = surface + "." if text[tokens[-1][2]:tokens[-1][2] + 1] == "." else surface
    if dotted.lower() in abbreviations:
        return True, dotted
    return surface.lower() in abbreviations, surface


# ---- applying ----------------------------------------------------------------------------------------------

def _starts_sentence(text: str, pos: int) -> bool:
    while pos > 0 and text[pos - 1] in _OPENING:
        pos -= 1
    return pos == 0 or text[pos - 1] in _SENTENCE_END


def _is_word(ch: str) -> bool:
    return bool(ch) and (ch.isalnum() or ch == "_")


class _Matcher:
    def __init__(self, forms):
        self.words = defaultdict(list)
        self.symbols = []
        for form in forms:
            if not form:
                continue
            if not _is_word(form[0]):
                self.symbols.append(form)
                continue
            variants = [(form, form, False)]
            if form[0].islower():
                variants.append((form[0].upper() + form[1:], form, True))
            for variant in variants:
                self.words[_WORD_RUN.match(variant[0]).group()].append(variant)
        for variants in self.words.values():
            variants.sort(key=lambda v: (-len(v[0]), v[2]))

    def hits(self, text: str) -> List[Tuple[int, int, str, bool]]:
        found = []
        for m in _WORD_RUN.finditer(text):
            for variant, form, capital in self.words.get(m.group(), ()):
                end = m.start() + len(variant)
                if (text.startswith(variant, m.start()) and not _is_word(text[end:end + 1])
                        and (not capital or _starts_sentence(text, m.start()))):
                    found.append((m.start(), end, form, capital))
                    break
        for form in self.symbols:
            for m in re.finditer(re.escape(form), text):
                if not (text[m.start() - 1:m.start()].isdigit() and text[m.end():m.end() + 1].isdigit()):
                    found.append((m.start(), m.end(), form, False))
        found.sort(key=lambda h: (h[0], h[0] - h[1]))
        kept, last = [], 0
        for hit in found:
            if hit[0] >= last:
                kept.append(hit)
                last = hit[1]
        return kept


def _symbol_edit(text: str, start: int, end: int, spoken: str) -> Tuple[int, int, str]:
    left = text[start - 1:start] if start else "\n"
    right = text[end:end + 1] or "\n"
    if not spoken:
        if right == " ":
            return start, end + 1, spoken
        if left == " ":
            return start - 1, end, spoken
        return start, end, "" if "\n" in (left + right) else " "
    return start, end, ("" if left in " \n" else " ") + spoken + ("" if right in " \n" else " ")


def apply_lexicon(text: str, entries: Dict[str, str]) -> Tuple[str, int]:
    hits = _Matcher(entries).hits(text)
    out, pos = [], 0
    for start, end, form, capital in hits:
        spoken = entries[form]
        if capital and spoken:
            spoken = spoken[0].upper() + spoken[1:]
        if not _is_word(form[0]):
            start, end, spoken = _symbol_edit(text, start, end, spoken)
        out += [text[pos:start], spoken]
        pos = end
    out.append(text[pos:])
    return "".join(out), len(hits)


# ---- file --------------------------------------------------------------------------------------------------

def write_lexicon(pairs: List[dict], path: str) -> None:
    lines = ["\t".join(_COLUMNS)]
    for p in pairs:
        for value in (p["book_form"], p["spoken_form"]):
            if "\t" in value or "\n" in value:
                raise ValueError(f"a lexicon form holds a TAB or a line break: {value!r}")
        lines.append("\t".join([p["book_form"], p["spoken_form"], p["category"], str(p["count"]),
                                "y" if p["consistent"] else "n", "y" if p["apply"] else "n"]))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def read_lexicon(path: str) -> Dict[str, str]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        lines = [line for line in re.split(r"\r?\n", f.read()) if line.strip()]
    if not lines:
        return {}
    header = lines[0].split("\t")
    missing = [c for c in _REQUIRED if c not in header]
    if missing:
        raise ValueError(f"{path}: the header lacks the column(s) {missing}")
    book, spoken, apply = (header.index(c) for c in _REQUIRED)
    entries: Dict[str, str] = {}
    for number, line in enumerate(lines[1:], 2):
        cells = line.split("\t")
        if len(cells) != len(header):
            raise ValueError(f"{path}: line {number} has {len(cells)} cells, the header {len(header)}")
        flag = cells[apply].strip().lower()
        if flag not in ("y", "n"):
            raise ValueError(f"{path}: line {number}: apply must be y or n, got {cells[apply]!r}")
        if flag == "n":
            continue
        if not cells[book]:
            raise ValueError(f"{path}: line {number}: an applied row has an empty book_form")
        if cells[book] in entries:
            raise ValueError(f"{path}: line {number}: two applied rows for the book form {cells[book]!r}")
        entries[cells[book]] = cells[spoken]
    return entries


# ---- deriving ----------------------------------------------------------------------------------------------

def derive(book_parts: Dict[str, str], rewrite_parts: Dict[str, str], normalizer=None) -> List[dict]:
    normalizer = normalizer or TextNormalizer("cs")
    cache: dict = {}
    seen: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    books = {}
    for name, raw in book_parts.items():
        book, rewrite = _prepare(raw), _prepare(rewrite_parts[name])
        books[name] = book
        bt, rt = _tokens(book, normalizer, cache), _tokens(rewrite)
        for tag, i1, i2, j1, j2 in _opcodes(bt, rt):
            if tag == "equal":
                continue
            b, r = bt[i1:i2], rt[j1:j2]
            limit = _MAX_SYMBOL_SIDE if any(not t[0][0].isalnum() for t in b) else _MAX_SIDE
            if i2 - i1 > limit or j2 - j1 > limit:
                continue
            if (b and "\n" in book[b[0][1]:b[-1][2]]) or (limit > _MAX_SIDE and r and "\n" in rewrite[r[0][1]:r[-1][2]]):
                continue
            abbreviation, form = _is_abbreviation(book, b, normalizer.abbreviations)
            category = _category([t[0] for t in b], [t[0] for t in r], abbreviation)
            seen[(form, _spoken(rewrite, r))][category] += 1
    forms = {form for form, _ in seen}
    occurrences: Counter = Counter()
    matcher = _Matcher(forms)
    for book in books.values():
        occurrences.update(h[2] for h in matcher.hits(book))
    changed: Counter = Counter()
    spoken_of: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for (form, spoken), votes in seen.items():
        changed[form] += sum(votes.values())
        spoken_of[form].append((-sum(votes.values()), spoken))
    pairs = []
    for (form, spoken), votes in seen.items():
        count, distinct = sum(votes.values()), len(spoken_of[form])
        consistent = distinct == 1 and (not form or changed[form] == occurrences[form])
        category = votes.most_common(1)[0][0]
        top = min(spoken_of[form])[1]
        apply = (category == "respelling" and consistent) or (category == "symbol" and spoken == top)
        pairs.append({"book_form": form, "spoken_form": spoken, "category": category, "count": count,
                      "distinct": distinct, "consistent": consistent, "apply": apply})
    pairs.sort(key=lambda p: (_CATEGORIES.index(p["category"]), -p["count"], p["book_form"], p["spoken_form"]))
    return pairs


def derive_to_file(book_dir: str, rewrite_dir: str, path: str, book_suffix: str = ".txt",
                   rewrite_suffix: str = ".txt.orig", normalizer=None) -> List[dict]:
    names = sorted((f[:-len(book_suffix)] for f in os.listdir(book_dir) if f.endswith(book_suffix)), key=_natural)
    if not names:
        raise ValueError(f"no <part>{book_suffix} file in {book_dir}")
    books, rewrites = {}, {}
    for name in names:
        for parts, folder, suffix in ((books, book_dir, book_suffix), (rewrites, rewrite_dir, rewrite_suffix)):
            with open(os.path.join(folder, name + suffix), "rb") as f:
                parts[name] = f.read().decode("utf-8-sig")
    pairs = derive(books, rewrites, normalizer)
    write_lexicon(pairs, path)
    return pairs


# ---- comparing ---------------------------------------------------------------------------------------------

def _matched(a: List[str], b: List[str], solid: int) -> int:
    blocks = difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks()
    return sum(size for _, _, size in blocks if size >= solid)


def _letters(tokens) -> List[str]:
    return [t[0] for t in tokens if t[0][0].isalpha()]


def _shown(text: str) -> str:
    return text if len(text) <= _SHOWN else text[:_SHOWN] + "…"


def compare(raw: str, given: str, normalised: str, rewrite: str, abbreviations, solid: int = 3) -> dict:
    """The rewrite words matched in blocks of at least `solid` words by the book text before the lexicon (raw) and by
    its normalised reading, and the rows (line, category, book form, reading, rewrite form) where the reading and the
    rewrite differ. given is the text that was normalised, normalised its reading."""
    rewrite = _prepare(rewrite)
    a, n, r = _tokens(given), _tokens(normalised), _tokens(rewrite)
    words = _letters(r)
    spans: List[Tuple[int, int]] = [(0, 0)] * len(n)
    for tag, i1, i2, j1, j2 in _opcodes(a, n):
        for j in range(j1, j2):
            spans[j] = (i1 + j - j1, i1 + j - j1 + 1) if tag == "equal" else (i1, i2)
    rows = []
    for tag, j1, j2, k1, k2 in _opcodes(n, r):
        if tag == "equal":
            continue
        if j2 > j1:
            lo, hi = spans[j1][0], spans[j2 - 1][1]
        else:
            lo = hi = spans[j1][0] if j1 < len(n) else len(a)
        b = a[lo:hi]
        abbreviation, form = _is_abbreviation(given, b, abbreviations)
        category = _category([t[0] for t in b], [t[0] for t in r[k1:k2]], abbreviation)
        at = a[min(lo, len(a) - 1)][1] if a else 0
        rows.append((given.count("\n", 0, at) + 1, category, _shown(form), _shown(_spoken(normalised, n[j1:j2])),
                     _shown(_spoken(rewrite, r[k1:k2]))))
    return {"rewrite_words": len(words), "matched_before": _matched(words, _letters(_tokens(raw)), solid),
            "matched_after": _matched(words, _letters(n), solid), "rows": rows}
