"""
Training units of normalised book text (plan Step 4.2): the pure function that turns the segmenter's sentences into
the texts of the training clips. The corpus preparation and the inference (Step 12) both call `build_units`, so
training and rendering see the same units.

A unit is one sentence, except that a quote or bracket still open at the end of a sentence keeps the next sentence
of the same paragraph in the unit while the text estimate of the unit stays within `max_s` ("„Ne. Nikdy. Rozumíš?“
zašeptala." is one unit). When the cap stops such a quote, the next unit starts inside it and goes on by the same
rule. A unit whose estimate exceeds `max_s` is cut after a clause mark (, ; : — – …), never inside a <cs>/<sk>/<en>
span; where no clause cut leaves three words on each side it stays whole and is flagged `over_cap`. A unit never
crosses a paragraph, short sentences are never merged, and a closed quote is never paired with the narration after it.

`boundary` is the label of the unit's last sentence (every piece of a cut sentence before its last is "mid") and
`pause` names the pause that follows the unit: "clause" after a clause cut, else "sentence", "paragraph" or "chapter"
by the boundary.

Quotes are tracked by position, so the straight " and ' that open and close alike work: „ ‚ ( [ open and ) ] ” ’
close; “ ‘ » « ‹ › " ' open after the start, a space, a dash or an opening mark when a non-space follows, and close
otherwise; a mark between two letters is an apostrophe; a closer never takes the depth below 0; an old Slovak
,Ahoj!' opens with a comma after a space.

Usage:
    build_units(segment("„Ne. Nikdy. Rozumíš?“ zašeptala."))   # one Unit holding the three sentences
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

from src.text.segmenter import Sentence

_SPAN = re.compile(r"<(cs|sk|en)>.*?</\1>", re.DOTALL)
_TAG = re.compile(r"</?(?:cs|sk|en)>")
_CLAUSE = re.compile(r"(?<=[,;:—–…])\s+(?=\S)")
_OPENING_CONTEXT = frozenset(" \t\n—–-([„‚“‘»«‹›\"'")
_PAUSE = {"mid": "sentence", "para_end": "paragraph", "chapter_end": "chapter"}


@dataclass(frozen=True)
class Unit:
    """`first` and `last` index the input sentences (equal for kind "sentence" and "split"); `kind` is "sentence",
    "joined" (two or more sentences) or "split" (a clause piece of one sentence). `open_before` and `open_after` are
    the quote or bracket depth entering and leaving the unit; `over_cap` marks a unit above the cap that has no
    clause cut."""
    text: str
    boundary: str
    pause: str
    chapter: int
    paragraph: int
    first: int
    last: int
    kind: str
    heading: bool = False
    open_before: int = 0
    open_after: int = 0
    over_cap: bool = False


def estimate_s(text: str, a: float = -0.52, b: float = 0.0843, c: float = 0.394, e: float = 1.27) -> float:
    """Spoken seconds of a text: a + b per letter + c per , ; : — – + e per . ! ? …, the language tags not counted;
    fitted on the narrator's old clips, to be re-fitted per narrator."""
    text = _TAG.sub("", text)
    return (a + b * sum(ch.isalpha() for ch in text) + c * sum(ch in ",;:—–" for ch in text)
            + e * sum(ch in ".!?…" for ch in text))


def _depths(text: str, depth: int) -> List[int]:
    """The quote and bracket depth after each character of text, entering with `depth`."""
    text = _SPAN.sub(lambda m: " " * len(m.group()), text)
    depths = []
    for i, ch in enumerate(text):
        prev = text[i - 1] if i else " "
        nxt = text[i + 1] if i + 1 < len(text) else " "
        if ch in "„‚([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        elif ch in "“”‘’»«‹›\"'":
            if prev.isalnum() and nxt.isalnum():
                pass
            elif ch in "”’":
                depth = max(0, depth - 1)
            elif prev in _OPENING_CONTEXT and not nxt.isspace():
                depth += 1
            else:
                depth = max(0, depth - 1)
        elif ch == "," and prev in " \t\n" and nxt.isalpha():
            depth += 1
        depths.append(depth)
    return depths


def quote_depth(text: str, depth: int = 0) -> int:
    """How many quotes or brackets are open at the end of text, scanning from `depth` open at its start."""
    return (_depths(text, depth) or [depth])[-1]


def quote_spans(text: str, depth: int = 0) -> List[Tuple[int, int]]:
    """The outermost quote or bracket spans of text as (start, end) offsets; a span open on entry (`depth`) starts at 0
    and one still open at the end stops at the end of the text."""
    spans, start, before = [], (0 if depth else None), depth
    for i, after in enumerate(_depths(text, depth)):
        if before == 0 and after > 0:
            start = i
        elif before > 0 and after == 0 and start is not None:
            spans.append((start, i + 1))
            start = None
        before = after
    if start is not None:
        spans.append((start, len(text)))
    return spans


def _pieces(text: str, depth: int, max_s: float, estimate: Callable[[str], float]) -> List[Tuple[str, int, int, bool]]:
    """The clause pieces of text as (text, depth entering, depth leaving, over the cap with no cut)."""
    if estimate(text) <= max_s:
        return [(text, depth, quote_depth(text, depth), False)]
    spans = [(s.start(), s.end()) for s in _SPAN.finditer(text)]
    cuts = [m.end() for m in _CLAUSE.finditer(text) if not any(s < m.start() < e for s, e in spans)]
    cuts = [c for c in cuts if len(text[:c].split()) >= 3 and len(text[c:].split()) >= 3]
    if not cuts:
        return [(text, depth, quote_depth(text, depth), True)]
    cuts = [c for c in cuts if quote_depth(text[:c], depth) == 0] or cuts
    longer = {c: max(estimate(text[:c]), estimate(text[c:])) for c in cuts}
    near = [c for c in cuts if longer[c] <= min(longer.values()) + 1.0]
    cut = min([c for c in near if text[:c].rstrip()[-1] in ";:—–…"] or near, key=longer.get)
    head, tail = text[:cut].strip(), text[cut:].strip()
    first = _pieces(head, depth, max_s, estimate)
    return first + _pieces(tail, first[-1][2], max_s, estimate)


def _continues(sentences: Sequence[Sentence], j: int, start: Sentence) -> bool:
    """Whether the sentence after index j is in the paragraph of `start`."""
    return (j + 1 < len(sentences)
            and (sentences[j + 1].chapter, sentences[j + 1].paragraph) == (start.chapter, start.paragraph))


def build_units(sentences: Sequence[Sentence], max_s: float = 17.0,
                estimate: Callable[[str], float] = estimate_s) -> List[Unit]:
    """The units of the sentences of segment(), in order; `estimate` gives the spoken seconds of a text, `max_s` the
    cap."""
    units: List[Unit] = []
    i, carry = 0, 0
    while i < len(sentences):
        start = sentences[i]
        j, text = i, start.text
        depth = quote_depth(text, carry)
        while depth > 0 and _continues(sentences, j, start):
            joined = f"{text} {sentences[j + 1].text}"
            if estimate(joined) > max_s:
                break
            j, text = j + 1, joined
            depth = quote_depth(text, carry)
        last = sentences[j]
        pieces = _pieces(text, carry, max_s, estimate)
        for k, (piece, opened, left, over) in enumerate(pieces):
            final = k + 1 == len(pieces)
            boundary = last.boundary if final else "mid"
            kind = "split" if len(pieces) > 1 else ("joined" if j > i else "sentence")
            units.append(Unit(piece, boundary, _PAUSE[boundary] if final else "clause", start.chapter, start.paragraph,
                              i, j, kind, start.heading, opened, left, over))
        carry = depth if _continues(sentences, j, start) else 0
        i = j + 1
    return units
