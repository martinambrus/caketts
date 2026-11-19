"""
Sentence and boundary segmentation of normalised book text (plan Step 2.3).

The input is the normalizer's output: a blank line ends a paragraph, a line starting with "# " is a chapter heading
that is read aloud as a paragraph of its own (its sentences have no "# "), and the lines of a paragraph are joined by
one space. Every sentence is labelled by what follows it: "chapter_end" for the last sentence of a chapter or of the text
(the end of the input ends its paragraph and its chapter), "para_end" for the last of any other paragraph, else "mid".
The pause comes from this label alone, in the training data (Step 4.2) and at inference (Step 12), which both call `segment`.

A sentence ends at . ! ? or … (with the quotes and brackets that close it) before a word that starts uppercase: the
normalizer keeps a "." where an abbreviation's or a Roman numeral's period ends a sentence and capitalises a
sentence-initial number word. A lowercase word goes on ("„Kam jdeš?“ zeptal se.", "A pak… nic."), a single capital before
the period is an initial ("V. Havel"), and a <cs>/<sk>/<en> span is never split.

Usage:
    segment("# První\\n\\nPřišel. Odešel.\\n")[2]   # Sentence('Odešel.', 'chapter_end', chapter=0, paragraph=1)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, List, Tuple

_CLOSING = re.escape("\"'“”‘’»«‹›)]")  # may follow the mark of the sentence they close
_OPENING = re.escape("\"'„“‚‘»«‹›([—–-,…")  # may open the next sentence; "," is an old Slovak opening quote: ,Ahoj!'
# A span is matched whole, so no mark inside it is a candidate; a mark ends a sentence after a word or a closing quote,
# before a space and a letter, whose case `_sentences` checks.
_CANDIDATE = re.compile(
    r"<(cs|sk|en)>.*?</\1>"
    rf"|(?<=[\w>{_CLOSING}])[.!?…]+[{_CLOSING}]*(?=\s[\s{_OPENING}]*(?:<(?:cs|sk|en)>)?(?P<next>[^\W\d_]))")


@dataclass(frozen=True)
class Sentence:
    """`boundary` is what follows the sentence: "mid", "para_end" or "chapter_end" (the keys of
    src.data.dataset.BOUNDARY_IDS). `chapter` and `paragraph` count from 0, the paragraph within its chapter."""
    text: str
    boundary: str
    chapter: int
    paragraph: int


def _paragraphs(text: str) -> Iterator[Tuple[bool, str]]:
    """The paragraphs of book text as (is_heading, text), the lines of a paragraph joined by one space. A heading line
    also ends the paragraph before it."""
    lines: List[str] = []
    for line in [*text.splitlines(), ""]:
        if line.startswith("# ") or not line.strip():
            if lines:
                yield False, " ".join(lines)
                lines = []
            if line.startswith("# "):
                yield True, line[2:]
        else:
            lines.append(line.strip())


def _initial(before: str) -> bool:
    """Whether `before` ends in a single capital: "V" in "V. Havel" is an initial, not the end of a sentence."""
    return before[-1:].isupper() and not before[-2:-1].isalpha()


def _sentences(paragraph: str) -> List[str]:
    """The sentences of one paragraph, without those that have no letter ("* * *")."""
    sentences, start = [], 0
    for m in _CANDIDATE.finditer(paragraph):
        letter = m.group("next")
        if letter and letter.isupper() and not (m.group() == "." and _initial(paragraph[:m.start()])):
            sentences.append(paragraph[start:m.end()])
            start = m.end()
    sentences.append(paragraph[start:])
    return [s for s in (s.strip() for s in sentences) if any(ch.isalpha() for ch in s)]


def segment(text: str) -> List[Sentence]:
    """Split normalised book text into sentences, each labelled with the boundary that follows it. A heading line starts
    a chapter, even one with no text; text before the first heading is the first chapter, so a text without a heading
    is one chapter."""
    found: List[Tuple[str, int, int]] = []  # sentence, chapter, paragraph
    chapter = paragraph = -1
    for heading, block in _paragraphs(text):
        sentences = _sentences(block)
        if heading or (chapter < 0 and sentences):
            chapter, paragraph = chapter + 1, -1
        if sentences:
            paragraph += 1
            found += [(s, chapter, paragraph) for s in sentences]
    result = []
    for i, (s, c, p) in enumerate(found):
        if i + 1 == len(found) or found[i + 1][1] != c:
            boundary = "chapter_end"
        elif found[i + 1][2] != p:
            boundary = "para_end"
        else:
            boundary = "mid"
        result.append(Sentence(s, boundary, c, p))
    return result
