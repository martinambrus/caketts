import dataclasses
import random
import re
from collections import namedtuple
from functools import lru_cache

import pytest
from src.text.phonemizer import CzechSlovakPhonemizer
from src.text.segmenter import Sentence, segment
from src.text.units import Unit, build_units, estimate_s, quote_depth

WORDS = ["dobrý", "starý", "velký", "hezký", "mladý"]  # five letters each


def words(n, k=0):
    return " ".join(WORDS[(k + i) % 5] for i in range(n))


def cap(s):
    return s[0].upper() + s[1:]


# Letters x 0.0765 - 0.70 + 0.297 per [,;:—–] or spaced hyphen + 0.788 per [.!?…] (a run of dots once), by hand.
# LONG: 30 + 30 + 27 words. Pieces 150 letters + "," = 11.475 - 0.70 + 0.297 = 11.072; 150 letters + ";" = 11.072;
# 135 letters + "." = 10.3275 - 0.70 + 0.788 = 10.4155. Whole: 435 letters, 2 marks, 1 stop = 33.2775 - 0.70 + 0.594 + 0.788 = 33.9595.
LONG = cap(words(30)) + ", " + words(30) + "; " + words(27) + "."
LONG_PIECES = [(cap(words(30)) + ",", "mid"), (words(30) + ";", "mid"), (words(27) + ".", "chapter_end")]
# ONLY_COMMA: 89 words, 445 letters, 1 comma, 1 stop = 34.0425 - 0.70 + 0.297 + 0.788 = 34.4275; the only cut leaves a 1-word piece.
ONLY_COMMA = "Dobrý, " + words(88, 1) + "."
# QUOTE4: sentences of 20, 18, 18 and 29 letters (the last with the attribution), one stop each except the last (two).
# S1 = 1.618; S1+S2 = 38 letters, 2 stops = 3.783; S1+S2+S3 = 5.948; S3+S4 = 47 letters, 3 stops = 5.2595;
# all four = 85 letters, 5 stops = 9.7425; S2 = S3 = 1.465; S4 = 29 letters, 2 stops = 3.0945.
QUOTE4 = ("{o}Věta číslo jedna je tady. Věta číslo dvě je tady. Věta číslo tři je tady. "
          "Věta číslo čtyři je tady.{c} dodal tiše.")

TABLE = {  # id: (book text, max_s, [(unit text, boundary)])
    "one-sentence": ("Přišel domů.", 17.0, [("Přišel domů.", "chapter_end")]),
    "two-sentences-stay-two": ("Přišel. Odešel.", 17.0, [("Přišel.", "mid"), ("Odešel.", "chapter_end")]),
    "quote-lowercase-attribution": (
        "„To nedává smysl,“ pomyslel si.", 17.0, [("„To nedává smysl,“ pomyslel si.", "chapter_end")]),
    "narration-colon-quote": ("Křičel: „Běž, utíkej!“", 17.0, [("Křičel: „Běž, utíkej!“", "chapter_end")]),
    "dash-turn-with-attribution": (
        "— Kam jdeš? — zeptal se.", 17.0, [("— Kam jdeš? — zeptal se.", "chapter_end")]),
    "three-sentence-quote-rejoined": (
        "„Ne. Nikdy. Rozumíš?“ zašeptala.", 17.0, [("„Ne. Nikdy. Rozumíš?“ zašeptala.", "chapter_end")]),
    "narration-plus-three-sentence-quote": (
        "Petr řekl: „Jdu domů. Přijdu zítra. Čekejte tu.“", 17.0,
        [("Petr řekl: „Jdu domů. Přijdu zítra. Čekejte tu.“", "chapter_end")]),
    "two-sentence-quote-then-narration": (
        "„Jdu domů. Přijdu zítra.“ Odešel.", 17.0,
        [("„Jdu domů. Přijdu zítra.“", "mid"), ("Odešel.", "chapter_end")]),
    "quote-then-name-led-narration": (
        "„Pojď sem!“ Petr se ani nehnul.", 17.0, [("„Pojď sem!“", "mid"), ("Petr se ani nehnul.", "chapter_end")]),
    "quote-split-by-attribution": (
        "„Jdu,“ řekl Petr, „ale vrátím se. Slibuji.“", 17.0,
        [("„Jdu,“ řekl Petr, „ale vrátím se. Slibuji.“", "chapter_end")]),
    "quote-exchange": (
        "Řekl: „Ahoj.“ „Ahoj,“ odpověděla. „Jak se máš?“", 17.0,
        [("Řekl: „Ahoj.“", "mid"), ("„Ahoj,“ odpověděla.", "mid"), ("„Jak se máš?“", "chapter_end")]),
    "nested-quote": (
        "„Řekl mi ‚Jdi pryč! Nechci tě vidět.‘ a odešel.“ Potom přišel.", 17.0,
        [("„Řekl mi ‚Jdi pryč! Nechci tě vidět.‘ a odešel.“", "mid"), ("Potom přišel.", "chapter_end")]),
    "straight-quotes": (
        '"Jdu domů. Přijdu zítra." Odešel.', 17.0, [('"Jdu domů. Přijdu zítra."', "mid"), ("Odešel.", "chapter_end")]),
    "guillemets": (
        "»Jdu domů. Přijdu zítra.« Odešel.", 17.0, [("»Jdu domů. Přijdu zítra.«", "mid"), ("Odešel.", "chapter_end")]),
    "english-quotes": (
        "“Jdu domů. Přijdu zítra.” Odešel.", 17.0, [("“Jdu domů. Přijdu zítra.”", "mid"), ("Odešel.", "chapter_end")]),
    "parenthesised-sentence": (
        "Odešel. (Nikdo nevěděl kam.) Potom přišel.", 17.0,
        [("Odešel.", "mid"), ("(Nikdo nevěděl kam.)", "mid"), ("Potom přišel.", "chapter_end")]),
    "parenthesis-rejoined": (
        "Odešel (nikdo nevěděl kam. Možná domů.) a nevrátil se.", 17.0,
        [("Odešel (nikdo nevěděl kam. Možná domů.) a nevrátil se.", "chapter_end")]),
    "dash-exchange": (
        "— Kam jdeš? — Domů. — Proč? — Protože je pozdě.", 17.0,
        [("— Kam jdeš?", "mid"), ("— Domů.", "mid"), ("— Proč?", "mid"), ("— Protože je pozdě.", "chapter_end")]),
    "initial-in-quote": (
        "„Dobrý den, pane V. Havle,“ řekla Anna.", 17.0, [("„Dobrý den, pane V. Havle,“ řekla Anna.", "chapter_end")]),
    "span-never-cut": (
        "Četl knihu <en>The Hobbit. Part One.</en> a odešel.", 17.0,
        [("Četl knihu <en>The Hobbit. Part One.</en> a odešel.", "chapter_end")]),
    "short-sentences-not-merged": (
        "Ano. Ne. Možná.", 17.0, [("Ano.", "mid"), ("Ne.", "mid"), ("Možná.", "chapter_end")]),
    "quote-over-two-paragraphs": (
        "„První odstavec řeči. Druhá věta.\n\n„Druhý odstavec řeči. Konec.“ Odešel.", 17.0,
        [("„První odstavec řeči. Druhá věta.", "para_end"), ("„Druhý odstavec řeči. Konec.“", "mid"),
         ("Odešel.", "chapter_end")]),
    "labels-and-heading": (
        "# První\n\nPřišel. Odešel.\n\nZůstal.", 17.0,
        [("První", "para_end"), ("Přišel.", "mid"), ("Odešel.", "para_end"), ("Zůstal.", "chapter_end")]),
    "long-sentence-clause-cut": (LONG, 17.0, LONG_PIECES),
    "long-quote-cap-cut-then-rejoined": (
        QUOTE4.format(o="„", c="“"), 5.5,
        [("„Věta číslo jedna je tady. Věta číslo dvě je tady.", "mid"),
         ("Věta číslo tři je tady. Věta číslo čtyři je tady.“ dodal tiše.", "chapter_end")]),
    "stray-closer-first": (
        "Nechci tě vidět.‘ a odešel.“ Potom přišel.", 17.0,
        [("Nechci tě vidět.‘ a odešel.“", "mid"), ("Potom přišel.", "chapter_end")]),
    "unclosed-quote-does-not-eat-next-paragraph": (
        "„Mluvil dlouho.\n\nPak ztichl.", 17.0, [("„Mluvil dlouho.", "para_end"), ("Pak ztichl.", "chapter_end")]),
    "only-comma-after-first-word": (ONLY_COMMA, 17.0, [(ONLY_COMMA, "chapter_end")]),
}

SENTENCE_COUNTS = {
    "one-sentence": 1, "two-sentences-stay-two": 2, "quote-lowercase-attribution": 1, "narration-colon-quote": 1,
    "dash-turn-with-attribution": 1, "three-sentence-quote-rejoined": 3, "narration-plus-three-sentence-quote": 3,
    "two-sentence-quote-then-narration": 3, "quote-then-name-led-narration": 2, "quote-split-by-attribution": 2,
    "quote-exchange": 3, "nested-quote": 3, "straight-quotes": 3, "guillemets": 3, "english-quotes": 3,
    "parenthesised-sentence": 3, "parenthesis-rejoined": 2, "dash-exchange": 4, "initial-in-quote": 1,
    "span-never-cut": 1, "short-sentences-not-merged": 3, "quote-over-two-paragraphs": 5, "labels-and-heading": 4,
    "long-sentence-clause-cut": 1, "long-quote-cap-cut-then-rejoined": 4, "stray-closer-first": 2,
    "unclosed-quote-does-not-eat-next-paragraph": 2, "only-comma-after-first-word": 1,
}


def check_row(builder, key):
    text, max_s, expected = TABLE[key]
    got = [(u.text, u.boundary) for u in builder(segment(text), max_s)]
    assert got == expected, f"{key}: {got} != {expected}"


Plain = namedtuple("Plain", "text boundary")


def identity_builder(sentences, max_s=17.0, estimate=None):
    return [Plain(s.text, s.boundary) for s in sentences]


def U(text, boundary, kind, open_before=0, open_after=0, over_cap=False):
    return (text, boundary, kind, open_before, open_after, over_cap)


# Cuts: the stripped piece texts; each piece estimated by hand as in the table comments (5-letter words, 0.3825 s each).
# cut-prefers-semicolon: 26 words "," 12 words ";" 23 words ".". Cut after "," = 9.542 | 13.7725 (longer 13.7725); cut after ";"
# = 14.429 | 8.8855 (longer 14.429); within 1 s (0.6565), so ";" wins and both pieces are <= 17.
# cut-prefers-minimiser: 26 "," 13 ";" 21 ".": cut after ";" = 14.8115 | 8.1195, 1.4215 above the comma cut (13.39), so "," wins.
# spaced-hyphen-is-the-only-clause-cut: 30 words, " - ", 30 words: pieces 11.072 (the hyphen counts) and 10.4155, whole 23.335.
# cut-prefers-spaced-hyphen-to-comma: 26 "," 11 " -" 23 ".": cut after "," = 9.542 | 13.39 (longer 13.39); cut after the hyphen
# = 14.047 | 8.8855 (longer 14.047); within 1 s (0.657), so the hyphen wins as a strong mark.
# short-outside-cut-is-not-admissible: "Ano," leaves a 1-word piece, so the cut inside the quote is the only one.
# cut-avoids-open-quote: 12 words "," then a quote of 25 "," 25 words: the only cut with no quote open is the first.
# The 50-word rest is cut at the comma inside the quote, as the only cut left.
PROBES = {  # id: (book text, max_s, [(text, boundary, kind, open_before, open_after, over_cap)])
    "quote-opens-paragraph": (
        'Přišel.\n\n"Ahoj. Jak se máš?" zeptal se.', 17.0,
        [U("Přišel.", "para_end", "sentence"), U('"Ahoj. Jak se máš?" zeptal se.', "chapter_end", "joined")]),
    "quote-closes-paragraph": (
        'Řekl: "Jdu pryč. Nevrátím se."\n\nOdešel.', 17.0,
        [U('Řekl: "Jdu pryč. Nevrátím se."', "para_end", "joined"), U("Odešel.", "chapter_end", "sentence")]),
    "lowercase-attribution": (
        '"Dobrý den," řekl Petr. Pak odešel.', 17.0,
        [U('"Dobrý den," řekl Petr.', "mid", "sentence"), U("Pak odešel.", "chapter_end", "sentence")]),
    "capitalised-narration-stays-two": (
        '"Pojď sem!" Petr se ani nehnul.', 17.0,
        [U('"Pojď sem!"', "mid", "sentence"), U("Petr se ani nehnul.", "chapter_end", "sentence")]),
    "multi-sentence-quote": (
        'Řekl: "Jdu domů. Přijdu zítra. Čekejte tu." Pak odešel.', 17.0,
        [U('Řekl: "Jdu domů. Přijdu zítra. Čekejte tu."', "mid", "joined"),
         U("Pak odešel.", "chapter_end", "sentence")]),
    "quote-split-by-attribution": (
        '"Jdu," řekl Petr, "ale vrátím se. Slibuji."', 17.0,
        [U('"Jdu," řekl Petr, "ale vrátím se. Slibuji."', "chapter_end", "joined")]),
    "single-inside-double": (
        "\"Řekl mi 'Jdi pryč! Nechci tě vidět.' a odešel.\" Potom přišel.", 17.0,
        [U("\"Řekl mi 'Jdi pryč! Nechci tě vidět.' a odešel.\"", "mid", "joined"),
         U("Potom přišel.", "chapter_end", "sentence")]),
    "apostrophe-straight": (
        "\"Pan d'Artagnan přišel. Pak odešel.\" Potom spal.", 17.0,
        [U("\"Pan d'Artagnan přišel. Pak odešel.\"", "mid", "joined"), U("Potom spal.", "chapter_end", "sentence")]),
    "apostrophe-curly": (
        '"Pan d’Artagnan přišel. Pak odešel." Potom spal.', 17.0,
        [U('"Pan d’Artagnan přišel. Pak odešel."', "mid", "joined"), U("Potom spal.", "chapter_end", "sentence")]),
    "two-quotes-one-paragraph": (
        '"Ano. Jistě." řekl. "Ne. Nikdy." dodal.', 17.0,
        [U('"Ano. Jistě." řekl.', "mid", "joined"), U('"Ne. Nikdy." dodal.', "chapter_end", "joined")]),
    "question-exclamation-before-closer": (
        '"Co?! Opravdu?!" Pak odešel.', 17.0,
        [U('"Co?! Opravdu?!"', "mid", "joined"), U("Pak odešel.", "chapter_end", "sentence")]),
    "ellipsis-in-quote": (
        '"Já nevím… Možná zítra…" řekl. Pak odešel.', 17.0,
        [U('"Já nevím… Možná zítra…" řekl.', "mid", "joined"), U("Pak odešel.", "chapter_end", "sentence")]),
    "quote-in-parentheses": (
        'Odešel (řekl "Jdu pryč. Nevrátím se.") a nevrátil se.', 17.0,
        [U('Odešel (řekl "Jdu pryč. Nevrátím se.") a nevrátil se.', "chapter_end", "joined")]),
    "quote-cut-two-and-two": (
        QUOTE4.format(o='"', c='"'), 5.5,
        [U('"Věta číslo jedna je tady. Věta číslo dvě je tady.', "mid", "joined", 0, 1),
         U('Věta číslo tři je tady. Věta číslo čtyři je tady." dodal tiše.', "chapter_end", "joined", 1, 0)]),
    "quote-cut-into-singletons": (
        QUOTE4.format(o='"', c='"'), 3.0,
        [U('"Věta číslo jedna je tady.', "mid", "sentence", 0, 1),
         U("Věta číslo dvě je tady.", "mid", "sentence", 1, 1),
         U("Věta číslo tři je tady.", "mid", "sentence", 1, 1),
         U('Věta číslo čtyři je tady." dodal tiše.', "chapter_end", "sentence", 1, 0, True)]),
    "unclosed-ascii-quote-next-paragraph": (
        '"Mluvil dlouho.\n\nPak ztichl.', 17.0,
        [U('"Mluvil dlouho.', "para_end", "sentence", 0, 1), U("Pak ztichl.", "chapter_end", "sentence")]),
    "english-curly": (
        "Řekl: “Jdi pryč. Nevracej se.” Pak odešel.", 17.0,
        [U("Řekl: “Jdi pryč. Nevracej se.”", "mid", "joined"), U("Pak odešel.", "chapter_end", "sentence")]),
    "czech-low-high": (
        "Řekl: „Jdi pryč. Nevracej se.“ Pak odešel.", 17.0,
        [U("Řekl: „Jdi pryč. Nevracej se.“", "mid", "joined"), U("Pak odešel.", "chapter_end", "sentence")]),
    "heading-quote-does-not-reach-next-chapter": (
        "# „Úvod\n\n# Konec", 17.0,
        [U("„Úvod", "chapter_end", "sentence", 0, 1), U("Konec", "chapter_end", "sentence")]),
    "cut-prefers-semicolon": (
        cap(words(26)) + ", " + words(12) + "; " + words(23) + ".", 17.0,
        [U(cap(words(26)) + ", " + words(12) + ";", "mid", "split"), U(words(23) + ".", "chapter_end", "split")]),
    "cut-prefers-minimiser": (
        cap(words(26)) + ", " + words(13) + "; " + words(21) + ".", 17.0,
        [U(cap(words(26)) + ",", "mid", "split"), U(words(13) + "; " + words(21) + ".", "chapter_end", "split")]),
    "cut-avoids-open-quote": (
        cap(words(12)) + ", „" + words(25) + ", " + words(25) + ".“", 17.0,
        [U(cap(words(12)) + ",", "mid", "split"), U("„" + words(25) + ",", "mid", "split", 0, 1),
         U(words(25) + ".“", "chapter_end", "split", 1, 0)]),
    "short-outside-cut-is-not-admissible": (
        "Ano, „" + words(30) + ", " + words(30) + ".“", 17.0,
        [U("Ano, „" + words(30) + ",", "mid", "split", 0, 1), U(words(30) + ".“", "chapter_end", "split", 1, 0)]),
    "spaced-hyphen-is-the-only-clause-cut": (
        cap(words(30)) + " - " + words(30) + ".", 17.0,
        [U(cap(words(30)) + " -", "mid", "split"), U(words(30) + ".", "chapter_end", "split")]),
    "cut-prefers-spaced-hyphen-to-comma": (
        cap(words(26)) + ", " + words(11) + " - " + words(23) + ".", 17.0,
        [U(cap(words(26)) + ", " + words(11) + " -", "mid", "split"), U(words(23) + ".", "chapter_end", "split")]),
    "span-comma-never-cut": (
        cap(words(19)) + " <en>a b c, d e f</en> " + words(19) + ", " + words(5) + ".", 17.0,
        [U(cap(words(19)) + " <en>a b c, d e f</en> " + words(19) + ",", "mid", "split"),
         U(words(5) + ".", "chapter_end", "split")]),
    "span-only-comma-over-cap": (
        cap(words(40)) + " <en>a b c, d e f</en> " + words(40) + ".", 17.0,
        [U(cap(words(40)) + " <en>a b c, d e f</en> " + words(40) + ".", "chapter_end", "sentence", 0, 0, True)]),
}

PAUSES = {"mid": "sentence", "para_end": "paragraph", "chapter_end": "chapter"}
_CLAUSE_CUT = re.compile(r"(?:(?<=[,;:—–…])|(?<=\s-))\s+(?=\S)")
_SPAN = re.compile(r"<(cs|sk|en)>.*?</\1>", re.DOTALL)


def has_admissible_cut(text):
    spans = list(_SPAN.finditer(text))
    for m in _CLAUSE_CUT.finditer(text):
        if any(s.start() < m.start() < s.end() for s in spans):
            continue
        if len(text[:m.end()].split()) >= 3 and len(text[m.end():].split()) >= 3:
            return True
    return False


def assert_properties(sentences, units, max_s):
    snapshot = list(sentences)
    assert build_units(sentences, max_s) == units
    assert sentences == snapshot
    n = len(sentences)
    if n == 0:
        assert units == []
        return
    assert units and units[0].first == 0 and units[-1].last == n - 1 and units[-1].pause != "clause"
    for k, u in enumerate(units):
        group = sentences[u.first:u.last + 1]
        assert 0 <= u.first <= u.last < n
        assert u.kind in ("sentence", "joined", "split")
        assert (u.first == u.last) == (u.kind != "joined")
        assert len({(s.chapter, s.paragraph) for s in group}) == 1
        assert (u.chapter, u.paragraph, u.heading) == (group[0].chapter, group[0].paragraph, group[0].heading)
        nxt = units[k + 1] if k + 1 < len(units) else None
        if nxt is not None and nxt.first == u.last:
            assert u.pause == "clause" and u.kind == "split" and nxt.kind == "split" and nxt.last == u.last
            assert u.boundary == "mid"
        else:
            if nxt is not None:
                assert nxt.first == u.last + 1
            assert u.boundary == sentences[u.last].boundary
            assert u.pause == PAUSES[u.boundary]
        if u.kind == "split":
            assert u.pause == "clause" or units[k - 1].pause == "clause"
        else:
            assert u.text == " ".join(s.text for s in group)
        assert u.text == u.text.strip()
        if u.over_cap:
            assert u.kind != "joined" and estimate_s(u.text) > max_s and not has_admissible_cut(u.text)
        else:
            assert estimate_s(u.text) <= max_s
        assert u.open_after == quote_depth(u.text, u.open_before)
        if k == 0 or (sentences[units[k - 1].last].chapter, sentences[units[k - 1].last].paragraph) != (u.chapter, u.paragraph):
            assert u.open_before == 0
        else:
            assert u.open_before == units[k - 1].open_after
    assert " ".join(" ".join(u.text for u in units).split()) == " ".join(" ".join(s.text for s in sentences).split())


@pytest.mark.parametrize("key", list(TABLE))
def test_table_row(key):
    check_row(build_units, key)


@pytest.mark.parametrize("key", list(TABLE))
def test_table_input_splits_as_assumed(key):
    text, _, expected = TABLE[key]
    sentences = segment(text)
    assert len(sentences) == SENTENCE_COUNTS[key]
    assert " ".join(s.text for s in sentences) == " ".join(t for t, _ in expected)


@pytest.mark.parametrize("key", list(TABLE))
def test_table_properties(key):
    text, max_s, _ = TABLE[key]
    sentences = segment(text)
    assert_properties(sentences, build_units(sentences, max_s), max_s)


def test_table_has_28_rows():
    assert len(TABLE) == 28


def test_identity_builder_fails_at_least_12_table_rows():
    failing = []
    for key in TABLE:
        try:
            check_row(identity_builder, key)
        except AssertionError:
            failing.append(key)
    assert len(failing) >= 12, failing


@pytest.mark.parametrize("key", list(PROBES))
def test_probe(key):
    text, max_s, expected = PROBES[key]
    sentences = segment(text)
    units = build_units(sentences, max_s)
    got = [(u.text, u.boundary, u.kind, u.open_before, u.open_after, u.over_cap) for u in units]
    assert got == expected
    assert_properties(sentences, units, max_s)


def test_at_least_ten_ascii_probes():
    ascii_probes = [k for k, (text, _, _) in PROBES.items() if '"' in text or "'" in text]
    assert len(ascii_probes) >= 10


def test_table_rows_never_over_cap_except_the_hopeless_one():
    for key, (text, max_s, _) in TABLE.items():
        flags = [u.over_cap for u in build_units(segment(text), max_s)]
        assert any(flags) == (key == "only-comma-after-first-word"), key


def test_heading_flags_chapters_and_paragraphs():
    units = build_units(segment(TABLE["labels-and-heading"][0]))
    assert [u.heading for u in units] == [True, False, False, False]
    assert [u.chapter for u in units] == [0, 0, 0, 0]
    assert [u.paragraph for u in units] == [0, 1, 1, 2]
    assert [(u.first, u.last) for u in units] == [(0, 0), (1, 1), (2, 2), (3, 3)]
    assert [u.pause for u in units] == ["paragraph", "sentence", "paragraph", "chapter"]


def test_long_sentence_pieces_carry_clause_pauses():
    units = build_units(segment(LONG), 17.0)
    assert [(u.kind, u.pause, u.first, u.last, u.over_cap) for u in units] == [
        ("split", "clause", 0, 0, False), ("split", "clause", 0, 0, False), ("split", "chapter", 0, 0, False)]


def test_joined_unit_indices_and_pause():
    units = build_units(segment("„Jdu domů. Přijdu zítra.“ Odešel."))
    assert [(u.kind, u.first, u.last, u.pause) for u in units] == [
        ("joined", 0, 1, "sentence"), ("sentence", 2, 2, "chapter")]


def test_chapter_change_with_the_same_paragraph_number_is_not_joined():
    sentences = [Sentence("„Ahoj.", "chapter_end", 0, 0), Sentence("Ne.“", "chapter_end", 1, 0)]
    units = build_units(sentences)
    assert [u.text for u in units] == ["„Ahoj.", "Ne.“"]
    assert [(u.open_before, u.open_after) for u in units] == [(0, 1), (0, 0)]


def test_paragraph_change_resets_open_before():
    sentences = [Sentence("„Ahoj.", "para_end", 0, 0), Sentence("Ne.“", "chapter_end", 0, 1)]
    assert [u.open_before for u in build_units(sentences)] == [0, 0]


def test_empty_input():
    assert build_units([]) == []


def test_default_max_s_is_17():
    sentences = segment(LONG)
    assert build_units(sentences) == build_units(sentences, 17.0)
    assert len(build_units(sentences)) == 3
    assert len(build_units(sentences, 40.0)) == 1


def test_injected_estimate_decides_joins_and_cuts():
    sentences = segment("„Ano. Ne.“ řekl.")
    assert [u.text for u in build_units(sentences)] == ["„Ano. Ne.“ řekl."]
    heavy = build_units(sentences, 17.0, estimate=lambda t: 100.0)
    assert [(u.text, u.over_cap) for u in heavy] == [("„Ano.", True), ("Ne.“ řekl.", True)]
    light = build_units(segment(LONG), 17.0, estimate=lambda t: 0.0)
    assert [(u.text, u.over_cap) for u in light] == [(LONG, False)]
    by_words = build_units(segment(LONG), 40.0, estimate=lambda t: float(len(t.split())))
    assert [len(u.text.split()) for u in by_words] == [30, 30, 27]


def test_unit_is_a_frozen_dataclass_with_defaults():
    u = Unit("Ahoj.", "mid", "sentence", 0, 0, 0, 0, "sentence")
    assert (u.heading, u.open_before, u.open_after, u.over_cap) == (False, 0, 0, False)
    assert [f.name for f in dataclasses.fields(Unit)] == [
        "text", "boundary", "pause", "chapter", "paragraph", "first", "last", "kind",
        "heading", "open_before", "open_after", "over_cap"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        u.text = "Ne."


VOCAB = ["dům", "les", "řeka", "starý", "nový", "šel", "viděl", "slyšel", "pán", "paní", "cesta", "hora", "oheň", "voda",
         "kámen", "ráno", "večer", "tiše", "rychle", "pomalu", "a", "ale", "protože", "když", "který", "jeho", "ten", "pes"]
STYLES = [("„", "“"), ('"', '"'), ("“", "”"), ("»", "«"), ("‚", "‘"), ("(", ")"), ("[", "]")]


def _sentence(rng):
    n = rng.choice([rng.randint(1, 12)] * 11 + [rng.randint(13, 35)] * 6 + [rng.randint(36, 90)] * 3)
    out = []
    for i in range(n):
        w = rng.choice(VOCAB)
        r = rng.random()
        if i + 1 < n and r < 0.07:
            w += ","
        elif i + 1 < n and r < 0.09:
            w += rng.choice([";", ":", "…"])
        elif i + 1 < n and r < 0.11:
            out.append(w)
            w = rng.choice(["—", "–"])
        elif r > 0.99:
            w = "<en>The Hobbit, Part One.</en>"
        out.append(w)
    text = " ".join(out)
    if not text[0].isalpha():
        text = "Pan " + text
    return text[0].upper() + text[1:] + rng.choice([".", ".", ".", "!", "?", "?!", "…"])


def _paragraph(rng):
    sents = [_sentence(rng) for _ in range(rng.randint(6, 14))]
    k = len(sents)
    mode = rng.random()
    if mode < 0.6:
        o, c = rng.choice(STYLES)
        i = rng.randrange(k)
        j = min(k - 1, i + rng.randrange(0, 4))
        sents[i] = o + sents[i]
        if rng.random() < 0.85:
            sents[j] += c + rng.choice(["", "", " řekl Petr.", " dodala tiše."])
        if rng.random() < 0.3:
            words_ = sents[i].split(" ")
            if len(words_) >= 3:
                words_[1], words_[2] = "‚" + words_[1], words_[2] + "‘"
                sents[i] = " ".join(words_)
        if rng.random() < 0.15:
            sents[rng.randrange(k)] += rng.choice(["“", "”", ")", "'"])
    elif mode < 0.7:
        sents = ["— " + s for s in sents]
    elif mode < 0.75:
        i = rng.randrange(k)
        sents[i] = "," + sents[i]
        sents[min(k - 1, i + 1)] += "'"
    return " ".join(sents)


@lru_cache(maxsize=1)
def corpus():
    rng = random.Random(7)
    blocks = []
    for chapter in range(3):
        blocks.append(f"# Kapitola {chapter + 1}")
        blocks += [_paragraph(rng) for _ in range(20)]
    return segment("\n\n".join(blocks))


def test_generated_corpus_is_large_enough():
    sentences = corpus()
    assert len(sentences) >= 400
    assert len({(s.chapter, s.paragraph) for s in sentences}) >= 40
    assert {s.chapter for s in sentences} == {0, 1, 2}
    assert max(len(s.text.split()) for s in sentences) >= 60
    assert sum(1 for s in sentences if quote_depth(s.text) > 0) >= 20


@pytest.mark.parametrize("max_s", [4.0, 8.0, 17.0, 40.0])
def test_generated_corpus_properties(max_s):
    sentences = corpus()
    units = build_units(sentences, max_s)
    assert_properties(sentences, units, max_s)
    if max_s == 17.0:
        assert any(u.kind == "joined" for u in units)
        assert any(u.kind == "split" for u in units)
        assert any(u.open_after > 0 and u.pause == "sentence" and u.last + 1 < len(sentences) for u in units)


QUOTE_DEPTH = {
    "empty": ("", 0, 0),
    "ascii-pair": ('"Ahoj"', 0, 0),
    "ascii-open": ('"Ahoj', 0, 1),
    "stray-closer": ('Ahoj"', 0, 0),
    "stray-closer-then-opener": ('Ahoj" "Ne', 0, 1),
    "ascii-closer-after-stop": ('Řekl: "Jdu domů."', 0, 0),
    "apostrophe-straight": ("d'Artagnan", 0, 0),
    "apostrophe-inside-open-quote": ('"Pan d\'Artagnan', 0, 1),
    "apostrophe-curly": ("d’Artagnan", 0, 0),
    "span-blanked": ("<en>„hi</en> Ahoj", 0, 0),
    "span-blanked-inside-quote": ('„Ahoj <cs>"</cs> světe', 0, 1),
    "carry-closes": ("Ahoj.“", 1, 0),
    "carry-stays": ("Dál.", 2, 2),
    "carry-adds": ("„Ahoj", 1, 2),
    "carry-floor": ("Dál.“", 0, 0),
    "carry-closes-two": ("Dál.“ Ne.“", 2, 0),
    "brackets-pair": ("(abc)", 0, 0),
    "brackets-open": ("(abc", 0, 1),
    "brackets-mixed": ("[a (b] c)", 0, 0),
    "brackets-stray": ("abc)", 0, 0),
    "czech-pair": ("„Ahoj“", 0, 0),
    "czech-nested": ("„Řekl mi ‚Jdi pryč‘ a odešel“", 0, 0),
    "czech-nested-open": ("„Řekl mi ‚Jdi pryč", 0, 2),
    "guillemets": ("»Ano«", 0, 0),
    "guillemets-open": ("»Ano", 0, 1),
    "english-pair": ("“Ano”", 0, 0),
    "after-dash": ('—"Ano"', 0, 0),
    "after-dash-open": ('—"Ano', 0, 1),
    "paren-then-quote": ('("Ano")', 0, 0),
    "slovak-comma-quote": (",Ahoj!'", 0, 0),
    "slovak-comma-quote-open": (",Ahoj!", 0, 1),
    "comma-after-word-is-not-a-quote": ("Ahoj, Petře", 0, 0),
}


@pytest.mark.parametrize("key", list(QUOTE_DEPTH))
def test_quote_depth(key):
    text, depth, expected = QUOTE_DEPTH[key]
    assert quote_depth(text, depth) == expected


def test_quote_depth_default_start_is_zero():
    assert quote_depth('„Ahoj') == quote_depth('„Ahoj', 0) == 1


# a = -0.70, b = 0.0765 per letter, c = 0.297 per [,;:—–] or spaced hyphen, e = 0.788 per [.!?…] (a run of dots once)
NEW = {"a": -0.70, "b": 0.0765, "c": 0.297, "e": 0.788}
ESTIMATES = {
    "empty": ("", {}, -0.70),
    "letters-and-stop": ("Ahoj.", {}, -0.70 + 4 * 0.0765 + 0.788),  # 0.394
    "every-mark-counts": ("a, b; c: d — e – f.", {}, -0.70 + 6 * 0.0765 + 5 * 0.297 + 0.788),  # 2.032
    "three-dots-count-once": ("Počkej...", NEW, -0.70 + 6 * 0.0765 + 0.788),
    "two-dots-count-once": ("Počkej..", NEW, -0.70 + 6 * 0.0765 + 0.788),
    "six-dots-count-once": ("......", NEW, -0.70 + 0.788),
    "spaced-hyphen-counts-as-a-clause-mark": ("a - b.", NEW, -0.70 + 2 * 0.0765 + 0.297 + 0.788),
    "hyphen-inside-a-word-is-not-a-mark": ("černo-bílý", NEW, -0.70 + 9 * 0.0765),
    "leading-hyphen-is-not-a-mark": ("- a", NEW, -0.70 + 0.0765),
    "hyphen-glued-on-one-side-is-not-a-mark": ("a -b- c", NEW, -0.70 + 3 * 0.0765),
    "one-ellipsis-char-counts-once": ("Počkej…", {}, -0.70 + 6 * 0.0765 + 0.788),  # 0.547
    "question-exclamation": ("Co?!", {}, -0.70 + 2 * 0.0765 + 2 * 0.788),  # 1.029
    "digits-and-quotes-are-not-letters": ("„A1B2“ 3", {}, -0.70 + 2 * 0.0765),  # -0.547
    "custom-coefficients": ("abc, d.", {"a": 1.0, "b": 2.0, "c": 3.0, "e": 5.0}, 1.0 + 4 * 2.0 + 3.0 + 5.0),  # 17
}


@pytest.mark.parametrize("key", list(ESTIMATES))
def test_estimate_s(key):
    text, kwargs, expected = ESTIMATES[key]
    assert estimate_s(text, **kwargs) == pytest.approx(expected, abs=1e-9)


def test_estimate_s_hand_values():
    assert estimate_s("Ahoj.") == pytest.approx(0.394, abs=1e-9)
    assert estimate_s("a, b; c: d — e – f.") == pytest.approx(2.032, abs=1e-9)
    assert estimate_s("Počkej...") == pytest.approx(0.547, abs=1e-9)
    assert estimate_s("a - b.") == pytest.approx(0.538, abs=1e-9)
    assert estimate_s(LONG) == pytest.approx(33.9595, abs=1e-9)
    assert estimate_s(ONLY_COMMA) == pytest.approx(34.4275, abs=1e-9)


@pytest.mark.parametrize("key", [k for k, (t, _, _) in QUOTE_DEPTH.items() if not any(c in t for c in "<()[]")])
def test_quote_depth_agrees_with_quote_tokens(key):
    text, depth, expected = QUOTE_DEPTH[key]
    for t in CzechSlovakPhonemizer("cs").tokenize(text):
        if t.symbol == "❝":
            depth += 1
        elif t.symbol == "❞":
            depth = max(0, depth - 1)
    assert depth == expected
