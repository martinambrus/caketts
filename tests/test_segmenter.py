import pytest
from src.data.dataset import BOUNDARY_IDS
from src.text.normalizer import TextNormalizer
from src.text.phonemizer import CzechSlovakPhonemizer
from src.text.segmenter import Sentence, segment

SPLITS = {  # id: (paragraph, sentences)
    "period-uppercase": ("Přišel domů. Šel spát.", ["Přišel domů.", "Šel spát."]),
    "two-short": ("Přišel. Odešel.", ["Přišel.", "Odešel."]),
    "three-sentences": ("Přišel. Odešel. Zůstal.", ["Přišel.", "Odešel.", "Zůstal."]),
    "question": ("Kdo přišel? Petr.", ["Kdo přišel?", "Petr."]),
    "exclamation": ("Pozor! Jde vlak.", ["Pozor!", "Jde vlak."]),
    "ellipsis-char-uppercase": ("Počkej… Pak přijde.", ["Počkej…", "Pak přijde."]),
    "ellipsis-dots-uppercase": ("Počkej... Pak přijde.", ["Počkej...", "Pak přijde."]),
    "mark-runs": ("Cože?! Ty jsi tady!!! Nikdo.", ["Cože?!", "Ty jsi tady!!!", "Nikdo."]),
    "no-final-mark": ("Přišel. Odešel", ["Přišel.", "Odešel"]),
    "one-sentence": ("Přišel domů.", ["Přišel domů."]),
    "slovak": ("Odišiel. Potom prišiel.", ["Odišiel.", "Potom prišiel."]),
    "several-spaces": ("Odešel.  Potom přišel.", ["Odešel.", "Potom přišel."]),
    "non-breaking-space": ("Odešel.\u00a0Potom přišel.", ["Odešel.", "Potom přišel."]),
    "period-lowercase": ("Přišel. odešel.", ["Přišel. odešel."]),
    "question-lowercase": ("Kam jdeš? domů.", ["Kam jdeš? domů."]),
    "ellipsis-char-lowercase": ("Já to najdu… já to najdu, pane.", ["Já to najdu… já to najdu, pane."]),
    "ellipsis-dots-lowercase": ("A pak... nic.", ["A pak... nic."]),
    "ellipsis-before-uppercase-ends": ("A pak… Nic.", ["A pak…", "Nic."]),
    "exclamation-quote-lowercase": ("„Jdi pryč!“ křikl.", ["„Jdi pryč!“ křikl."]),
    "question-quote-lowercase": ("„Kam jdeš?“ zeptal se.", ["„Kam jdeš?“ zeptal se."]),
    "comma-quote": ("„Jdu,“ řekl.", ["„Jdu,“ řekl."]),
    "period-quote-lowercase": ("„Jdu domů.“ řekl.", ["„Jdu domů.“ řekl."]),
    "no-space-after-period": ("Přišel.Odešel.", ["Přišel.Odešel."]),
    "no-space-after-question": ("Kdo?Petr.", ["Kdo?Petr."]),
    "no-space-after-closing-quote": ("„Jdu.“Odešel.", ["„Jdu.“Odešel."]),
    "quote-after-colon": ("Řekl: „Jdu.“ Odešel.", ["Řekl: „Jdu.“", "Odešel."]),
    "closing-quote-stays-opening-quote-moves": (
        "„Kam jdeš?“ zeptal se. „Domů.“ Odešel.", ["„Kam jdeš?“ zeptal se.", "„Domů.“", "Odešel."]),
    "opening-quote-next": ("Odešel. „Jdu domů,“ řekl.", ["Odešel.", "„Jdu domů,“ řekl."]),
    "opening-bracket-next": ("Odešel. (Potom přišel.)", ["Odešel.", "(Potom přišel.)"]),
    "closing-bracket": ("(Odešel.) Potom přišel.", ["(Odešel.)", "Potom přišel."]),
    "closing-square-bracket": ("[Odešel.] Potom přišel.", ["[Odešel.]", "Potom přišel."]),
    "bracket-then-lowercase": ("Přišel (viz výše.) a odešel.", ["Přišel (viz výše.) a odešel."]),
    "curly-double-quotes": ("“Jdu.” Odešel.", ["“Jdu.”", "Odešel."]),
    "straight-double-quotes": ('"Jdu." Odešel.', ['"Jdu."', "Odešel."]),
    "guillemets-outward": ("»Jdu.« Odešel.", ["»Jdu.«", "Odešel."]),
    "guillemets-inward": ("«Jdu.» Odešel.", ["«Jdu.»", "Odešel."]),
    "single-quotes": ("‚Jdu.‘ Odešel.", ["‚Jdu.‘", "Odešel."]),
    "ellipsis-inside-quote": ("„Počkej…“ Odešel.", ["„Počkej…“", "Odešel."]),
    "nested-quotes": ("„Řekl mi ‚Jdi pryč!‘ a odešel.“ Potom přišel.",
                      ["„Řekl mi ‚Jdi pryč!‘ a odešel.“", "Potom přišel."]),
    "old-style-comma-quote": ("Odišiel. ,Ahoj!' povedal. ,Kam ideš?' spýtal sa.",
                              ["Odišiel.", ",Ahoj!' povedal.", ",Kam ideš?' spýtal sa."]),
    "sentences-inside-quote": ("„Jdu domů. Přijdu zítra.“ Odešel.", ["„Jdu domů.", "Přijdu zítra.“", "Odešel."]),
    "dash-inside-sentence": ("— Kam jdeš? — zeptal se.", ["— Kam jdeš? — zeptal se."]),
    "dash-opens-next": ("Odešel. — Kam jdeš?", ["Odešel.", "— Kam jdeš?"]),
    "dashes-three-replies": ("— Kam jdeš? — Domů. — Proč?", ["— Kam jdeš?", "— Domů.", "— Proč?"]),
    "en-dashes": ("– Kam jdeš? – Domů.", ["– Kam jdeš?", "– Domů."]),
    "hyphens": ("- Kam jdeš? - Domů.", ["- Kam jdeš?", "- Domů."]),
    "dash-inside-then-split": ("Přišel — a odešel. Potom spal.", ["Přišel — a odešel.", "Potom spal."]),
    "initial-before-surname": ("Pozdravil V. Havel. Potom odešel.", ["Pozdravil V. Havel.", "Potom odešel."]),
    "initials-at-start": ("T. G. Masaryk přišel. Odešel.", ["T. G. Masaryk přišel.", "Odešel."]),
    "initials-two": ("J. K. Rowling napsala knihu. Potom odešla.", ["J. K. Rowling napsala knihu.", "Potom odešla."]),
    "kept-acronym-in-capitals": ("ROKU TŘI STA N. L. VLÁDL KAREL ČTVRTÝ.", ["ROKU TŘI STA N. L. VLÁDL KAREL ČTVRTÝ."]),
    "capital-word-period-ends": ("Byl to velký DOM. Potom odešel.", ["Byl to velký DOM.", "Potom odešel."]),
    "span-inner-marks-stay": ("Četl <en>Mr. Smith. Hello World.</en> a odešel.",
                              ["Četl <en>Mr. Smith. Hello World.</en> a odešel."]),
    "span-then-period": ("Četl <en>The Hobbit</en>. Potom odešel.", ["Četl <en>The Hobbit</en>.", "Potom odešel."]),
    "span-ends-with-mark": ("Řekl <en>Hello!</en> Potom odešel.", ["Řekl <en>Hello!</en> Potom odešel."]),
    "span-sk": ("Řekl <sk>Ahoj. Ako sa máš?</sk> a odišiel.", ["Řekl <sk>Ahoj. Ako sa máš?</sk> a odišiel."]),
    "span-opens-next": ("Odešel. <en>Harry</en> přišel.", ["Odešel.", "<en>Harry</en> přišel."]),
    "leading-ellipsis-in-quote": ("„…a pak odešel.“", ["„…a pak odešel.“"]),
    "leading-ellipsis-bare": ("…A pak odešel.", ["…A pak odešel."]),
    "leading-ellipsis-after-sentence": ("Odešel. „…A pak přišel.“", ["Odešel.", "„…A pak přišel.“"]),
    "leading-ellipsis-spaced-in-quote": ("„… A pak odešel.“", ["„… A pak odešel.“"]),
    "leading-ellipsis-spaced-bare": ("… A pak odešel.", ["… A pak odešel."]),
}

STRUCTURE = {  # id: (book text, [(text, boundary, chapter, paragraph)])
    "one-sentence": ("Přišel.", [("Přišel.", "chapter_end", 0, 0)]),
    "two-paragraphs": ("Přišel.\n\nOdešel.", [("Přišel.", "para_end", 0, 0), ("Odešel.", "chapter_end", 0, 1)]),
    "mid-para-end-chapter-end": ("Přišel. Odešel.\n\nKonec.", [
        ("Přišel.", "mid", 0, 0), ("Odešel.", "para_end", 0, 0), ("Konec.", "chapter_end", 0, 1)]),
    "headingless-three-paragraphs": ("Přišel.\n\nOdešel.\n\nZůstal.", [
        ("Přišel.", "para_end", 0, 0), ("Odešel.", "para_end", 0, 1), ("Zůstal.", "chapter_end", 0, 2)]),
    "heading-and-body": ("# První\n\nPřišel.", [("První", "para_end", 0, 0), ("Přišel.", "chapter_end", 0, 1)]),
    "two-chapters": ("# První\n\nPřišel.\n\n# Druhá\n\nOdešel.", [
        ("První", "para_end", 0, 0), ("Přišel.", "chapter_end", 0, 1),
        ("Druhá", "para_end", 1, 0), ("Odešel.", "chapter_end", 1, 1)]),
    "paragraph-index-restarts": ("# A\n\nPrvní.\n\nDruhá.\n\n# B\n\nTřetí.", [
        ("A", "para_end", 0, 0), ("První.", "para_end", 0, 1), ("Druhá.", "chapter_end", 0, 2),
        ("B", "para_end", 1, 0), ("Třetí.", "chapter_end", 1, 1)]),
    "text-before-first-heading": ("Úvod.\n\n# První\n\nPřišel.", [
        ("Úvod.", "chapter_end", 0, 0), ("První", "para_end", 1, 0), ("Přišel.", "chapter_end", 1, 1)]),
    "heading-without-blank-lines": ("Přišel.\n# První\nOdešel.", [
        ("Přišel.", "chapter_end", 0, 0), ("První", "para_end", 1, 0), ("Odešel.", "chapter_end", 1, 1)]),
    "headings-back-to-back": ("# A\n\n# B\n\nText.", [
        ("A", "chapter_end", 0, 0), ("B", "para_end", 1, 0), ("Text.", "chapter_end", 1, 1)]),
    "heading-alone": ("# A", [("A", "chapter_end", 0, 0)]),
    "heading-at-the-end": ("Text.\n\n# Konec", [("Text.", "chapter_end", 0, 0), ("Konec", "chapter_end", 1, 0)]),
    "heading-with-two-sentences": ("# Kapitola jedna. Návrat\n\nText.", [
        ("Kapitola jedna.", "mid", 0, 0), ("Návrat", "para_end", 0, 0), ("Text.", "chapter_end", 0, 1)]),
    "joined-lines": ("Přišel\nodešel. Potom\nšel spát.\n\nKonec.", [
        ("Přišel odešel.", "mid", 0, 0), ("Potom šel spát.", "para_end", 0, 0), ("Konec.", "chapter_end", 0, 1)]),
    "crlf": ("Přišel.\r\nOdešel.\r\n\r\nKonec.\r\n", [
        ("Přišel.", "mid", 0, 0), ("Odešel.", "para_end", 0, 0), ("Konec.", "chapter_end", 0, 1)]),
    "joined-lines-crlf": ("Přišel\r\nodešel. Potom\r\nšel spát.\r\n", [
        ("Přišel odešel.", "mid", 0, 0), ("Potom šel spát.", "chapter_end", 0, 0)]),
    "lone-carriage-returns": ("Přišel\rodešel.\r\rKonec.", [
        ("Přišel odešel.", "para_end", 0, 0), ("Konec.", "chapter_end", 0, 1)]),
    "unicode-line-separator": ("Přišel\u2028odešel.", [("Přišel odešel.", "chapter_end", 0, 0)]),
    "indented-lines-joined-by-one-space": ("Přišel\n    odešel.\n", [("Přišel odešel.", "chapter_end", 0, 0)]),
    "blank-lines-with-spaces": ("Přišel.\n \n\n\nOdešel.", [
        ("Přišel.", "para_end", 0, 0), ("Odešel.", "chapter_end", 0, 1)]),
    "leading-and-trailing-blank-lines": ("\n\nPřišel.\n\n", [("Přišel.", "chapter_end", 0, 0)]),
    "empty": ("", []),
    "whitespace-only": (" \n\t\n  ", []),
    "letterless-paragraph-stars": ("Přišel.\n\n* * *\n\nOdešel.", [
        ("Přišel.", "para_end", 0, 0), ("Odešel.", "chapter_end", 0, 1)]),
    "letterless-paragraph-dashes": ("Přišel.\n\n---\n\nOdešel.", [
        ("Přišel.", "para_end", 0, 0), ("Odešel.", "chapter_end", 0, 1)]),
    "letterless-paragraph-keeps-indices-dense": ("# A\n\n* * *\n\nText.", [
        ("A", "para_end", 0, 0), ("Text.", "chapter_end", 0, 1)]),
    # a heading starts a chapter even when it has no letter; it adds no sentence
    "empty-heading": ("Text.\n\n# \n\nMore.", [("Text.", "chapter_end", 0, 0), ("More.", "chapter_end", 1, 0)]),
    "empty-heading-first": ("# \n\nText.", [("Text.", "chapter_end", 0, 0)]),
    "heading-with-trailing-spaces": ("# První  \n\nPřišel.", [
        ("První", "para_end", 0, 0), ("Přišel.", "chapter_end", 0, 1)]),
}

PIPELINES = {  # id: (language, book text, [(text, boundary, chapter, paragraph)])
    "czech": ("cs", """# Kapitola 1

Pan V. Havel přišel domů. Vládl Karel IV. Potom odešel.

Koupil chleba, mléko atd. Potom
šel domů, protože
už byla tma. 5 lidí přišlo.

„Kam jdeš?“ zeptal se. „Domů.“ Odešel.

# Druhá kapitola

— Kam jdeš? — zeptal se. — Domů.

Počkej… pak přijde. Počkej… Pak přijde.

Četl <en>The Hobbit</en>. Potom spal.

# Konec

Přišel jsem. Odešel jsem.
""", [
        ("Kapitola jedna", "para_end", 0, 0),
        ("Pan V. Havel přišel domů.", "mid", 0, 1),
        ("Vládl Karel čtvrtý.", "mid", 0, 1),
        ("Potom odešel.", "para_end", 0, 1),
        ("Koupil chleba, mléko a tak dále.", "mid", 0, 2),
        ("Potom šel domů, protože už byla tma.", "mid", 0, 2),
        ("Pět lidí přišlo.", "para_end", 0, 2),
        ("„Kam jdeš?“ zeptal se.", "mid", 0, 3),
        ("„Domů.“", "mid", 0, 3),
        ("Odešel.", "chapter_end", 0, 3),
        ("Druhá kapitola", "para_end", 1, 0),
        ("— Kam jdeš? — zeptal se.", "mid", 1, 1),
        ("— Domů.", "para_end", 1, 1),
        ("Počkej… pak přijde.", "mid", 1, 2),
        ("Počkej…", "mid", 1, 2),
        ("Pak přijde.", "para_end", 1, 2),
        ("Četl <en>The Hobbit</en>.", "mid", 1, 3),
        ("Potom spal.", "chapter_end", 1, 3),
        ("Konec", "para_end", 2, 0),
        ("Přišel jsem.", "mid", 2, 1),
        ("Odešel jsem.", "chapter_end", 2, 1),
    ]),
    "slovak": ("sk", """# Kapitola 1

Pán V. Havel prišiel domov. Vládol Karol IV. Potom odišiel.

Kúpil chlieb, mlieko atď. Potom
išiel domov, pretože
už bola tma. 5 ľudí prišlo.

„Kam ideš?“ spýtal sa. „Domov.“ Odišiel.

# Druhá kapitola

— Kam ideš? — spýtal sa. — Domov.

Počkaj… potom príde. Počkaj… Potom príde.

Čítal <en>The Hobbit</en>. Potom spal.

# Koniec

Prišiel som. Odišiel som.
""", [
        ("Kapitola jeden", "para_end", 0, 0),
        ("Pán V. Havel prišiel domov.", "mid", 0, 1),
        ("Vládol Karol štvrtý.", "mid", 0, 1),
        ("Potom odišiel.", "para_end", 0, 1),
        ("Kúpil chlieb, mlieko a tak ďalej.", "mid", 0, 2),
        ("Potom išiel domov, pretože už bola tma.", "mid", 0, 2),
        ("Päť ľudí prišlo.", "para_end", 0, 2),
        ("„Kam ideš?“ spýtal sa.", "mid", 0, 3),
        ("„Domov.“", "mid", 0, 3),
        ("Odišiel.", "chapter_end", 0, 3),
        ("Druhá kapitola", "para_end", 1, 0),
        ("— Kam ideš? — spýtal sa.", "mid", 1, 1),
        ("— Domov.", "para_end", 1, 1),
        ("Počkaj… potom príde.", "mid", 1, 2),
        ("Počkaj…", "mid", 1, 2),
        ("Potom príde.", "para_end", 1, 2),
        ("Čítal <en>The Hobbit</en>.", "mid", 1, 3),
        ("Potom spal.", "chapter_end", 1, 3),
        ("Koniec", "para_end", 2, 0),
        ("Prišiel som.", "mid", 2, 1),
        ("Odišiel som.", "chapter_end", 2, 1),
    ]),
}

G2P = {"cs": CzechSlovakPhonemizer("cs"), "sk": CzechSlovakPhonemizer("sk")}


@pytest.mark.parametrize("paragraph,expected", SPLITS.values(), ids=list(SPLITS))
def test_sentence_splits(paragraph, expected):
    assert [s.text for s in segment(paragraph)] == expected


@pytest.mark.parametrize("text,expected", STRUCTURE.values(), ids=list(STRUCTURE))
def test_structure_and_labels(text, expected):
    assert [(s.text, s.boundary, s.chapter, s.paragraph) for s in segment(text)] == expected


def test_sentence_fields_and_order():
    assert segment("Přišel.") == [Sentence("Přišel.", "chapter_end", 0, 0)]


def test_every_label_occurs_and_is_a_known_id():
    text = "# A\n\nPřišel. Odešel.\n\nZůstal.\n\n# B\n\nKonec."
    assert {s.boundary for s in segment(text)} == set(BOUNDARY_IDS)


@pytest.mark.parametrize("text", [paragraph for paragraph, _ in SPLITS.values()] + [text for text, _ in STRUCTURE.values()],
                         ids=[*SPLITS, *(f"structure-{key}" for key in STRUCTURE)])
def test_sentences_are_clean_single_lines(text):
    for s in segment(text):
        assert len(s.text.splitlines()) == 1
        assert not s.text.startswith("# ")
        assert s.text == s.text.strip()


@pytest.mark.parametrize("language,text,expected", PIPELINES.values(), ids=list(PIPELINES))
def test_normalizer_to_segmenter_to_g2p(language, text, expected):
    sentences = segment(TextNormalizer(language).normalize(text))
    assert [(s.text, s.boundary, s.chapter, s.paragraph) for s in sentences] == expected
    for s in sentences:
        assert "# " not in s.text and "\n" not in s.text
        G2P[language].tokenize(s.text)
