import pytest
from src.text.normalizer import TextNormalizer


@pytest.fixture
def cs():
    return TextNormalizer(language="cs")


@pytest.fixture
def sk():
    return TextNormalizer(language="sk")


def test_no_digits_or_symbols_survive(cs):
    out = cs.normalize("Dne 1.1.2024 v 14:30 zaplatil 100 Kč, tj. cca 4 € (20 %).")
    assert not any(c.isdigit() for c in out) and "%" not in out and "€" not in out


def test_numbers_cs(cs):
    assert cs.normalize("Mám 5 jablek.") == "Mám pět jablek."
    assert cs.normalize("Je mi 25 let.") == "Je mi dvacet pět let."          # Czech: separate words


def test_numbers_sk(sk):
    assert sk.normalize("Mám 25 rokov.") == "Mám dvadsaťpäť rokov."          # Slovak: one word


def test_case_from_context_cs(cs):
    assert cs.normalize("Šel s 5 přáteli.") == "Šel s pěti přáteli."


@pytest.mark.parametrize("text,expected", [
    ("1. ledna 2024", "Prvního ledna dva tisíce dvacet čtyři"),
    ("15.3.2024", "Patnáctého března dva tisíce dvacet čtyři"),
])
def test_dates_cs(cs, text, expected):
    assert cs.normalize(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("Karel IV.", "Karel čtvrtý."), ("XXI. století", "Dvacáté první století"), ("III. díl", "Třetí díl"),
])
def test_roman_numerals_cs(cs, text, expected):
    assert cs.normalize(text) == expected


def test_abbreviations(cs, sk):
    assert cs.normalize("např. toto") == "například toto"
    assert sk.normalize("atď.") == "a tak ďalej."


def test_dashes_and_ellipsis(cs):
    assert cs.normalize("A pak – nic...") == "A pak — nic…"


def test_punctuation_and_empty(cs):
    assert cs.normalize("Máš 5 jablek?").endswith("?")
    assert cs.normalize("") == ""
    assert cs.normalize("Toto je věta bez čísel.") == "Toto je věta bez čísel."


# ---- added with Prompt 2.2 ------------------------------------------------------------------
import logging

from src.text.phonemizer import CzechSlovakPhonemizer

EXAMPLES = {  # id: (language, book_config, input, expected output)
    "english-phrases": ("cs", {"english": ["Harry", "Harry Potter"]},
                        "Harry Potter a Harry, ale ne Harrymu ani harry.",
                        "<en>Harry Potter</en> a <en>Harry</en>, ale ne Harrymu ani harry."),
    "hand-tagged-span": ("cs", None, "Četl <en>The Hobbit</en> 2 roky.", "Četl <en>The Hobbit</en> dva roky."),
    "sk-personal-masculine": ("sk", None, "Prišli 2 muži.", "Prišli dvaja muži."),
    "unit": ("cs", None, "Ujel 5 km.", "Ujel pět kilometrů."),
    "time-cs": ("cs", None, "Vlak jede v 14:30.", "Vlak jede ve čtrnáct třicet."),
    "time-sk": ("sk", None, "Stretneme sa o 14:30.", "Stretneme sa o štrnástej tridsať."),
    "currency-cs": ("cs", None, "Zaplatil 4,50 €.", "Zaplatil čtyři eura padesát centů."),
    "currency-sk": ("sk", None, "Zaplatil 2 €.", "Zaplatil dve eurá."),
    "negative-amount": ("cs", None, "Dlužil −4,50 € a -0,50 €.", "Dlužil mínus čtyři eura padesát centů a mínus padesát centů."),
    "num2words-variant": ("cs", {"num2words": {"construction": "inverted"}}, "Je mi 25 let.",
                          "Je mi pětadvacet let."),
    "period-ends-sentence": ("sk", None, "Kúpil chlieb, mlieko atď. Potom odišiel.",
                             "Kúpil chlieb, mlieko a tak ďalej. Potom odišiel."),
    "period-inside-sentence": ("cs", None, "Navštívil např. Prahu a Brno.", "Navštívil například Prahu a Brno."),
    "capital-at-sentence-start": ("cs", None, "5 lidí přišlo. 3 lidé odešli.", "Pět lidí přišlo. Tři lidé odešli."),
    "locative-abbreviation": ("sk", None, "Na str. 45 sa píše o tom.", "Na strane štyridsaťpäť sa píše o tom."),
    "slash-between-words": ("cs", None, "Přijde on a/nebo ona, on/ona.", "Přijde on a nebo ona, on nebo ona."),
    "legal-reference": ("cs", None, "Podle § 5 odst. 2 platí.", "Podle paragrafu pět odstavce dva platí."),
    "math-signs": ("sk", None, "Platí 3 × 4 = 12.", "Platí tri krát štyri rovná sa dvanásť."),
    "square-metres": ("cs", None, "Byt má 80 m² a sklep 2 m2.",
                      "Byt má osmdesát metrů čtverečních a sklep dva metry čtvereční."),
    "dot-thousands": ("cs", None, "Stálo to 10.000 Kč.", "Stálo to deset tisíc korun."),
    "time-range": ("cs", None, "Otevřeno 10:00–12:00.", "Otevřeno deset hodin až dvanáct hodin."),
    "label-cs": ("cs", None, "Kapitola 1 začíná.", "Kapitola jedna začíná."),
    "same-form-in-every-case": ("cs", None, "Pak začalo XXI. století.", "Pak začalo dvacáté první století."),
    "tagger-gender-fix": ("sk", None, "Potom vyšiel 2. diel.", "Potom vyšiel druhý diel."),
    "ordinal-range": ("cs", None, "Od 1.–5. ledna.", "Od prvního až pátého ledna."),
    "shared-noun": ("cs", None, "Přelom XIX. a XX. století.", "Přelom devatenáctého a dvacátého století."),
    "noun-after-tisíc": ("cs", None, "S 1 000 Kč vyrazil.", "S tisícem korun vyrazil."),
    "grouped-digits": ("cs", None, "Po 1 000 letech.", "Po tisíci letech."),
}
LOGGED = {  # id: (language, input, expected output, part of the WARNING)
    "fallback": ("cs", "Zbyl jen 1.", "Zbyl jen jeden.", "nominative masculine inanimate"),
    "doubtful-tag": ("cs", "Vyšly 2. díly.", "Vyšly druhé díly.", "plural noun"),
}
HEADINGS = ("# Kapitola 5\n\nPetr koupil 5\njablek.\n\n\n# 2. kapitola\n\nBylo 8:00.\n",
            "# Kapitola pět\n\nPetr koupil pět\njablek.\n\n\n# Druhá kapitola\n\nBylo osm hodin.\n")
TEST2_INPUTS = [  # every input of the Test 2 block above
    ("cs", "Dne 1.1.2024 v 14:30 zaplatil 100 Kč, tj. cca 4 € (20 %)."), ("cs", "Mám 5 jablek."),
    ("cs", "Je mi 25 let."), ("sk", "Mám 25 rokov."), ("cs", "Šel s 5 přáteli."), ("cs", "1. ledna 2024"),
    ("cs", "15.3.2024"), ("cs", "Karel IV."), ("cs", "XXI. století"), ("cs", "III. díl"), ("cs", "např. toto"),
    ("sk", "atď."), ("cs", "A pak – nic..."), ("cs", "Máš 5 jablek?"), ("cs", ""), ("cs", "Toto je věta bez čísel."),
]
G2P = {"cs": CzechSlovakPhonemizer("cs"), "sk": CzechSlovakPhonemizer("sk")}


@pytest.mark.parametrize("language,config,text,expected", EXAMPLES.values(), ids=list(EXAMPLES))
def test_examples(language, config, text, expected):
    assert TextNormalizer(language, config).normalize(text) == expected


@pytest.mark.parametrize("language,text,expected,warning", LOGGED.values(), ids=list(LOGGED))
def test_reading_is_logged_for_review(caplog, language, text, expected, warning):
    with caplog.at_level(logging.WARNING, logger="src.text.normalizer"):
        assert TextNormalizer(language).normalize(text) == expected
    records = [r for r in caplog.records if r.name == "src.text.normalizer"]
    assert [r.levelno for r in records] == [logging.WARNING]
    assert warning in records[0].getMessage()


def test_line_structure_and_headings(cs):
    text, expected = HEADINGS
    assert cs.normalize(text) == expected


@pytest.mark.parametrize("text", ["Četl <en>Apollo 13</en>.", "Firma <en>R&D</en>."])
def test_digit_or_symbol_in_span_raises(cs, text):
    with pytest.raises(ValueError):
        cs.normalize(text)


def test_unknown_num2words_variant_raises():
    with pytest.raises(ValueError):
        TextNormalizer("sk", {"num2words": {"inverted": True}})  # a Czech-only keyword


# inputs that raise have no output to tokenize; the heading example keeps "# ", which the G2P rejects
@pytest.mark.parametrize("language,config,text", [(lang, None, text) for lang, text in TEST2_INPUTS]
                         + [example[:3] for example in EXAMPLES.values()]
                         + [(language, None, text) for language, text, _, _ in LOGGED.values()])
def test_output_is_tokenizable(language, config, text):
    G2P[language].tokenize(TextNormalizer(language, config).normalize(text))
