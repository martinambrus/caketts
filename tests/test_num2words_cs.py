# -*- coding: utf-8 -*-
"""
Tests for num2words_cs v2. Expected values are quoted from the sources listed in
NUM2WORDS_CHANGES.md (IJP = Internetová jazyková příručka ÚJČ AV ČR, chapters 670/671/791/792 and
dictionary entries; WIKI-ČČ / WIKI-ŘČ = cs.wikipedia "České číslovky" / "Řadová číslovka";
ČRo = Český rozhlas language columns; NK-07 = Národní knihovna "Ptejte se knihovny", citing the
ÚJČ helpline; UK-hnd = Charles University handout; CHL = Chlumská 2009, quoting Sedláček and Gebauer).
Run: pytest test_num2words_cs.py
"""
import re

import pytest

try:  # standalone: the module sits next to this file
    from num2words_cs import CASES, Num2Word_CS, num2words as w
except ImportError:  # inside the TTS repository
    from src.text.num2words_cs import CASES, Num2Word_CS, num2words as w

CASE_ORDER = ["nominative", "genitive", "dative", "accusative", "instrumental", "locative"]
GENDERS = ("masculine", "feminine", "neuter")


def paradigm(n, **kw):
    return [w(n, case=c, **kw) for c in CASE_ORDER]


# ---------------------------------------------------------------- 1-4 (IJP:jeden, IJP-670, IJP-671)
def test_jeden_by_gender_and_animacy():
    assert paradigm(1) == ["jeden", "jednoho", "jednomu", "jeden", "jedním", "jednom"]
    assert w(1, case="accusative", animacy="animate") == "jednoho"
    assert paradigm(1, gender="feminine") == ["jedna", "jedné", "jedné", "jednu", "jednou", "jedné"]
    assert paradigm(1, gender="neuter") == ["jedno", "jednoho", "jednomu", "jedno", "jedním", "jednom"]


def test_dva_tri_ctyri():
    assert paradigm(2) == ["dva", "dvou", "dvěma", "dva", "dvěma", "dvou"]
    assert paradigm(2, gender="feminine") == ["dvě", "dvou", "dvěma", "dvě", "dvěma", "dvou"]
    assert paradigm(2, gender="neuter")[0] == "dvě"
    assert w(2, case="accusative", animacy="animate") == "dva"       # no animacy split for dva
    assert paradigm(3) == ["tři", "tří", "třem", "tři", "třemi", "třech"]
    assert paradigm(4, gender="feminine") == ["čtyři", "čtyř", "čtyřem", "čtyři", "čtyřmi", "čtyřech"]


# ---------------------------------------------------------------- 5-99 (IJP:pět, IJP:devět, IJP:dvacet)
def test_pet_paradigm():
    assert paradigm(5) == ["pět", "pěti", "pěti", "pět", "pěti", "pěti"]


@pytest.mark.parametrize("n,obl", [(9, "devíti"), (10, "deseti"), (11, "jedenácti"), (19, "devatenácti"),
                                   (20, "dvaceti"), (40, "čtyřiceti"), (90, "devadesáti")])
def test_oblique_stems(n, obl):
    assert w(n, case="genitive") == obl


def test_compounds_are_separate_words():
    """IJP-791: 'Vypisujeme-li číslovky slovy, píšeme každé slovo zvlášť, např. dvacet jedna'."""
    assert w(21) == "dvacet jedna"                                   # IJP-791 (abstract number)
    assert w(54) == "padesát čtyři"
    assert w(99) == "devadesát devět"


def test_three_constructions_for_x1_to_x4():
    """IJP-792: dvacet jeden žák / dvacet jedna žáků / jednadvacet žáků; 2. p. is 'běžnější'."""
    assert {w(21, gender=g) for g in GENDERS} == {"dvacet jedna"}             # type B, default
    assert {w(22, gender=g) for g in GENDERS} == {"dvacet dva"}               # WIKI-ČČ "dvacet dva žen"
    assert w(21, construction="agreement") == "dvacet jeden"                  # dvacet jeden žák
    assert w(21, construction="agreement", gender="neuter") == "dvacet jedno"
    assert w(22, construction="agreement", gender="feminine") == "dvacet dvě"  # dvacet dvě ženy
    assert w(21, construction="inverted") == "jednadvacet"
    assert w(54, construction="inverted") == "čtyřiapadesát"                 # IJP-791
    assert w(99, construction="inverted") == "devětadevadesát"              # IJP-791


def test_compound_obliques_decline_both_words():
    assert w(27, case="dative") == "dvaceti sedmi"                   # IJP-791 "k dvaceti sedmi stupňům"
    assert w(27, case="dative", construction="inverted") == "sedmadvaceti"
    assert w(25, case="dative") == "dvaceti pěti"                    # ČRo-05 "k dvaceti pěti lidem"
    assert w(52, case="instrumental") == "padesáti dvěma"            # ČRo-05 "padesáti dvěma korunami"
    assert w(23, case="genitive") == "dvaceti tří"                   # NK-07
    assert w(23, case="dative") == "dvaceti třem"                    # NK-07
    assert w(23, case="locative") == "dvaceti třech"                 # IJP-791
    assert w(22, case="dative", gender="feminine") == "dvaceti dvěma"
    assert w(22, case="dative", construction="inverted") == "dvaadvaceti"


def test_agreement_type_is_nominative_accusative_only():
    """Native review: 'dvaceti jednoho' is not acceptable; oblique cases use jednadvaceti / jedna."""
    assert w(21, construction="agreement") == "dvacet jeden"
    assert w(21, case="genitive", construction="agreement") == "jednadvaceti"
    assert w(21, case="genitive", construction="agreement", gender="feminine") == "jednadvaceti"
    assert w(201, case="genitive", construction="agreement") == "dvou set jedna"
    assert w(21_000, construction="agreement") == "dvacet jeden tisíc"
    assert w(21_000, case="genitive", construction="agreement") == "jednadvaceti tisíc"
    assert w(25_000, construction="agreement") == "dvacet pět tisíc"      # regression: not "tisíce"
    assert w(29_999, construction="agreement").startswith("dvacet devět tisíc ")


def test_native_review_decisions():
    """Decided in native review (25 Sep 2026) where the sources are silent or allow variants."""
    assert w(101, case="genitive") == "sto jedna"
    assert w(101, case="instrumental", gender="feminine") == "sto jedna"
    assert w(1001, case="dative") == "tisíci jedna"                     # confirmed: 1001 follows 101
    assert w(1001, case="genitive") == "tisíce jedna"
    assert w("21,3") == "dvacet jedna celá tři desetiny"
    assert w(10**12) == "bilion"
    assert w(21_000, to="ordinal") == "jednadvacetitisící"
    assert w(2300, case="genitive") == "dvou tisíc tří set"                  # up to two parts: all decline
    assert w(1847, case="genitive") == "tisíc osm set čtyřiceti sedmi"       # three parts: only the last


def test_x1_obliques_use_the_inverted_form():
    """No source gives 'dvaceti jedné' / 'dvaceti jednoho'; the inverted type is documented."""
    for c in ("genitive", "dative", "instrumental", "locative"):
        for g in GENDERS:
            assert w(21, case=c, gender=g) == "jednadvaceti"
    assert w(91, case="genitive") == "jednadevadesáti"


# ---------------------------------------------------------------- sto (IJP:sto, IJP-792, ÚJOP, ČRo-05)
def test_hundreds_are_nouns():
    assert paradigm(100) == ["sto", "sta", "stu", "sto", "stem", "stu"]
    assert paradigm(200) == ["dvě stě", "dvou set", "dvěma stům", "dvě stě", "dvěma sty", "dvou stech"]
    assert paradigm(300)[4] == "třemi sty"                           # IJP:tři "před třemi sty lety"
    assert paradigm(500) == ["pět set", "pěti set", "pěti stům", "pět set", "pěti sty", "pěti stech"]
    for g in GENDERS:                                                # stá/sté are ordinals
        assert w(100, gender=g) == "sto" and w(100, gender=g, case="genitive") == "sta"
    assert w(200, case="dative") == "dvěma stům"                     # ČRo-05 "k dvěma stům posluchačům"
    assert w(550, case="instrumental") == "pěti sty padesáti"        # ČRo-05 "s pěti sty padesáti lidmi"


def test_sto_followed_by_a_numeral_stays_sto():
    """IJP-792: 've sto dvaceti případech, po sto padesáti letech'."""
    assert w(120, case="locative") == "sto dvaceti"
    assert w(150, case="locative") == "sto padesáti"
    assert w(121, case="genitive") == "sto jednadvaceti"


# ---------------------------------------------------------------- tisíc, milion, miliarda
def test_tisic():
    assert paradigm(1000) == ["tisíc", "tisíce", "tisíci", "tisíc", "tisícem", "tisíci"]
    assert paradigm(2000) == ["dva tisíce", "dvou tisíc", "dvěma tisícům", "dva tisíce",
                              "dvěma tisíci", "dvou tisících"]
    assert paradigm(5000) == ["pět tisíc", "pěti tisíc", "pěti tisícům", "pět tisíc",
                              "pěti tisíci", "pěti tisících"]
    assert w(3000, case="instrumental") == "třemi tisíci"            # IJP-792 "s třemi tisíci diváků"
    assert w(333_000) == "tři sta třicet tři tisíc"                  # IJP:sto
    assert w(100_000) == "sto tisíc"
    assert paradigm(100_000) == ["sto tisíc", "sta tisíc", "stu tisícům", "sto tisíc", "stem tisíci",
                                 "stu tisících"]                           # review: "bez sta tisíc korun"
    assert w(21_000) == "dvacet jedna tisíc"
    assert w(22_000) == "dvacet dva tisíc"
    assert w(22_000, construction="agreement") == "dvacet dva tisíce"
    assert w(21_000, construction="inverted") == "jednadvacet tisíc"


def test_milion_miliarda():
    assert paradigm(10**6) == ["milion", "milionu", "milionu", "milion", "milionem", "milionu"]
    assert paradigm(2 * 10**6) == ["dva miliony", "dvou milionů", "dvěma milionům", "dva miliony",
                                   "dvěma miliony", "dvou milionech"]
    assert paradigm(5 * 10**6) == ["pět milionů", "pěti milionů", "pěti milionům", "pět milionů",
                                   "pěti miliony", "pěti milionech"]
    assert paradigm(10**9) == ["miliarda", "miliardy", "miliardě", "miliardu", "miliardou", "miliardě"]
    assert paradigm(2 * 10**9) == ["dvě miliardy", "dvou miliard", "dvěma miliardám", "dvě miliardy",
                                   "dvěma miliardami", "dvou miliardách"]
    assert w(3 * 10**9) == "tři miliardy"                            # IJP:miliarda
    assert w(53 * 10**6) == "padesát tři milionů"                    # IJP:milion "53 miliony i 53 milionů"
    assert w(53 * 10**6, construction="agreement") == "padesát tři miliony"
    assert w(22 * 10**9, construction="agreement") == "dvacet dvě miliardy"


def test_nominative_long_numbers():
    assert w(2345) == "dva tisíce tři sta čtyřicet pět"                               # WIKI-ČČ
    assert w(2231) == "dva tisíce dvě stě třicet jedna"                               # IJP-791
    assert w(5225) == "pět tisíc dvě stě dvacet pět"                                  # IJP:pět
    assert w(12_157_582) == "dvanáct milionů sto padesát sedm tisíc pět set osmdesát dva"  # IJP:milion
    assert w(1_300_783) == "milion tři sta tisíc sedm set osmdesát tři"               # IJP-791 (jeden optional)


# ---------------------------------------------------------------- long numbers in oblique cases (IJP-791)
def test_two_components_decline_fully():
    assert w(365, case="instrumental") == "třemi sty šedesáti pěti"                   # IJP-791 (a)
    assert w(365, case="instrumental", construction="inverted") == "třemi sty pětašedesáti"
    assert w(2300, case="genitive") == "dvou tisíc tří set"


def test_three_or_more_components_decline_partially():
    assert w(1847, case="genitive") == "tisíc osm set čtyřiceti sedmi"                # IJP-791 (b)
    assert w(1847, case="genitive", construction="inverted") == "tisíc osm set sedmačtyřiceti"
    assert w(1_358_423, case="locative") == \
        "milion tři sta padesát osm tisíc čtyři sta dvaceti třech"                  # IJP-791 (b)
    assert w(1223, case="genitive") == "tisíc dvě stě dvaceti tří"                    # NK-07
    assert w(1223, case="dative") == "tisíc dvě stě dvaceti třem"                     # NK-07
    assert w(11_255, case="instrumental") == "jedenáct tisíc dvě stě padesáti pěti"   # WIKI-ČČ


def test_oblique_style_full():
    assert w(1847, case="genitive", oblique_style="full") == "tisíce osmi set čtyřiceti sedmi"  # IJP-791 (a)
    assert w(1223, case="genitive", oblique_style="full") == "tisíce dvou set dvaceti tří"      # NK-07
    assert w(1223, case="dative", oblique_style="full") == "tisíci dvěma stům dvaceti třem"     # NK-07
    assert w(2152, case="instrumental", oblique_style="full") == "dvěma tisíci sto padesáti dvěma"  # ČRo-05
    assert w(11_255, case="instrumental", oblique_style="full") == \
        "jedenácti tisíci dvěma sty padesáti pěti"                                             # WIKI-ČČ
    assert w(365, case="instrumental", oblique_style="partial") == "tři sta šedesáti pěti"   # IJP-791 (b)


def test_nula_declines_as_a_noun():
    assert paradigm(0) == ["nula", "nuly", "nule", "nulu", "nulou", "nule"]            # IJP:nula


# ---------------------------------------------------------------- regressions for v1
_UNIT = r"(?:jedn|dva|tři|čtyři|pět|šest|sedm|osm|devět)"
_TENS = r"(?:dvacet|třicet|čtyřicet|padesát|šedesát|sedmdesát|osmdesát|devadesát)"
INVERTED = re.compile(rf"^{_UNIT}a{_TENS}i?$")
SIMPLE_WORDS = set("""
    nula nuly nule nulu nulou jeden jednoho jednomu jedním jednom jedna jedné jednu jednou jedno
    dva dvě dvou dvěma tři tří třem třemi třech čtyři čtyř čtyřem čtyřmi čtyřech
    sto sta stu stem stě set stům sty stech
    tisíc tisíce tisíci tisícem tisícům tisících
    milion milionu milionem miliony milionů milionům milionech
    miliarda miliardy miliardě miliardu miliardou miliard miliardám miliardami miliardách
""".split())
for _k in range(5, 100):
    if _k <= 20 or _k % 10 == 0:
        SIMPLE_WORDS |= {w(_k), w(_k, case="genitive")}


def test_every_word_is_a_dictionary_word():
    """v1 wrote 'dvacetjeden', 'stodvacetjeden', 'dvěstě', 'dvoutisícdvacetčtyři', 'dvamiliony'."""
    numbers = list(range(0, 1200)) + [1999, 2000, 2024, 2345, 5000, 21_000, 100_000, 10**6, 2 * 10**6,
                                      1_234_567, 10**9, 2 * 10**9]
    for n in numbers:
        for g in GENDERS:
            for c in CASE_ORDER:
                for style in ("auto", "full"):
                    for word in w(n, gender=g, case=c, oblique_style=style).split():
                        assert word in SIMPLE_WORDS or INVERTED.match(word), (n, g, c, style, word)


def test_no_ordinal_forms_leak_into_cardinals():
    """v1: 21 G 'dvacátéhojednoho', 100 G 'stého', 1000 G 'tisícího', 10^6 G 'miliontého'."""
    bad = re.compile(r"(ého|ému|ém|ým|ých|tý|tá|té)\b|\bst[áéý]\b|tisící(ho|mu|m)?\b")
    for n in list(range(0, 130)) + [200, 345, 999, 1000, 1001, 2000, 2024, 21_000, 10**6, 2 * 10**6, 10**9]:
        for g in GENDERS:
            for c in CASE_ORDER:
                out = w(n, gender=g, case=c)
                assert not bad.search(out), (n, g, c, out)


# ---------------------------------------------------------------- ordinals (WIKI-ŘČ, IJP-791, IJP:tisíc)
def test_simple_ordinals():
    assert paradigm(1, to="ordinal") == ["první", "prvního", "prvnímu", "první", "prvním", "prvním"]
    assert w(1, to="ordinal", case="accusative", animacy="animate") == "prvního"
    assert paradigm(1, to="ordinal", gender="feminine") == ["první"] * 6
    assert paradigm(2, to="ordinal") == ["druhý", "druhého", "druhému", "druhý", "druhým", "druhém"]
    assert w(2, to="ordinal", case="accusative", animacy="animate") == "druhého"
    assert paradigm(100, to="ordinal", gender="feminine") == ["stá", "sté", "sté", "stou", "stou", "sté"]
    assert paradigm(100, to="ordinal", gender="neuter") == ["sté", "stého", "stému", "sté", "stým", "stém"]
    assert paradigm(3, to="ordinal")[:3] == ["třetí", "třetího", "třetímu"]      # soft, like jarní
    assert w(0, to="ordinal") == "nultý"                                          # IJP:nula


def test_ordinal_plurals():
    assert w(2, to="ordinal", plural=True, animacy="animate") == "druzí"
    assert w(4, to="ordinal", plural=True, animacy="animate") == "čtvrtí"
    assert w(100, to="ordinal", plural=True, animacy="animate") == "stí"
    assert w(2, to="ordinal", plural=True) == "druhé"
    assert w(2, to="ordinal", plural=True, gender="neuter") == "druhá"
    assert w(2, to="ordinal", plural=True, case="accusative", animacy="animate") == "druhé"
    assert w(2, to="ordinal", plural=True, case="genitive") == "druhých"
    assert w(1, to="ordinal", plural=True, animacy="animate") == "první"
    assert w(1, to="ordinal", plural=True, case="instrumental") == "prvními"


def test_round_ordinals():
    assert [w(n, to="ordinal") for n in (200, 300, 400, 500, 900)] == \
        ["dvoustý", "třístý", "čtyřstý", "pětistý", "devítistý"]
    assert w(1000, to="ordinal") == "tisící"                                     # "tisící návštěvník"
    assert w(1000, to="ordinal", case="genitive") == "tisícího"
    assert w(2000, to="ordinal") == "dvoutisící"                                 # WIKI-ŘČ
    assert w(25_000, to="ordinal") == "pětadvacetitisící"                         # WIKI-ŘČ
    assert w(100_000, to="ordinal") == "stotisící"                                # WIKI-ŘČ
    assert w(10**6, to="ordinal") == "miliontý"                                   # NESČ
    assert w(10**9, to="ordinal") == "miliardtý"


def test_compound_ordinals_every_part_ordinal():
    """ordinal_style="all": every part is ordinal (the basic type in IJP and WIKI-ŘČ)."""
    a = dict(to="ordinal", ordinal_style="all")
    assert w(25, **a) == "dvacátý pátý"                                          # IJP-791
    assert w(150, **a) == "stý padesátý"                                         # IJP-791
    assert w(478, **a) == "čtyřstý sedmdesátý osmý"                              # IJP-791
    assert w(1956, **a) == "tisící devítistý padesátý šestý"                     # IJP-791
    assert w(152, **a) == "stý padesátý druhý"                                   # WIKI-ŘČ
    assert w(215, **a) == "dvoustý patnáctý"                                     # WIKI-ŘČ
    assert w(368, **a) == "třístý šedesátý osmý"                                 # WIKI-ŘČ
    assert w(2345, **a) == "dvoutisící třístý čtyřicátý pátý"                    # WIKI-ČČ
    assert w(1520, **a) == "tisící pětistý dvacátý"                              # IJP:tisíc
    assert w(5205, **a) == "pětitisící dvoustý pátý"                             # CHL (Gebauer)
    assert w(25, to="ordinal", inverted=True) == "pětadvacátý"                   # IJP-791
    assert w(21, to="ordinal", inverted=True) == "jednadvacátý"                  # NESČ


def test_compound_ordinals_decline_every_part():
    a = dict(to="ordinal", ordinal_style="all")
    assert w(1875, case="genitive", **a) == "tisícího osmistého sedmdesátého pátého"   # WIKI-ŘČ
    assert w(1905, case="genitive", **a) == "tisícího devítistého pátého"              # SSJČ
    assert w(2024, case="genitive", **a) == "dvoutisícího dvacátého čtvrtého"
    assert w(2024, case="genitive", gender="feminine", **a) == "dvoutisící dvacáté čtvrté"
    assert paradigm(101, gender="feminine", **a) == \
        ["stá první", "sté první", "sté první", "stou první", "stou první", "sté první"]


def test_mixed_ordinal_style_is_the_default():
    """Native review (25 Sep 2026) chose the mixed type for years; attested in CHL and WIKI-ŘČ."""
    assert w(1991, to="ordinal") == "tisíc devět set devadesátý první"
    assert w(1875, to="ordinal", case="genitive") == \
        "tisíc osm set sedmdesátého pátého"                                      # WIKI-ŘČ (Brus 1877)
    assert w(101, to="ordinal") == "sto první"                                   # CHL (Sedláček)
    assert w(158, to="ordinal") == "sto padesátý osmý"                           # CHL (Sedláček)
    assert w(365, to="ordinal") == "tři sta šedesátý pátý"                       # native review: not only years
    assert w(2024, to="ordinal") == "dva tisíce dvacátý čtvrtý"
    assert w(2024, to="ordinal", case="genitive", gender="feminine") == "dva tisíce dvacáté čtvrté"
    assert w(2000, to="ordinal") == "dvoutisící"                                 # one part: stays ordinal
    assert w(1991, to="ordinal", ordinal_style="mixed") == w(1991, to="ordinal")


# ---------------------------------------------------------------- decimals (IJP-791)
@pytest.mark.parametrize("x,expected", [
    (0.1, "nula celá jedna desetina"),
    (0.2, "nula celá dvě desetiny"),
    (0.5, "nula celá pět desetin"),
    (1.2, "jedna celá dvě desetiny"),
    (1.9, "jedna celá devět desetin"),
    (2.3, "dvě celé tři desetiny"),
    (5.1, "pět celých jedna desetina"),
    (25.4, "dvacet pět celých čtyři desetiny"),
    (100.6, "sto celých šest desetin"),
    (103.8, "sto tři celé osm desetin"),
    (1000.05, "tisíc celých pět setin"),
    (1024.007, "tisíc dvacet čtyři celé sedm tisícin"),
    (3.14, "tři celé čtrnáct setin"),
    (0.26, "nula celá dvacet šest setin"),                                        # HRD
    ("2,5", "dvě celé pět desetin"),
    (-2.5, "mínus dvě celé pět desetin"),
])
def test_decimals(x, expected):
    assert w(x) == expected


# ---------------------------------------------------------------- API
def test_api():
    assert CASES == CASE_ORDER
    assert Num2Word_CS().to_cardinal(5, case="genitive") == "pěti"
    assert w(1, case="accusative", animate=True) == "jednoho"        # v1-style alias
    assert w(-5, case="genitive") == "mínus pěti"
    assert w(2.5, case="genitive") == "dvě celé pět desetin"         # decimals: nominative reading (review)
    with pytest.raises(ValueError):
        w(5, case="vocative")
    with pytest.raises(ValueError):
        w(5, gender="plural")
    with pytest.raises(ValueError):
        w(5, to="fraction")
    with pytest.raises(ValueError):
        w(-1, to="ordinal")
