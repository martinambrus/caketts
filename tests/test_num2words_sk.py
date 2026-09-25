# -*- coding: utf-8 -*-
"""
Tests for num2words_sk v2. Expected values are quoted from the sources listed in
NUM2WORDS_CHANGES.md (MSJ = Morfológia slovenského jazyka 1966; PSP = Pravidlá slovenského
pravopisu 1991/1998, via Jarošová 2021; NAV = Navrátil 2003; PAL = Páleníková; TASR18 = TASR
style guide; DUCH = Duchková), or were decided in the native-speaker review of 25 Sep 2026
where the sources are silent or allow variants (test_native_review_decisions).
Run: pytest test_num2words_sk.py
"""
import re

import pytest

try:  # standalone: the module sits next to this file
    from num2words_sk import CASES, Num2Word_SK, num2words as w
except ImportError:  # inside the TTS repository
    from src.text.num2words_sk import CASES, Num2Word_SK, num2words as w

CASE_ORDER = ["nominative", "genitive", "dative", "accusative", "instrumental", "locative"]


def paradigm(n, **kw):
    return [w(n, case=c, **kw) for c in CASE_ORDER]


# ---------------------------------------------------------------- cardinals 1-4 (MSJ pp. 318-323)
def test_jeden_by_gender_and_animacy():
    assert paradigm(1) == ["jeden", "jedného", "jednému", "jeden", "jedným", "jednom"]
    assert w(1, case="accusative", animacy="animate") == "jedného"
    assert paradigm(1, gender="feminine") == ["jedna", "jednej", "jednej", "jednu", "jednou", "jednej"]
    assert paradigm(1, gender="neuter") == ["jedno", "jedného", "jednému", "jedno", "jedným", "jednom"]


def test_dva_tri_styri_with_masculine_personal_forms():
    assert paradigm(2) == ["dva", "dvoch", "dvom", "dva", "dvoma", "dvoch"]
    assert paradigm(2, gender="feminine") == ["dve", "dvoch", "dvom", "dve", "dvomi", "dvoch"]
    assert paradigm(2, animacy="personal")[0::3] == ["dvaja", "dvoch"]       # N, A
    assert paradigm(3, animacy="personal")[0::3] == ["traja", "troch"]
    assert paradigm(4, animacy="personal")[0::3] == ["štyria", "štyroch"]
    assert paradigm(3)[0::3] == ["tri", "tri"]                               # not "troch" for inanimates
    # instrumental: dvoma/troma for masculine and neuter, dvomi/tromi for feminine (native review);
    # MSJ p. 320-323 and SNK 2011 list both forms as standard; štyrmi is the only form of 4
    assert w(3, case="instrumental") == "troma"
    assert w(3, gender="feminine", case="instrumental") == "tromi"
    assert {w(4, gender=g, case="instrumental") for g in ("masculine", "feminine", "neuter")} == {"štyrmi"}


# ---------------------------------------------------------------- 5-99 (MSJ p. 324; POV18)
def test_pat_paradigm():
    assert paradigm(5) == ["päť", "piatich", "piatim", "päť", "piatimi", "piatich"]
    assert w(5, animacy="personal", construction="agreement") == "piati"
    assert w(5, animacy="personal", construction="agreement", case="accusative") == "piatich"


@pytest.mark.parametrize("n,gen", [(7, "siedmich"), (8, "ôsmich"), (17, "sedemnástich"),
                                   (70, "sedemdesiatich"), (80, "osemdesiatich"), (20, "dvadsiatich")])
def test_stems(n, gen):
    assert w(n, case="genitive") == gen


def test_compounds_ending_in_jeden_never_decline():
    """PAL/NAV/MSJ p. 326: dvadsaťjeden... sa nikdy neskloňujú; jeden for all genders."""
    for g in ("masculine", "feminine", "neuter"):
        assert set(paradigm(21, gender=g)) == {"dvadsaťjeden"}
    assert w(61, case="instrumental") == "šesťdesiatjeden"      # "so šesťdesiatjeden hosťami"
    assert w(41, case="locative", gender="feminine") == "štyridsaťjeden"  # "o štyridsaťjeden ženách"


def test_22_declines_both_parts_as_cardinals():
    """MSJ p. 326: dvadsiatich dvoch žiakov, dvadsiatim dvom žiakom, s dvadsiatimi dvoma žiakmi."""
    assert paradigm(22)[1:3] + paradigm(22)[4:5] == ["dvadsiatich dvoch", "dvadsiatim dvom", "dvadsiatimi dvoma"]
    assert w(22, gender="feminine") == "dvadsaťdva"             # masculine form for all genders
    assert w(22, case="locative", declined=False) == "dvadsaťdva"  # the undeclined variant
    assert w(25, animacy="personal", construction="agreement") == "dvadsiati piati"  # NAV


def test_psp_and_tasr_quotes():
    assert w(124, case="genitive", animacy="personal") == "stodvadsiatich štyroch"      # PSP 91
    assert w(124, animacy="personal", construction="agreement") == "stodvadsiati štyria"  # PSP 91
    assert w(245, case="instrumental") == "dvestoštyridsiatimi piatimi"                 # TASR18
    assert w(352, case="dative") == "tristopäťdesiatim dvom"                            # TASR18
    assert w(1225, case="genitive") == "tisícdvestodvadsiatich piatich"                 # JAR p. 214


# ---------------------------------------------------------------- sto, tisíc, milión (BEL, NAV, JAR)
@pytest.mark.parametrize("n", [100, 200, 500, 900, 1000, 2000, 5000, 21000, 100000, 101, 121, 1001])
def test_invariable_with_counted_nouns(n):
    for g in ("masculine", "feminine", "neuter"):
        assert len(set(paradigm(n, gender=g))) == 1, (n, g, paradigm(n, gender=g))


def test_milion_miliarda_are_declined_nouns_written_apart():
    assert paradigm(1_000_000) == ["milión", "milióna", "miliónu", "milión", "miliónom", "milióne"]
    assert paradigm(2_000_000) == ["dva milióny", "dvoch miliónov", "dvom miliónom", "dva milióny",
                                   "dvoma miliónmi", "dvoch miliónoch"]
    assert paradigm(5_000_000) == ["päť miliónov", "piatich miliónov", "piatim miliónom", "päť miliónov",
                                   "piatimi miliónmi", "piatich miliónoch"]
    assert paradigm(2_000_000_000) == ["dve miliardy", "dvoch miliárd", "dvom miliardám", "dve miliardy",
                                       "dvomi miliardami", "dvoch miliardách"]
    assert w(5_000_000_000) == "päť miliárd"
    assert w(5_234_567) == "päť miliónov dvestotridsaťštyritisícpäťstošesťdesiatsedem"  # JAR fn. 18


def test_undeclined_variant_keeps_every_compound_invariable():
    """MSJ p. 326: compounds 'sa alebo neskloňujú, alebo sa skloňujú v oboch častiach'."""
    assert w(202, case="genitive") == "dvesto dvoch"                        # PSP 91: declined units apart
    assert w(202, case="genitive", declined=False) == "dvestodva"
    assert w(112, case="instrumental", declined=False) == "stodvanásť"
    assert w(1_000_005, case="genitive") == "milióna piatich"
    assert w(1_000_005, case="genitive", declined=False) == "milióna päť"      # milión is a noun
    assert w(5, case="genitive", declined=False) == "piatich"               # simple numerals always decline


def test_scale_groups_always_decline():
    """milión/miliarda are nouns: they decline also before a smaller part (native review)."""
    assert w(2_000_005, case="genitive") == "dvoch miliónov piatich"
    assert w(2_500_000, case="dative") == "dvom miliónom päťstotisíc"
    assert w(1_500_000_000, case="dative") == "miliarde päťsto miliónom"
    assert w(1_000_002_000_000, case="instrumental") == "biliónom dvoma miliónmi"


# ---------------------------------------------------------------- native review, 25 Sep 2026
def test_native_review_decisions():
    """The sources are silent here, or allow variants; a native Slovak speaker decided."""
    assert w(22_000) == "dvadsaťdvatisíc"
    assert w(102, gender="feminine") == "stodve"                       # "stodve knihy", not "stodva kníh"
    assert w(102) == "stodva"
    assert w(122, gender="feminine") == "stodvadsaťdva"                # after tens the MSJ rule applies
    assert w(2, case="instrumental") == "dvoma"                        # "dvoma stromami"
    assert w(2, gender="feminine", case="instrumental") == "dvomi"     # "dvomi stenami"
    assert w(22, gender="feminine", case="instrumental") == "dvadsiatimi dvomi"
    assert w(2, gender="neuter", case="instrumental") == "dvoma"       # "dvoma mestami"
    assert w(3, gender="feminine", case="instrumental") == "tromi"     # "tromi ženami"
    assert w(3, case="instrumental") == "troma"                        # "troma mužmi"
    assert w(102, gender="neuter") == "stodve"                         # dve mestá
    assert w(2002, gender="feminine") == "dvetisícdve"
    assert w(1_000_002, gender="feminine") == "milión dve"
    assert w(102 * 10**6) == "stodva milióny"                          # like "stodve knihy"
    assert w(102 * 10**9) == "stodve miliardy"
    assert w(105 * 10**6) == "stopäť miliónov"                         # 5 and up: genitive plural
    assert w(22 * 10**9) == "dvadsaťdva miliárd"                       # after tens: genitive plural
    assert w(2.5, case="genitive") == "dve celé päť desatín"           # decimals: nominative reading
    assert w(300, to="ordinal") == "trojstý" and w(400, to="ordinal") == "štvorstý"
    assert w(1_000_001, to="ordinal") == "milión prvý"
    assert w(5000, to="ordinal") == "päťtisíci"
    assert w(21_000, to="ordinal") == "dvadsaťjedentisíci"


def test_nula_declines_as_a_noun():
    assert paradigm(0) == ["nula", "nuly", "nule", "nulu", "nulou", "nule"]


def test_no_ordinal_forms_leak_into_cardinals():
    """Regression for v1: 21 G 'dvadsiatehojedného', 100 G 'stého', 1000 G 'tisíceho'."""
    bad = re.compile(r"(ého|ému|eho|emu|ým|ym)\b|stej\b|tisícej\b")
    for n in list(range(0, 130)) + [200, 345, 999, 1000, 1001, 2024, 21000, 10**6, 2 * 10**6, 10**9]:
        for g in ("masculine", "feminine", "neuter"):
            for c in CASE_ORDER:
                out = w(n, gender=g, case=c)
                if n == 1 and g != "feminine":  # jedného/jednému/jedným are genuine cardinal forms
                    continue
                assert not bad.search(out), (n, g, c, out)


# ---------------------------------------------------------------- ordinals (MSJ, SSSJ, PSP)
def test_simple_ordinals():
    assert paradigm(1, to="ordinal") == ["prvý", "prvého", "prvému", "prvý", "prvým", "prvom"]
    assert w(1, to="ordinal", case="accusative", animacy="animate") == "prvého"
    assert paradigm(3, to="ordinal") == ["tretí", "tretieho", "tretiemu", "tretí", "tretím", "treťom"]
    assert paradigm(3, to="ordinal", gender="feminine") == ["tretia", "tretej", "tretej", "tretiu", "treťou", "tretej"]
    assert paradigm(5, to="ordinal")[:3] == ["piaty", "piateho", "piatemu"]   # rhythmic law
    assert w(7, to="ordinal", case="genitive") == "siedmeho"


def test_ordinal_plurals():
    assert w(1, to="ordinal", plural=True, animacy="personal") == "prví"
    assert w(5, to="ordinal", plural=True, animacy="personal") == "piati"
    assert w(1, to="ordinal", plural=True, gender="feminine") == "prvé"
    assert w(5, to="ordinal", plural=True, case="genitive") == "piatych"


def test_special_ordinals():
    assert w(1000, to="ordinal", gender="feminine") == "tisíca"          # SSSJ: -tisíci -ca -ce
    assert w(1000, to="ordinal", gender="neuter") == "tisíce"
    assert w(1000, to="ordinal", case="genitive") == "tisíceho"
    assert w(2000, to="ordinal") == "dvetisíci"                           # native review; SRA: more frequent
    assert w(2000, to="ordinal", codified=True) == "dvojtisíci"           # SSSJ: the codified form
    assert w(200, to="ordinal") == "dvestý" and w(200, to="ordinal", codified=True) == "dvojstý"
    assert w(3000, to="ordinal") == "trojtisíci"                          # SRA: *tritisíci is unattested
    assert w(1200, to="ordinal") == "tisícdvestý"                         # native review: 200th follows 2000th
    assert w(2 * 10**6, to="ordinal") == "dvemiliónty"
    assert w(10**6, to="ordinal", case="genitive") == "miliónteho"        # not "milióny"
    assert w(0, to="ordinal") == "nultý"


def test_compound_ordinals_keep_cardinal_prefixes():
    assert w(21, to="ordinal") == "dvadsiaty prvý"                                   # BEL: two words
    assert w(21, to="ordinal", case="genitive") == "dvadsiateho prvého"
    assert w(101, to="ordinal") == "stoprvý"                                        # PSP 98
    assert w(2002, to="ordinal") == "dvetisícdruhý"                                 # PSP 98
    assert w(124, to="ordinal", animacy="personal") == "stodvadsiaty štvrtý"        # PSP 91
    assert w(124, to="ordinal", case="genitive", animacy="personal") == "stodvadsiateho štvrtého"
    assert w(1985, to="ordinal", case="locative") == "tisícdeväťstoosemdesiatom piatom"  # PSP 91
    assert w(2024, to="ordinal") == "dvetisícdvadsiaty štvrtý"
    assert w(102, to="ordinal", case="genitive") == "stodruhého"                    # JAR p. 217


# ---------------------------------------------------------------- decimals (DUCH, HS)
@pytest.mark.parametrize("x,expected", [
    (0.25, "nula celých dvadsaťpäť stotín"),
    (1.9, "jedna celá deväť desatín"),
    (2.84, "dve celé osemdesiatštyri stotín"),
    (41.2, "štyridsaťjeden celých dve desatiny"),
    (396.4, "tristodeväťdesiatšesť celých štyri desatiny"),
    (2.531, "dve celé päťstotridsaťjeden tisícin"),
    ("68,50", "šesťdesiatosem celých päť desatín"),
    (3.14, "tri celé štrnásť stotín"),
    (-2.5, "mínus dve celé päť desatín"),
])
def test_decimals(x, expected):
    assert w(x) == expected


# ---------------------------------------------------------------- API
def test_api():
    assert CASES == CASE_ORDER
    assert Num2Word_SK().to_cardinal(5, case="genitive") == "piatich"
    assert w(1, case="accusative", animate=True) == "jedného"        # v1-style alias
    assert w(-5, case="genitive") == "mínus piatich"
    with pytest.raises(ValueError):
        w(5, case="vocative")
    with pytest.raises(ValueError):
        w(5, gender="plural")
    with pytest.raises(ValueError):
        w(5, to="fraction")
