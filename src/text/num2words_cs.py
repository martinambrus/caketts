# -*- coding: utf-8 -*-
"""
Czech numbers to words — cardinals, ordinals and decimals with case, gender and animacy,
for TTS text normalisation.

v2 (25 September 2026): rewritten from a sourced normative spec (Internetová jazyková
příručka ÚJČ AV ČR, Czech Wikipedia "České číslovky"/"Řadová číslovka", Nový encyklopedický
slovník češtiny; see NUM2WORDS_CHANGES.md). v1 was generated from the Slovak design and wrote
compounds as one word ("dvoutisícdvacetčtyři"), used "dvoutisíc" for 2000 and built oblique
cardinals from ordinal stems (21 f. G "dvacátéhojedné", 100 f. G "sté").

Main rules implemented
  * every word of a numeral is written separately: "dvacet pět", "dva tisíce dvacet čtyři";
    only the inverted type is one word ("pětadvacet", "jednadvacátý")
  * 1-4 decline (jeden/jedna/jedno with animate accusative "jednoho"; dva/dvě, dvou, dvěma;
    tři, tří, třem, třemi, třech; čtyři, čtyř, čtyřem, čtyřmi, čtyřech); 5-99 end in -i
    in oblique cases (pěti, devíti, dvaceti)
  * compounds 21-99 in N/A: "dvacet jedna/dva" + genitive plural noun (neutral, more common),
    or the agreement type "dvacet jeden muž / dvacet dvě ženy" (construction="agreement"),
    or the inverted type (construction="inverted"); oblique: both words decline
    ("dvaceti pěti"); compounds ending in 1 use the inverted "jednadvaceti". The agreement type
    applies to the nominative and accusative only: native review rejected "dvaceti jednoho"
  * sto, dvě stě, tři sta, pět set, tisíc, milion, miliarda decline as nouns
    (sta/stu/stem, dvou set, tisíce/tisícem, dvou tisíc, milionu, miliardu);
    "sto" followed by another numeral stays "sto" ("ve sto dvaceti případech"), but it declines
    before tisíc/milion: "bez sta tisíc korun" (native review)
  * large numbers in oblique cases: all parts decline for up to two components ("dvou tisíc
    tří set"); from three components only the last one declines ("dva tisíce tři sta
    čtyřiceti pěti"), both IJP-sanctioned
  * ordinals: by default (ordinal_style="mixed", chosen in native review for years) the
    leading hundreds/thousands stay cardinal and only the last part is ordinal and declines:
    "sto dvacátý první", "tisíc devět set devadesátý první", "dva tisíce dvacátý čtvrtý",
    G "tisíc osm set sedmdesátého pátého". ordinal_style="all" makes every part ordinal
    (IJP: "tisící devítistý padesátý šestý", "dvoutisící dvacátý čtvrtý")
  * decimals agree with "celá": "jedna celá pět desetin", "tři celé čtrnáct setin"
  * nula declines as a noun (nuly, nule, nulu, nulou)

API (compatible with v1):
    num2words(number, to="cardinal"|"ordinal", gender="masculine"|"feminine"|"neuter",
              case="nominative"|..., animacy="inanimate"|"animate"|"personal", plural=False,
              construction="genitive"|"agreement"|"inverted", oblique_style="auto"|"full"|"partial",
              ordinal_style="mixed"|"all", inverted=False)
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import List, Tuple

CASES = ["nominative", "genitive", "dative", "accusative", "instrumental", "locative"]
N, G, D, A, I, L = range(6)
GENDERS = ("masculine", "feminine", "neuter")
ANIMACY = ("inanimate", "animate", "personal")

ZERO_FORMS = ("nula", "nuly", "nule", "nulu", "nulou", "nule")
MINUS = "mínus"

# --------------------------------------------------------------------------------------
# Cardinal building blocks (N, G, D, A, I, L)
# --------------------------------------------------------------------------------------

ONE = {
    "masculine": ("jeden", "jednoho", "jednomu", None, "jedním", "jednom"),  # A by animacy
    "feminine": ("jedna", "jedné", "jedné", "jednu", "jednou", "jedné"),
    "neuter": ("jedno", "jednoho", "jednomu", "jedno", "jedním", "jednom"),
}
TWO_OBL = {G: "dvou", D: "dvěma", I: "dvěma", L: "dvou"}
THREE = ("tři", "tří", "třem", "tři", "třemi", "třech")
FOUR = ("čtyři", "čtyř", "čtyřem", "čtyři", "čtyřmi", "čtyřech")
# 5-20 and round tens: (nominative, oblique = G = D = I = L)
PET = {
    5: ("pět", "pěti"), 6: ("šest", "šesti"), 7: ("sedm", "sedmi"), 8: ("osm", "osmi"),
    9: ("devět", "devíti"), 10: ("deset", "deseti"), 11: ("jedenáct", "jedenácti"),
    12: ("dvanáct", "dvanácti"), 13: ("třináct", "třinácti"), 14: ("čtrnáct", "čtrnácti"),
    15: ("patnáct", "patnácti"), 16: ("šestnáct", "šestnácti"), 17: ("sedmnáct", "sedmnácti"),
    18: ("osmnáct", "osmnácti"), 19: ("devatenáct", "devatenácti"), 20: ("dvacet", "dvaceti"),
    30: ("třicet", "třiceti"), 40: ("čtyřicet", "čtyřiceti"), 50: ("padesát", "padesáti"),
    60: ("šedesát", "šedesáti"), 70: ("sedmdesát", "sedmdesáti"), 80: ("osmdesát", "osmdesáti"),
    90: ("devadesát", "devadesáti"),
}
INVERTED_UNIT = {1: "jedn", 2: "dva", 3: "tři", 4: "čtyři", 5: "pět", 6: "šest", 7: "sedm",
                 8: "osm", 9: "devět"}  # jedn-a-dvacet, pět-a-dvacet

# hundreds: N form and oblique forms; "sto" is a neuter noun (vzor město)
HUNDREDS = {
    1: ("sto", "sta", "stu", "sto", "stem", "stu"),
    2: ("dvě stě", "dvou set", "dvěma stům", "dvě stě", "dvěma sty", "dvou stech"),
    3: ("tři sta", "tří set", "třem stům", "tři sta", "třemi sty", "třech stech"),
    4: ("čtyři sta", "čtyř set", "čtyřem stům", "čtyři sta", "čtyřmi sty", "čtyřech stech"),
}
for _h in range(5, 10):
    _n, _o = PET[_h]
    HUNDREDS[_h] = (f"{_n} set", f"{_o} set", f"{_o} stům", f"{_n} set", f"{_o} sty", f"{_o} stech")

# scale nouns (sg forms, pl forms); tisíc takes G pl "tisíc" after numerals
_HRAD = lambda s: ((s, s + "u", s + "u", s, s + "em", s + "u"),
                   (s + "y", s + "ů", s + "ům", s + "y", s + "y", s + "ech"))
_ZENA = lambda s: ((s + "a", s + "y", s + "ě", s + "u", s + "ou", s + "ě"),
                   (s + "y", s, s + "ám", s + "y", s + "ami", s + "ách"))
TISIC = (("tisíc", "tisíce", "tisíci", "tisíc", "tisícem", "tisíci"),
         ("tisíce", "tisíc", "tisícům", "tisíce", "tisíci", "tisících"))
SCALES = [  # (exponent, gender, sg, pl, ordinal lemma)
    (21, "feminine", *_ZENA("triliard"), "triliardtý"),
    (18, "masculine", *_HRAD("trilion"), "triliontý"),
    (15, "feminine", *_ZENA("biliard"), "biliardtý"),
    (12, "masculine", *_HRAD("bilion"), "biliontý"),
    (9, "feminine", *_ZENA("miliard"), "miliardtý"),
    (6, "masculine", *_HRAD("milion"), "miliontý"),
    (3, "masculine", *TISIC, "tisící"),
]


def _idx(case: str) -> int:
    if case not in CASES:
        raise ValueError(f"case must be one of {CASES}, got {case!r}")
    return CASES.index(case)


def _check(gender: str, animacy: str) -> None:
    if gender not in GENDERS:
        raise ValueError(f"gender must be one of {GENDERS}, got {gender!r}")
    if animacy not in ANIMACY:
        raise ValueError(f"animacy must be one of {ANIMACY}, got {animacy!r}")


def _simple(n: int, c: int, gender: str, animacy: str) -> str:
    """1-20 and round tens, with agreement."""
    if n == 1:
        if c == A and gender == "masculine":
            return "jednoho" if animacy != "inanimate" else "jeden"
        return ONE[gender][c]
    if n == 2:
        if c in (N, A):
            return "dva" if gender == "masculine" else "dvě"
        return TWO_OBL[c]
    if n == 3:
        return THREE[c]
    if n == 4:
        return FOUR[c]
    nom, obl = PET[n]
    return nom if c in (N, A) else obl


def _below_100(n: int, c: int, gender: str, animacy: str, construction: str) -> str:
    if n <= 20 or n % 10 == 0:
        return _simple(n, c, gender, animacy)
    tens, unit = n - n % 10, n % 10
    oblique = c not in (N, A)
    inverted = construction == "inverted" or (unit == 1 and oblique)  # never "dvaceti jednoho"
    if inverted:  # jednadvacet / jednadvaceti, pětadvacet / pětadvaceti
        t_nom, t_obl = PET[tens]
        return INVERTED_UNIT[unit] + "a" + (t_obl if oblique else t_nom)
    t = PET[tens][1] if oblique else PET[tens][0]
    if not oblique and construction == "genitive" and unit in (1, 2):
        return f"{t} {'jedna' if unit == 1 else 'dva'}"  # invariable with a G pl noun
    return f"{t} {_simple(unit, c, gender, animacy)}"


def _below_1000(n: int, c: int, gender: str, animacy: str, construction: str, keep_sto: bool = True) -> str:
    h, rest = divmod(n, 100)
    words = []
    if h:
        # "sto" followed by another numeral stays undeclined (IJP); other hundreds decline
        hc = N if (h == 1 and rest and keep_sto and c not in (N, A)) else c
        words.append(HUNDREDS[h][hc])
    if rest:
        if h and rest == 1 and c not in (N, A):
            words.append("jedna")  # no inverted form exists for 101; keep the invariable unit
        elif h and rest in (1, 2) and c in (N, A) and construction == "genitive":
            words.append("jedna" if rest == 1 else "dva")
        else:
            words.append(_below_100(rest, c, gender, animacy, construction))
    return " ".join(words)


def _count_noun(count: int, sg: tuple, pl: tuple, sgender: str, c: int, construction: str) -> str:
    """count x tisíc/milion/miliarda in case c (both parts decline)."""
    if count == 1:
        return sg[c]
    if count in (2, 3, 4):
        return f"{_simple(count, c, sgender, 'inanimate')} {pl[c]}"
    if count == 100:  # sto declines before tisíc/milion: "bez sta tisíc", "se stem tisíci" (review)
        return f"{HUNDREDS[1][c]} {pl[G] if c in (N, A) else pl[c]}"
    if c in (N, A) and construction == "agreement":
        # type A: the noun agrees with the last units word (dvacet jeden milion, dvacet dvě miliardy)
        t = count % 100
        unit = t % 10 if t > 20 else (t if t < 5 else 0)
        if unit in (1, 2, 3, 4):
            num = _below_1000(count, c, sgender, "inanimate", "agreement")
            return f"{num} {sg[c] if unit == 1 else pl[c]}"
    num = _below_1000(count, c, sgender, "inanimate", "genitive" if construction == "agreement" else construction)
    return f"{num} {pl[G] if c in (N, A) else pl[c]}"  # after 5+ and type-B compounds: G pl


def _components(n: int) -> List[Tuple[int, tuple]]:
    """Split into (value, scale) components: scale groups, then the sub-1000 tail."""
    comps = []
    rest = n
    for exp, sgender, sg, pl, _lemma in SCALES:
        count, rest = divmod(rest, 10 ** exp)
        if count:
            if count >= 1000:
                raise ValueError("number too large")
            comps.append((count, (sg, pl, sgender)))
    h, tu = divmod(rest, 100)
    if h:
        comps.append((h * 100, None))
    if tu:
        comps.append((tu, None))
    return comps


def int_to_cardinal(n: int, gender: str = "masculine", case: str = "nominative",
                    animacy: str = "inanimate", construction: str = "genitive",
                    oblique_style: str = "auto") -> str:
    _check(gender, animacy)
    c = _idx(case)
    if n < 0:
        return f"{MINUS} {int_to_cardinal(-n, gender, case, animacy, construction, oblique_style)}"
    if n == 0:
        return ZERO_FORMS[c]
    comps = _components(n)
    style = oblique_style
    if style == "auto":
        style = "partial" if len(comps) >= 3 else "full"
    words = []
    for k, (value, scale) in enumerate(comps):
        last = k == len(comps) - 1
        cc = c if (style == "full" or last or c in (N, A)) else N
        if scale is not None:
            sg, pl, sgender = scale
            words.append(_count_noun(value, sg, pl, sgender, cc, construction))
        elif value >= 100:
            followed = not last
            hc = N if (value == 100 and followed and cc not in (N, A)) else cc
            words.append(HUNDREDS[value // 100][hc])
        else:
            multi = len(comps) > 1
            if multi and value == 1:  # tisíc jedna; agreement type: tisíc jeden (muž), tisíce jednoho
                agree = construction == "agreement" and cc in (N, A)
                words.append(_simple(1, cc, gender, animacy) if agree else "jedna")
            elif multi and value == 2 and cc in (N, A) and construction == "genitive":
                words.append("dva")
            else:
                words.append(_below_100(value, cc, gender, animacy, construction))
    return " ".join(words)


# --------------------------------------------------------------------------------------
# Ordinals
# --------------------------------------------------------------------------------------

ORDINALS = {
    0: "nultý", 1: "první", 2: "druhý", 3: "třetí", 4: "čtvrtý", 5: "pátý", 6: "šestý",
    7: "sedmý", 8: "osmý", 9: "devátý", 10: "desátý", 11: "jedenáctý", 12: "dvanáctý",
    13: "třináctý", 14: "čtrnáctý", 15: "patnáctý", 16: "šestnáctý", 17: "sedmnáctý",
    18: "osmnáctý", 19: "devatenáctý", 20: "dvacátý", 30: "třicátý", 40: "čtyřicátý",
    50: "padesátý", 60: "šedesátý", 70: "sedmdesátý", 80: "osmdesátý", 90: "devadesátý",
    100: "stý", 200: "dvoustý", 300: "třístý", 400: "čtyřstý", 500: "pětistý",
    600: "šestistý", 700: "sedmistý", 800: "osmistý", 900: "devítistý",
}
# combining forms for multiplied ordinals: dvoutisící, pětitisící, dvacetitisící, stotisící
COMBINING = {2: "dvou", 3: "tří", 4: "čtyř", 5: "pěti", 6: "šesti", 7: "sedmi", 8: "osmi",
             9: "devíti", 10: "deseti", 11: "jedenácti", 12: "dvanácti", 13: "třinácti",
             14: "čtrnácti", 15: "patnácti", 16: "šestnácti", 17: "sedmnácti", 18: "osmnácti",
             19: "devatenácti", 20: "dvaceti", 30: "třiceti", 40: "čtyřiceti", 50: "padesáti",
             60: "šedesáti", 70: "sedmdesáti", 80: "osmdesáti", 90: "devadesáti", 100: "sto"}


def _combining(count: int) -> str:
    if count in COMBINING:
        return COMBINING[count]
    if count < 100:  # pětadvacetitisící [WIKI-ŘČ]; jednadvacetitisící [inferred]
        unit, tens = count % 10, count - count % 10
        return INVERTED_UNIT[unit] + "a" + PET[tens][1]
    raise ValueError(f"no ordinal combining form for {count}; use ordinal_style='mixed'")


_HARD = {  # mladý: N G D A I L
    "m": ("ý", "ého", "ému", None, "ým", "ém"), "f": ("á", "é", "é", "ou", "ou", "é"),
    "n": ("é", "ého", "ému", "é", "ým", "ém"),
}
_HARD_PL = {"anim": ("í", "ých", "ým", "é", "ými", "ých"), "m": ("é", "ých", "ým", "é", "ými", "ých"),
            "f": ("é", "ých", "ým", "é", "ými", "ých"), "n": ("á", "ých", "ým", "á", "ými", "ých")}
_SOFT = {  # jarní
    "m": ("í", "ího", "ímu", None, "ím", "ím"), "f": ("í", "í", "í", "í", "í", "í"),
    "n": ("í", "ího", "ímu", "í", "ím", "ím"),
}
_SOFT_PL = ("í", "ích", "ím", "í", "ími", "ích")
_ANIM_PL_STEM = {"ch": "š", "h": "z", "k": "c", "r": "ř"}  # druhý -> druzí ("ch" before "h")


def decline_ordinal(lemma: str, c: int, gender: str, animacy: str, plural: bool) -> str:
    stem, last = lemma[:-1], lemma[-1]
    g = {"masculine": "m", "feminine": "f", "neuter": "n"}[gender]
    anim = gender == "masculine" and animacy != "inanimate"
    if last == "í":  # soft
        if plural:
            return stem + _SOFT_PL[c]
        e = _SOFT[g][c]
        if e is None:
            e = _SOFT["m"][G] if anim else _SOFT["m"][N]
        return stem + e
    if plural:
        if anim and c == N:  # druzí, pátí, stí; the plural accusative is -é for all animacy
            for k, v in _ANIM_PL_STEM.items():
                if stem.endswith(k):
                    return stem[: -len(k)] + v + "í"
            return stem + "í"
        return stem + _HARD_PL[g][c]
    e = _HARD[g][c]
    if e is None:
        e = _HARD["m"][G] if anim else _HARD["m"][N]
    return stem + e


def _ordinal_parts(n: int, ordinal_style: str, inverted: bool) -> Tuple[List[str], List[str]]:
    """(cardinal words kept as they are, ordinal lemmas to decline)."""
    comps = _components(n)
    card: List[str] = []
    lemmas: List[str] = []
    for k, (value, scale) in enumerate(comps):
        last = k == len(comps) - 1
        mixed_prefix = ordinal_style == "mixed" and not last
        if scale is not None:
            sg, pl, sgender = scale
            lemma = [s for s in SCALES if s[2] is sg][0][4]
            if mixed_prefix:
                card.append(_count_noun(value, sg, pl, sgender, N, "genitive"))
            elif value == 1:
                lemmas.append(lemma)
            elif value < 100 or value == 100:
                lemmas.append(_combining(value) + lemma)
            else:  # 345 000th and similar: cardinal count + ordinal scale word
                card.append(_below_1000(value, N, sgender, "inanimate", "genitive"))
                lemmas.append(lemma)
        elif value >= 100:
            if mixed_prefix:
                card.append(HUNDREDS[value // 100][N])
            else:
                lemmas.append(ORDINALS[value])
        elif value in ORDINALS:
            lemmas.append(ORDINALS[value])
        elif inverted:  # jednadvacátý, pětadvacátý
            lemmas.append(INVERTED_UNIT[value % 10] + "a" + ORDINALS[value - value % 10])
        else:
            lemmas += [ORDINALS[value - value % 10], ORDINALS[value % 10]]
    return card, lemmas


def int_to_ordinal(n: int, gender: str = "masculine", case: str = "nominative",
                   animacy: str = "inanimate", plural: bool = False,
                   ordinal_style: str = "mixed", inverted: bool = False) -> str:
    _check(gender, animacy)
    if n < 0:
        raise ValueError("ordinal numbers must be >= 0")
    c = _idx(case)
    if n == 0:
        return decline_ordinal(ORDINALS[0], c, gender, animacy, plural)
    card, lemmas = _ordinal_parts(n, ordinal_style, inverted)
    return " ".join(card + [decline_ordinal(l, c, gender, animacy, plural) for l in lemmas])


# --------------------------------------------------------------------------------------
# Decimals: "jedna celá pět desetin", "tři celé čtrnáct setin"
# --------------------------------------------------------------------------------------

DENOMINATORS = {
    1: ("desetina", "desetiny", "desetin"), 2: ("setina", "setiny", "setin"),
    3: ("tisícina", "tisíciny", "tisícin"), 4: ("desetitisícina", "desetitisíciny", "desetitisícin"),
    5: ("stotisícina", "stotisíciny", "stotisícin"), 6: ("miliontina", "miliontiny", "miliontin"),
}


def _count_with_noun(v: int, forms: Tuple[str, str, str], agreement: bool) -> str:
    """Feminine count + noun: jedna celá / dvě celé / pět celých (agreement for compounds)."""
    if v == 0:
        return f"nula {forms[0]}"  # nula celá (IJP)
    last2 = v % 100
    tail = last2 % 10 if last2 > 20 or v > 100 else last2
    construction = "agreement" if agreement else "genitive"
    num = int_to_cardinal(v, "feminine", "nominative", construction=construction)
    if tail == 1 and (v == 1 or agreement):
        return f"{num} {forms[0]}"
    if tail in (2, 3, 4) and (v < 5 or agreement):
        return f"{num} {forms[1]}"
    return f"{num} {forms[2]}"


def float_to_cardinal(x, gender: str = "masculine", case: str = "nominative", **_) -> str:
    """Decimals are read in the nominative; "celá" agrees with the integer part."""
    try:
        d = Decimal(str(x).replace(",", "."))
    except InvalidOperation:
        raise ValueError(f"not a number: {x!r}")
    sign = MINUS + " " if d < 0 else ""
    d = abs(d)
    whole = int(d)
    text = format(d, "f")
    frac = text.split(".")[1].rstrip("0") if "." in text else ""
    if not frac:
        return sign + int_to_cardinal(whole, gender, case)
    integer = _count_with_noun(whole, ("celá", "celé", "celých"), agreement=True)
    if len(frac) > 6:
        return f"{sign}{integer} " + " ".join(int_to_cardinal(int(ch), "feminine") for ch in frac)
    return f"{sign}{integer} {_count_with_noun(int(frac), DENOMINATORS[len(frac)], agreement=False)}"


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------

def _options(kwargs: dict) -> dict:
    animacy = kwargs.get("animacy")
    if animacy is None:
        animacy = "animate" if kwargs.get("animate") else "inanimate"
    return {"gender": kwargs.get("gender", "masculine"), "case": kwargs.get("case", "nominative"),
            "animacy": animacy}


class Num2Word_CS:
    """Czech number to words converter (v1-compatible class interface)."""

    negword = MINUS

    def to_cardinal(self, number, **kwargs) -> str:
        o = _options(kwargs)
        if isinstance(number, (float, Decimal)) or (isinstance(number, str) and any(s in number for s in ".,")):
            return float_to_cardinal(number, o["gender"], o["case"])
        return int_to_cardinal(int(number), o["gender"], o["case"], o["animacy"],
                               kwargs.get("construction", "genitive"), kwargs.get("oblique_style", "auto"))

    def to_ordinal(self, number, **kwargs) -> str:
        o = _options(kwargs)
        return int_to_ordinal(int(number), o["gender"], o["case"], o["animacy"],
                              kwargs.get("plural", False), kwargs.get("ordinal_style", "mixed"),
                              kwargs.get("inverted", False))


def num2words(number, to: str = "cardinal", **kwargs) -> str:
    """
    Convert a number to Czech words.
        to: "cardinal" | "ordinal"
        gender: "masculine" | "feminine" | "neuter"                 (default "masculine")
        case: "nominative" | "genitive" | "dative" | "accusative" | "instrumental" | "locative"
        animacy: "inanimate" | "animate" | "personal"                (masculine; default "inanimate")
        construction: "genitive" (dvacet jedna žáků, default) | "agreement" (dvacet jeden žák)
                      | "inverted" (jednadvacet žáků)
        oblique_style: "auto" | "full" | "partial" — how much of a long number declines
        plural: ordinal in the plural (první, druzí, druhých ...)
        ordinal_style: "mixed" (sto dvacátý první, default) | "all" (stý dvacátý první)
        inverted: inverted ordinals (jednadvacátý)
    """
    converter = Num2Word_CS()
    if to == "ordinal":
        return converter.to_ordinal(number, **kwargs)
    if to != "cardinal":
        raise ValueError("to must be 'cardinal' or 'ordinal'")
    return converter.to_cardinal(number, **kwargs)
