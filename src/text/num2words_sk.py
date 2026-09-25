# -*- coding: utf-8 -*-
"""
Slovak numbers to words — cardinals, ordinals and decimals with case, gender and animacy,
for TTS text normalisation.

v2 (25 September 2026): rewritten from a sourced normative spec (Morfológia slovenského jazyka
1966, Pravidlá slovenského pravopisu 1991/1998, Navrátil 2003, Beliana, JÚĽŠ SAV language
columns; see NUM2WORDS_CHANGES.md). v1 produced ORDINAL forms wherever a cardinal should be
declined (21 G "dvadsiatehojedného", 100 G "stého", 1000 G "tisíceho", 2 000 000 G
"dvamiliónteho"); those are gone.

Main rules implemented
  * 1-4 decline by gender; 2-4 have masculine PERSONAL forms (dvaja/traja/štyria, A = G).
    Instrumental: dvoma/troma with masculine and neuter nouns, dvomi/tromi with feminine nouns
    ("dvoma stromami", "dvomi stenami"; both forms are codified), štyrmi for all.
  * 5-99 decline like "päť" (piatich, piatim, piatimi); masculine personal N "piati" is an
    optional congruent form (construction="agreement").
  * compounds ending in -jeden (21, 101, 1001 ...) NEVER decline and use "jeden" for all genders;
    tens + 2 use the masculine "dva" for all genders ("dvadsaťdva žien", "dvadsaťdvatisíc"), but a
    bare 2 after sto-/tisíc- agrees like a simple "dve": "stodve knihy" (native review).
  * 22-99: both parts decline as cardinals, units written apart: "dvadsiatich dvoch"
    (declined=False keeps every compound undeclined — "dvadsaťdva", "stodva" — also standard).
  * sto, dvesto ... deväťsto and every -tisíc compound are INVARIABLE with counted nouns;
    in larger numbers only the final tens and units decline ("tristoštyridsiatich piatich").
  * milión/miliarda are nouns: separate words, both parts decline ("dvoch miliónov"), also when
    a smaller part follows ("dvoch miliónov piatich", native review); sto-/-tisíc parts stay
    invariable and the final tens and units decline.
  * ordinals: thousands and hundreds stay CARDINAL prefixes, only tens and units are ordinal:
    "stoprvý", "stodvadsiaty prvý", "dvetisícdvadsiaty štvrtý"; bare 200th/2000th "dvestý",
    "dvetisíci" (the frequent forms, chosen in native review; codified=True gives the codified
    "dvojstý", "dvojtisíci"). Rhythmic law applied (piateho, miliónteho, tisíca).
  * decimals agree with "celá": "jedna celá päť desatín", "tri celé štrnásť stotín".
  * nula declines as a noun (nuly, nule, nulu, nulou).

API (compatible with v1):
    num2words(number, to="cardinal"|"ordinal", gender="masculine"|"feminine"|"neuter",
              case="nominative"|"genitive"|"dative"|"accusative"|"instrumental"|"locative",
              animacy="inanimate"|"animate"|"personal", plural=False,
              construction="genitive"|"agreement", declined=True, codified=False)
`animacy` only matters for the masculine: "animate" (incl. animals) changes the accusative of
jeden and of ordinals; "personal" (people) also selects dvaja/traja/štyria and plural forms.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import List, Optional, Tuple

CASES = ["nominative", "genitive", "dative", "accusative", "instrumental", "locative"]
N, G, D, A, I, L = range(6)  # indices into CASES
GENDERS = ("masculine", "feminine", "neuter")
ANIMACY = ("inanimate", "animate", "personal")

ZERO_FORMS = ("nula", "nuly", "nule", "nulu", "nulou", "nule")
MINUS = "mínus"

# --------------------------------------------------------------------------------------
# Cardinal building blocks
# --------------------------------------------------------------------------------------

ONE = {
    "masculine": ("jeden", "jedného", "jednému", None, "jedným", "jednom"),  # A by animacy
    "feminine": ("jedna", "jednej", "jednej", "jednu", "jednou", "jednej"),
    "neuter": ("jedno", "jedného", "jednému", "jedno", "jedným", "jednom"),
}
# 2-4: (N other, N masc personal, G, D, I, L); A = N, or G for masculine personal
SMALL = {
    2: ("dva", "dvaja", "dvoch", "dvom", "dvoma", "dvoch"),
    3: ("tri", "traja", "troch", "trom", "troma", "troch"),
    4: ("štyri", "štyria", "štyroch", "štyrom", "štyrmi", "štyroch"),
}
# 5-20 and round tens: (nominative, oblique stem); endings G/L -ich, D -im, I -imi, pers. N -i
PAT = {
    5: ("päť", "piat"), 6: ("šesť", "šiest"), 7: ("sedem", "siedm"), 8: ("osem", "ôsm"),
    9: ("deväť", "deviat"), 10: ("desať", "desiat"), 11: ("jedenásť", "jedenást"),
    12: ("dvanásť", "dvanást"), 13: ("trinásť", "trinást"), 14: ("štrnásť", "štrnást"),
    15: ("pätnásť", "pätnást"), 16: ("šestnásť", "šestnást"), 17: ("sedemnásť", "sedemnást"),
    18: ("osemnásť", "osemnást"), 19: ("devätnásť", "devätnást"), 20: ("dvadsať", "dvadsiat"),
    30: ("tridsať", "tridsiat"), 40: ("štyridsať", "štyridsiat"), 50: ("päťdesiat", "päťdesiat"),
    60: ("šesťdesiat", "šesťdesiat"), 70: ("sedemdesiat", "sedemdesiat"),
    80: ("osemdesiat", "osemdesiat"), 90: ("deväťdesiat", "deväťdesiat"),
}
PAT_ENDINGS = {G: "ich", D: "im", I: "imi", L: "ich"}
HUNDREDS = {1: "sto", 2: "dvesto", 3: "tristo", 4: "štyristo", 5: "päťsto", 6: "šesťsto",
            7: "sedemsto", 8: "osemsto", 9: "deväťsto"}

# scale nouns: (exponent, gender, singular forms, plural forms); milión = vzor dub, miliarda = vzor žena
_M = lambda s: ((s, s + "a", s + "u", s, s + "om", s + "e"),
                (s + "y", s + "ov", s + "om", s + "y", s + "mi", s + "och"))
_F = lambda s, gpl: ((s + "a", s + "y", s + "e", s + "u", s + "ou", s + "e"),
                     (s + "y", gpl, s + "ám", s + "y", s + "ami", s + "ách"))
SCALES = [
    (27, "feminine", *_F("kvadriliard", "kvadriliárd")),
    (24, "masculine", *_M("kvadrilión")),
    (21, "feminine", *_F("triliard", "triliárd")),
    (18, "masculine", *_M("trilión")),
    (15, "feminine", *_F("biliard", "biliárd")),
    (12, "masculine", *_M("bilión")),
    (9, "feminine", *_F("miliard", "miliárd")),
    (6, "masculine", *_M("milión")),
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


def _simple(n: int, c: int, gender: str, animacy: str, construction: str, compound: bool) -> str:
    """
    1-20 and round tens in case c. compound=True: part of a larger number ("jeden" for every
    gender; for 21-99 the caller passes the masculine, so 22 is "dvadsaťdva" for every gender).
    """
    personal = gender == "masculine" and animacy == "personal"
    if n == 1:
        if compound:
            return "jeden"
        if c == A and gender == "masculine":
            return "jedného" if animacy != "inanimate" else "jeden"
        return ONE[gender][c]
    if n in SMALL:
        nom, pers, g, d, i, l = SMALL[n]
        if c in (N, A):
            if personal and (not compound or construction == "agreement"):
                return pers if c == N else g
            if n == 2 and gender != "masculine":
                return "dve"  # dve knihy; also after sto-/tisíc-: "stodve knihy" (native review)
            return nom
        if c == I and gender == "feminine" and n in (2, 3):
            return "dvomi" if n == 2 else "tromi"  # dvomi stenami, but dvoma stromami (native review)
        return {G: g, D: d, I: i, L: l}[c]
    nom, stem = PAT[n]
    if c in (N, A):
        if personal and construction == "agreement":
            return stem + ("i" if c == N else "ich")
        return nom
    return stem + PAT_ENDINGS[c]


def _below_100(n: int, c: int, gender: str, animacy: str, construction: str, declined: bool,
               compound: bool) -> Tuple[str, str]:
    """
    Returns (head, tail): `head` is joined to a preceding hundreds/thousands prefix, `tail`
    (possibly '') is a separate word — PSP 1991: declined units are written apart.
    """
    if n <= 20 or n % 10 == 0:
        if compound and not declined and c not in (N, A):  # undeclined compound: "so stodva žiakmi"
            return _simple(n, N, gender, "inanimate", "genitive", True), ""
        return _simple(n, c, gender, animacy, construction, compound), ""
    tens, unit = n - n % 10, n % 10
    if unit == 1:  # -jeden compounds never decline and never change gender
        return PAT[tens][0] + "jeden", ""
    personal = gender == "masculine" and animacy == "personal"
    oblique = c not in (N, A)
    congruent = personal and construction == "agreement"
    if (oblique and declined) or (congruent and c in (N, A)):
        return (_simple(tens, c, gender, animacy, construction, True),
                _simple(unit, c, gender, animacy, construction, True))
    return PAT[tens][0] + _simple(unit, N, "masculine", "inanimate", "genitive", True), ""


def _below_1000(n: int, c: int, gender: str, animacy: str, construction: str, declined: bool,
                prefix: str = "", in_compound: bool = False) -> str:
    """in_compound: the number follows a milión/miliarda group (1 000 005 = milión päť)."""
    h, rest = divmod(n, 100)
    prefix = prefix + (HUNDREDS[h] if h else "")
    if rest == 0:
        return prefix  # sto, dvesto, ... and -tisíc compounds are invariable
    compound = bool(prefix) or in_compound
    head, tail = _below_100(rest, c, gender, animacy, construction, declined, compound)
    personal = gender == "masculine" and animacy == "personal"
    if prefix and 1 < rest < 10 and ((c not in (N, A) and declined) or
                                     (personal and construction == "agreement")):
        # a declined or congruent unit after sto-/tisíc- is written apart: "sto piatich", "sto dvaja"
        return f"{prefix} {head}"
    word = prefix + head
    return f"{word} {tail}" if tail else word


def _thousands_prefix(t: int) -> str:
    """Invariable -tisíc prefix: tisíc, dvetisíc, päťtisíc, dvadsaťjedentisíc, stotisíc."""
    if t == 1:
        return "tisíc"
    if t == 2:
        return "dvetisíc"
    return _below_1000(t, N, "masculine", "inanimate", "genitive", False) + "tisíc"


def _scale_group(count: int, sg: tuple, pl: tuple, sgender: str, c: int, declined: bool) -> str:
    """count x milión/miliarda... in case c. Both parts decline (nouns, vzor dub / žena)."""
    if count == 1:
        return sg[c]
    if count in SMALL and count <= 4:
        num = _simple(count, c, sgender, "inanimate", "genitive", False)
        return f"{num} {pl[N] if c in (N, A) else pl[c]}"
    num = _below_1000(count, c, sgender, "inanimate", "genitive", declined)
    if c in (N, A):
        # G pl after 5+ and after tens (dvadsaťdva miliónov); after sto- + a bare 2-4 the noun
        # agrees as after a simple 2-4, like "stodve knihy" (native review): stodve miliardy
        noun = pl[N] if count % 100 in (2, 3, 4) else pl[G]
    else:
        noun = pl[c]
    return f"{num} {noun}"


def int_to_cardinal(n: int, gender: str = "masculine", case: str = "nominative",
                    animacy: str = "inanimate", construction: str = "genitive",
                    declined: bool = True) -> str:
    _check(gender, animacy)
    c = _idx(case)
    if n < 0:
        return f"{MINUS} {int_to_cardinal(-n, gender, case, animacy, construction, declined)}"
    if n == 0:
        return ZERO_FORMS[c]
    groups: List[str] = []
    rest = n
    big: List[Tuple[int, tuple, tuple, str]] = []
    for exp, sgender, sg, pl in SCALES:
        count, rest = divmod(rest, 10 ** exp)
        if count:
            if count >= 1000:
                raise ValueError("number too large")
            big.append((count, sg, pl, sgender))
    t, r = divmod(rest, 1000)
    has_tail = rest > 0
    # milión/miliarda groups are nouns and always decline, also before a smaller part:
    # "dvoch miliónov piatich", "miliarde päťsto miliónom" (native review)
    for count, sg, pl, sgender in big:
        groups.append(_scale_group(count, sg, pl, sgender, c, declined))
    if rest:
        prefix = _thousands_prefix(t) if t else ""
        if r == 0:
            groups.append(prefix)  # round thousands: invariable
        else:
            groups.append(_below_1000(r, c, gender, animacy, construction, declined, prefix,
                                      in_compound=bool(big)))
    return " ".join(groups)


# --------------------------------------------------------------------------------------
# Ordinals: adjective declension (vzor pekný / cudzí, with the rhythmic law)
# --------------------------------------------------------------------------------------

ORDINALS = {
    0: "nultý", 1: "prvý", 2: "druhý", 3: "tretí", 4: "štvrtý", 5: "piaty", 6: "šiesty",
    7: "siedmy", 8: "ôsmy", 9: "deviaty", 10: "desiaty", 11: "jedenásty", 12: "dvanásty",
    13: "trinásty", 14: "štrnásty", 15: "pätnásty", 16: "šestnásty", 17: "sedemnásty",
    18: "osemnásty", 19: "devätnásty", 20: "dvadsiaty", 30: "tridsiaty", 40: "štyridsiaty",
    50: "päťdesiaty", 60: "šesťdesiaty", 70: "sedemdesiaty", 80: "osemdesiaty",
    90: "deväťdesiaty", 100: "stý", 200: "dvojstý", 300: "trojstý", 400: "štvorstý",
    500: "päťstý", 600: "šesťstý", 700: "sedemstý", 800: "osemstý", 900: "deväťstý",
}
MULTIPLIER = {2: "dvoj", 3: "troj", 4: "štvor"}      # dvojstý, dvojtisíci, dvojmiliónty
UNCODIFIED = {2: "dve", 3: "troj", 4: "štvor"}       # dvestý, dvetisíci (frequent, not codified);
# *tritisíci and *štyritisíci are unattested (Šrámeková 2023), so 3 and 4 keep troj-/štvor-

_HARD_LONG = {  # after a short syllable: prvý, druhý, stý
    "m": ("ý", "ého", "ému", None, "ým", "om"), "f": ("á", "ej", "ej", "ú", "ou", "ej"),
    "n": ("é", "ého", "ému", "é", "ým", "om"),
    "pl": ("é", "ých", "ým", "é", "ými", "ých"), "pl_pers": ("í", "ých", "ým", "ých", "ými", "ých"),
}
_HARD_SHORT = {  # rhythmic law after a long syllable: piaty, miliónty
    "m": ("y", "eho", "emu", None, "ym", "om"), "f": ("a", "ej", "ej", "u", "ou", "ej"),
    "n": ("e", "eho", "emu", "e", "ym", "om"),
    "pl": ("e", "ych", "ym", "e", "ymi", "ych"), "pl_pers": ("i", "ych", "ym", "ych", "ymi", "ych"),
}
_SOFT_LONG = {  # tretí
    "m": ("í", "ieho", "iemu", None, "ím", "om"), "f": ("ia", "ej", "ej", "iu", "ou", "ej"),
    "n": ("ie", "ieho", "iemu", "ie", "ím", "om"),
    "pl": ("ie", "ích", "ím", "ie", "ími", "ích"), "pl_pers": ("í", "ích", "ím", "ích", "ími", "ích"),
}
_SOFT_SHORT = {  # tisíci (rhythmic law): tisíca, tisíce, tisíceho
    "m": ("i", "eho", "emu", None, "im", "om"), "f": ("a", "ej", "ej", "u", "ou", "ej"),
    "n": ("e", "eho", "emu", "e", "im", "om"),
    "pl": ("e", "ich", "im", "e", "imi", "ich"), "pl_pers": ("i", "ich", "im", "ich", "imi", "ich"),
}
_SOFTEN = {"t": "ť", "d": "ď", "n": "ň", "l": "ľ"}  # treťom, treťou


def decline_ordinal(lemma: str, c: int, gender: str, animacy: str, plural: bool) -> str:
    last = lemma[-1]
    table = {"ý": _HARD_LONG, "y": _HARD_SHORT, "í": _SOFT_LONG, "i": _SOFT_SHORT}[last]
    stem = lemma[:-1]
    personal = gender == "masculine" and animacy == "personal"
    if plural:
        ending = table["pl_pers" if personal else "pl"][c]
    else:
        key = {"masculine": "m", "feminine": "f", "neuter": "n"}[gender]
        ending = table[key][c]
        if ending is None:  # masculine accusative
            ending = table["m"][G] if animacy != "inanimate" else table["m"][N]
    if table is _SOFT_LONG and ending.startswith("o") and stem[-1] in _SOFTEN:
        stem = stem[:-1] + _SOFTEN[stem[-1]]
    return stem + ending


def _ordinal_lemmas(n: int, codified: bool) -> Tuple[str, List[str]]:
    """(cardinal prefix, ordinal lemmas to decline). The prefix is glued to the first lemma."""
    mult = MULTIPLIER if codified else UNCODIFIED
    rest = n
    parts: List[str] = []
    for exp, sgender, sg, pl in SCALES:
        count, rest = divmod(rest, 10 ** exp)
        if count and rest == 0:  # the scale group itself is the ordinal: miliónty, dvojmiliardtý
            base = sg[N][:-1] if sgender == "feminine" else sg[N]
            lemma = base + ("tý" if base.endswith("d") else "ty")  # miliardtý / miliónty
            if count == 1:
                pre = ""
            elif count in mult:
                pre = mult[count]
            else:
                pre = _below_1000(count, N, "masculine", "inanimate", "genitive", False)
            return (" ".join(parts) + " " if parts else ""), [pre + lemma]
        if count:
            parts.append(_scale_group(count, sg, pl, sgender, N, False))
    lead = (" ".join(parts) + " ") if parts else ""
    t, r = divmod(rest, 1000)
    if r == 0 and t:  # ...th thousand: tisíci, dvojtisíci, päťtisíci
        pre = "" if t == 1 else (mult[t] if t in mult else _thousands_prefix(t)[:-5])
        return lead, [pre + "tisíci"]
    prefix = _thousands_prefix(t) if t else ""
    h, tu = divmod(r, 100)
    if tu == 0:  # ...th hundred: stý, dvojstý, tisícstý
        lemma = "dvestý" if (h == 2 and not codified) else ORDINALS[h * 100]
        return lead + prefix, [lemma]
    prefix += HUNDREDS[h] if h else ""
    if tu in ORDINALS:
        return lead + prefix, [ORDINALS[tu]]
    return lead + prefix, [ORDINALS[tu - tu % 10], ORDINALS[tu % 10]]


def int_to_ordinal(n: int, gender: str = "masculine", case: str = "nominative",
                   animacy: str = "inanimate", plural: bool = False, codified: bool = False) -> str:
    _check(gender, animacy)
    if n < 0:
        raise ValueError("ordinal numbers must be >= 0")
    c = _idx(case)
    if n == 0:
        return decline_ordinal(ORDINALS[0], c, gender, animacy, plural)
    prefix, lemmas = _ordinal_lemmas(n, codified)
    words = [decline_ordinal(l, c, gender, animacy, plural) for l in lemmas]
    # PSP: the cardinal prefix is glued to the first ordinal word, the unit is a separate word
    return prefix + " ".join(words)


# --------------------------------------------------------------------------------------
# Decimals: "jedna celá päť desatín", "tri celé štrnásť stotín"
# --------------------------------------------------------------------------------------

DENOMINATORS = {  # (1, 2-4, 5+ and compounds)
    1: ("desatina", "desatiny", "desatín"), 2: ("stotina", "stotiny", "stotín"),
    3: ("tisícina", "tisíciny", "tisícin"), 4: ("desaťtisícina", "desaťtisíciny", "desaťtisícin"),
    5: ("stotisícina", "stotisíciny", "stotisícin"), 6: ("milióntina", "milióntiny", "milióntin"),
}


def _count_with_noun(v: int, forms: Tuple[str, str, str]) -> str:
    """Feminine count + noun in the nominative: jedna celá / dve celé / päť celých."""
    if v == 0:
        return f"nula {forms[2]}"
    if v == 1:
        return f"jedna {forms[0]}"
    if v in (2, 3, 4):
        return f"{_simple(v, N, 'feminine', 'inanimate', 'genitive', False)} {forms[1]}"
    return f"{int_to_cardinal(v, 'feminine')} {forms[2]}"


def float_to_cardinal(x, gender: str = "masculine", case: str = "nominative", **_) -> str:
    """
    Decimal numbers are read in the nominative (no source covers oblique cases); the integer
    part agrees with the feminine "celá" and the fraction takes desatina/stotina/tisícina.
    """
    try:
        d = Decimal(str(x).replace(",", "."))
    except InvalidOperation:
        raise ValueError(f"not a number: {x!r}")
    sign = MINUS + " " if d < 0 else ""
    d = abs(d)
    whole = int(d)
    frac = format(d, "f").split(".")[1].rstrip("0") if "." in format(d, "f") else ""
    if not frac:
        return sign + int_to_cardinal(whole, gender, case)
    if len(frac) > 6:
        digits = " ".join(int_to_cardinal(int(ch), "feminine") for ch in frac)
        return f"{sign}{_count_with_noun(whole, ('celá', 'celé', 'celých'))} {digits}"
    return (f"{sign}{_count_with_noun(whole, ('celá', 'celé', 'celých'))} "
            f"{_count_with_noun(int(frac), DENOMINATORS[len(frac)])}")


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------

def _options(kwargs: dict) -> dict:
    animacy = kwargs.get("animacy")
    if animacy is None:
        animacy = "animate" if kwargs.get("animate") else "inanimate"
    return {"gender": kwargs.get("gender", "masculine"), "case": kwargs.get("case", "nominative"),
            "animacy": animacy}


class Num2Word_SK:
    """Slovak number to words converter (v1-compatible class interface)."""

    negword = MINUS

    def to_cardinal(self, number, **kwargs) -> str:
        o = _options(kwargs)
        if isinstance(number, (float, Decimal)) or (isinstance(number, str) and any(s in number for s in ".,")):
            return float_to_cardinal(number, o["gender"], o["case"])
        return int_to_cardinal(int(number), o["gender"], o["case"], o["animacy"],
                               kwargs.get("construction", "genitive"), kwargs.get("declined", True))

    def to_ordinal(self, number, **kwargs) -> str:
        o = _options(kwargs)
        return int_to_ordinal(int(number), o["gender"], o["case"], o["animacy"],
                              kwargs.get("plural", False), kwargs.get("codified", False))


def num2words(number, to: str = "cardinal", **kwargs) -> str:
    """
    Convert a number to Slovak words.
        to: "cardinal" | "ordinal"
        gender: "masculine" | "feminine" | "neuter"          (default "masculine")
        case: "nominative" | "genitive" | "dative" | "accusative" | "instrumental" | "locative"
        animacy: "inanimate" | "animate" | "personal"         (masculine only; default "inanimate")
        construction: "genitive" (dvadsaťdva žiakov) | "agreement" (dvadsiati dvaja žiaci)
        declined: decline 22-99 in oblique cases (True) or keep "dvadsaťdva" (False)
        plural: ordinal in the plural (prví, prvých ...)
        codified: the frequent dvestý / dvetisíci (False, default) or the codified dvojstý / dvojtisíci
    """
    converter = Num2Word_SK()
    if to == "ordinal":
        return converter.to_ordinal(number, **kwargs)
    if to != "cardinal":
        raise ValueError("to must be 'cardinal' or 'ordinal'")
    return converter.to_cardinal(number, **kwargs)
