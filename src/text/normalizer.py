"""
Czech / Slovak text normalisation for TTS (plan Step 2.2).

Outside <cs>/<sk>/<en> spans the output contains no digits and no symbols, because the G2P
raises on them. Every line break stays where it was, and a chapter heading keeps its leading
"# " (Prompt 2.3 segments this output).

Numbers are read through num2words in the case, gender and animacy of the governing noun or
preposition, as tagged by Stanza. A number without a governing word is read in the nominative
masculine inanimate, and the fallback is logged as a WARNING for native review.

Usage:
    norm = TextNormalizer("sk", book_config={"english": ["Harry Potter"], "num2words": {"codified": True}})
    norm.normalize("Prišli 2 muži.")   # 'Prišli dvaja muži.'
"""
from __future__ import annotations

import bisect
import logging
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from . import num2words_cs, num2words_sk

log = logging.getLogger(__name__)

# The treebanks behind these are CC BY-SA; Stanza's Czech "pdt" and "fictree" are non-commercial.
STANZA_PACKAGES = {"cs": "cac", "sk": "snk"}

CASES = num2words_cs.CASES
NOM, GEN, DAT, ACC, INS, LOC = CASES
_UD_CASES = {"Nom": NOM, "Gen": GEN, "Dat": DAT, "Acc": ACC, "Ins": INS, "Loc": LOC, "Voc": NOM}
_UD_GENDERS = {"Masc": "masculine", "Fem": "feminine", "Neut": "neuter"}

# Step 2.1 variant keywords a book config may set; num2words silently ignores unknown keywords
NUM2WORDS_VARIANTS = {"cs": {"construction", "oblique_style", "ordinal_style", "inverted"},
                      "sk": {"construction", "declined", "codified"}}

MONTHS_GENITIVE = {
    "cs": ("ledna", "února", "března", "dubna", "května", "června", "července", "srpna", "září",
           "října", "listopadu", "prosince"),
    "sk": ("januára", "februára", "marca", "apríla", "mája", "júna", "júla", "augusta", "septembra",
           "októbra", "novembra", "decembra"),
}


def _forms(stem: str, endings: str) -> Tuple[str, ...]:
    """Forms in CASES order: the singular, then (for NOUNS) the plural."""
    return tuple(stem + e for e in endings.split(","))


_CS_HRAD = ",u,u,,em,u,y,ů,ům,y,y,ech"
_CS_ZENA = "a,y,ě,u,ou,ě,y,,ám,y,ami,ách"
_CS_MESTO = "o,a,u,o,em,u,a,,ům,a,y,ech"
_SK_DUB = ",u,u,,om,e,y,ov,om,y,ami,och"
_SK_METER = "er,ra,ru,er,rom,ri,re,rov,rom,re,rami,roch"
_SK_MESTO = "o,a,u,o,om,e,á,,ám,á,ami,ách"
_INDECLINABLE = ",,,,,,,,,,,"

NOUNS = {  # gender and forms of every noun the normalizer writes after a number
    "cs": {
        "kilometr": ("masculine", _forms("kilometr", _CS_HRAD)),
        "metr": ("masculine", _forms("metr", _CS_HRAD)),
        "centimetr": ("masculine", _forms("centimetr", _CS_HRAD)),
        "milimetr": ("masculine", _forms("milimetr", _CS_HRAD)),
        "kilogram": ("masculine", _forms("kilogram", _CS_HRAD)),
        "gram": ("masculine", _forms("gram", _CS_HRAD)),
        "litr": ("masculine", _forms("litr", _CS_HRAD)),
        "mililitr": ("masculine", _forms("mililitr", _CS_HRAD)),
        "dolar": ("masculine", _forms("dolar", _CS_HRAD)),
        "cent": ("masculine", _forms("cent", _CS_HRAD)),
        "haléř": ("masculine", _forms("haléř", ",e,i,,em,i,e,ů,ům,e,i,ích")),
        "stupeň": ("masculine", _forms("stup", "eň,ně,ni,eň,něm,ni,ně,ňů,ňům,ně,ni,ních")),
        "koruna": ("feminine", _forms("korun", _CS_ZENA)),
        "hodina": ("feminine", _forms("hodin", _CS_ZENA)),
        "minuta": ("feminine", _forms("minut", _CS_ZENA)),
        "libra": ("feminine", _forms("lib", "ra,ry,ře,ru,rou,ře,ry,er,rám,ry,rami,rách")),
        "euro": ("neuter", _forms("eur", _CS_MESTO)),
        "procento": ("neuter", _forms("procent", _CS_MESTO)),
        "promile": ("neuter", _forms("promile", _INDECLINABLE)),
        "penny": ("feminine", _forms("pen", "ny,ny,ny,ny,ny,ny,ce,cí,cím,ce,cemi,cích")),
    },
    "sk": {
        "kilometer": ("masculine", _forms("kilomet", _SK_METER)),
        "meter": ("masculine", _forms("met", _SK_METER)),
        "centimeter": ("masculine", _forms("centimet", _SK_METER)),
        "milimeter": ("masculine", _forms("milimet", _SK_METER)),
        "liter": ("masculine", _forms("lit", _SK_METER)),
        "mililiter": ("masculine", _forms("mililit", _SK_METER)),
        "kilogram": ("masculine", _forms("kilogram", _SK_DUB)),
        "gram": ("masculine", _forms("gram", _SK_DUB)),
        "cent": ("masculine", _forms("cent", _SK_DUB)),
        "dolár": ("masculine", _forms("dolár", ",a,u,,om,i,e,ov,om,e,mi,och")),
        "halier": ("masculine", _forms("halier", ",a,u,,om,i,e,ov,om,e,mi,och")),
        "stupeň": ("masculine", _forms("stup", "eň,ňa,ňu,eň,ňom,ni,ne,ňov,ňom,ne,ňami,ňoch")),
        "koruna": ("feminine", _forms("kor", "una,uny,une,unu,unou,une,uny,ún,unám,uny,unami,unách")),
        "hodina": ("feminine", _forms("hod", "ina,iny,ine,inu,inou,ine,iny,ín,inám,iny,inami,inách")),
        "minúta": ("feminine", _forms("minút", "a,y,e,u,ou,e,y,,am,y,ami,ach")),
        "libra": ("feminine", _forms("lib", "ra,ry,re,ru,rou,re,ry,ier,rám,ry,rami,rách")),
        "euro": ("neuter", _forms("eur", _SK_MESTO)),
        "percento": ("neuter", _forms("percent", _SK_MESTO)),
        "promile": ("neuter", _forms("promile", _INDECLINABLE)),
        "penny": ("feminine", _forms("pen", "ny,ny,ny,ny,ny,ny,ce,cí,ciam,ce,cami,ciach")),
    },
}

UNITS = {  # symbol -> noun in NOUNS
    "cs": {"km": "kilometr", "km/h": "kilometr", "m": "metr", "m/s": "metr", "cm": "centimetr", "mm": "milimetr",
           "kg": "kilogram", "g": "gram", "l": "litr", "ml": "mililitr", "°C": "stupeň", "°": "stupeň",
           "%": "procento", "‰": "promile", "hod": "hodina", "min": "minuta", "Kč": "koruna",
           "€": "euro", "EUR": "euro", "$": "dolar", "USD": "dolar", "£": "libra"},
    "sk": {"km": "kilometer", "km/h": "kilometer", "m": "meter", "m/s": "meter", "cm": "centimeter",
           "mm": "milimeter",
           "kg": "kilogram", "g": "gram", "l": "liter", "ml": "mililiter", "°C": "stupeň", "°": "stupeň",
           "%": "percento", "‰": "promile", "hod": "hodina", "min": "minúta", "Kč": "koruna",
           "€": "euro", "EUR": "euro", "$": "dolár", "USD": "dolár", "£": "libra"},
}
UNIT_SUFFIXES = {"cs": {"km/h": " za hodinu", "m/s": " za sekundu", "°C": " Celsia"},
                 "sk": {"km/h": " za hodinu", "m/s": " za sekundu", "°C": " Celzia"}}
UNIT_ADJECTIVES = {"cs": {"²": "čtvereční", "³": "krychlový"}, "sk": {"²": "štvorcový", "³": "kubický"}}
MINOR_UNITS = {"cs": {"koruna": "haléř", "euro": "cent", "dolar": "cent", "libra": "penny"},
               "sk": {"koruna": "halier", "euro": "cent", "dolár": "cent", "libra": "penny"}}
SCALES = {"tis": 3, "mil": 6, "mld": 9}
# gender of the multiplier in "dva tisíce", "dve miliardy", sk "dvetisíc"
SCALE_GENDERS = {"cs": {"tis": "masculine", "mil": "masculine", "mld": "feminine"},
                 "sk": {"tis": "feminine", "mil": "masculine", "mld": "feminine"}}
SCALE_GENITIVES = {"cs": {"tis": "tisíce", "mil": "milionu", "mld": "miliardy"},
                   "sk": {"tis": "tisíca", "mil": "milióna", "mld": "miliardy"}}
SIGNS = {"cs": {"&": "a", "+": "plus", "@": "zavináč", "=": "rovná se", "×": "krát", "x": "krát",
               "±": "plus minus", "#": "číslo"},
         "sk": {"&": "a", "+": "plus", "@": "zavináč", "=": "rovná sa", "×": "krát", "x": "krát",
               "±": "plus mínus", "#": "číslo"}}
PER_UNITS = {  # "100 Kč/kg" -> "za kilogram": the unit after "/" in the accusative singular
    "cs": {"kg": "kilogram", "g": "gram", "l": "litr", "ml": "mililitr", "m": "metr", "km": "kilometr",
           "ks": "kus", "hod": "hodinu", "h": "hodinu", "min": "minutu", "s": "sekundu"},
    "sk": {"kg": "kilogram", "g": "gram", "l": "liter", "ml": "mililiter", "m": "meter", "km": "kilometer",
           "ks": "kus", "hod": "hodinu", "h": "hodinu", "min": "minútu", "s": "sekundu"},
}
SLASH_WORDS = {"cs": {"word": "nebo", "number": "lomeno"}, "sk": {"word": "alebo", "number": "lomené"}}
RANGE_WORD = "až"
# the feminine of an inclusive "Vážený/á", "přišel/a", "studenti/ky": (suffixes after the slash,
# endings of the masculine form, the feminine ending); any other suffix is appended: "on/a", "student/ka"
FEMININE_ENDINGS = [(("a", "á"), ("ý",), "á"), (("é",), ("í",), "é"), (("a",), ("šel", "šiel"), "šla"),
                    (("a",), ("sám",), "sama"), (("y",), ("i",), "y"), (("ky",), ("i", "é"), "ky"),
                    (("ce",), ("ník", "níci"), "nice"), (("čka",), ("k",), "čka"), (("čky",), ("ci",), "čky"),
                    (("yně",), ("a",), "yně")]

# Slovak animal plurals in -i that the tagger may mark animate like people; animals count with
# dva/tri/štyri, people with dvaja/traja/štyria
SK_ANIMAL_PLURALS = frozenset("vlci býci vtáci psi orli sokoli holubi levi tigri sloni barani kocúri "
                              "kohúti capi diviaci jeleni kanci ježkovia zajkovia macíkovia vtáčikovia "
                              "koníkovia psíčkovia kocúrikovia škrečkovia kohútikovia".split())

# forms the tagger misreads, with the features they have: UD Slovak-SNK takes "diel" (a part or
# volume, masculine) for feminine, even alone
FEATURE_FIXES = {"cs": {},
                 "sk": {"diel": {"Gender": "Masc", "Animacy": "Inan", "Number": "Sing", "Case": "Nom"},
                        "diely": {"Gender": "Masc", "Animacy": "Inan", "Number": "Plur"},
                        "dielov": {"Gender": "Masc", "Animacy": "Inan", "Number": "Plur", "Case": "Gen"}}}

# endings every plural noun has in these cases, in both languages (hradech, ženách, dubom, mužmi)
PLURAL_ENDINGS = {DAT: ("m",), INS: ("mi", "ma", "y", "i"), LOC: ("ch",)}
LOCATIVE_PLURAL_ENDINGS = ("ech", "ách", "och", "iach")  # only the locative plural ends so

# Clock times: Czech "ve čtrnáct třicet" is accusative, Slovak "o štrnástej" locative,
# whatever case the tagger gives the preposition.
TIME_PREPOSITIONS = {"cs": {"v": ACC, "ve": ACC}, "sk": {"o": LOC}}
# a Slovak clock time after these has an ordinal hour ("o druhej", "pred druhou"); a duration keeps the
# cardinal ("za dve pätnásť")
SK_CLOCK_PREPOSITIONS = frozenset({"o", "po", "pred", "okolo", "od", "do", "medzi", "k", "ku", "na"})

# preposition -> (vocalised form, starts of the number words that call for it)
VOCALISATION = {
    "cs": {"v": ("ve", ("v", "f", "dv", "tř", "čt", "st")), "k": ("ke", ("k", "g", "dv", "tř", "čt", "st")),
           "s": ("se", ("s", "z", "š", "ž", "dv", "tř", "čt")), "z": ("ze", ("s", "z", "š", "ž", "dv", "tř", "čt"))},
    "sk": {"v": ("vo", ("v", "f", "dv", "št")), "k": ("ku", ("k", "g")),
           "s": ("so", ("s", "z", "š", "ž")), "z": ("zo", ("s", "z", "š", "ž"))},
}

# abbreviations that introduce what follows, so their period never ends a sentence
NON_FINAL_ABBREVIATIONS = {"např.", "napr.", "tzn.", "tj.", "t.j.", "resp.", "cca.", "č.", "str.", "r.",
                           "mj.", "popř.", "příp.", "príp.", "zejm.", "vč.", "vr.", "max.", "sv.", "tzv.",
                           "odst.", "ods.", "písm."}
# keys whose capital form is an acronym, a name or initials: "TURNAJ ATP.", "TJ SOKOL", "voliči ODS."
CAPITAL_ACRONYMS = frozenset({"aj.", "atp.", "max.", "mj.", "n.l.", "ods.", "t.j.", "tj.", "vr."})
AGREEING_ABBREVIATIONS = {"sv.", "tzv."}  # adjectives: they take the case and gender of the next word
LABEL_ABBREVIATIONS = {"č.", "str.", "r.", "§", "odst.", "ods.", "písm."}  # the number after them names
# prepositions with the accusative or the locative; before a page, number or year the locative
# is meant: "na str. 45" -> "na straně"
LOCATIVE_PREPOSITIONS = {"cs": {"v", "ve", "na", "o", "po"}, "sk": {"v", "vo", "na", "o", "po"}}
DECLINED_ABBREVIATIONS = {  # nouns declined after a preposition: "v r. 1990" -> "v roce"
    "cs": {"č.": _forms("čísl", "o,a,u,o,em,e"), "str.": _forms("stran", "a,y,ě,u,ou,ě"),
           "r.": _forms("ro", "k,ku,ku,k,kem,ce"), "§": _forms("paragraf", ",u,u,,em,u"),
           "odst.": _forms("odstav", "ec,ce,ci,ec,cem,ci"), "písm.": _forms("písmen", "o,a,u,o,em,u")},
    "sk": {"č.": _forms("čísl", "o,a,u,o,om,e"), "str.": _forms("stran", "a,y,e,u,ou,e"),
           "§": _forms("paragraf", ",u,u,,om,e"), "ods.": _forms("odsek", ",u,u,,om,u"),
           "písm.": _forms("písmen", "o,a,u,o,om,e")},
}

_SPEAKABLE_PUNCT = frozenset(",.!?:;…—–-()[]\"'„“”‚‘’«»‹›`´*_~")
_QUOTES = frozenset("\"'„“”‚‘’«»‹›")
_MATH_SIGNS = frozenset("×=+±/")
_SPAN_RE = re.compile(r"<(cs|sk|en)>(.*?)</\1>", re.DOTALL)
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}

_HS = r"[ \t\u00a0\u202f]"  # horizontal space: no item may swallow a line break
_INT = r"[1-9]\d{0,2}(?:[ \u00a0\u202f]\d{3})+(?!\d)|[1-9]\d{0,2}(?:\.\d{3})+(?!\d)|\d+"  # 10 000, 10.000
# a minus sign starts after a space, bracket, quote or operator: „-5 °C“, "=-5"; not after a letter,
# digit or period: "COVID-19", "5-3", "1.-5."
_SIGN_START = r"(?<![^\s(\[{\"'„“”‚‘’«»‹›=:×/+])"
_EN_AMOUNT = r"\d{1,3}(?:(?:,\d{3}){2,}(?:\.\d+)?|,\d{3}\.\d+)(?!\d)"  # "1,234.56 USD": never a Czech decimal
_UNSIGNED = rf"(?:{_EN_AMOUNT}|(?:{_INT})(?:[.,]\d+)?)"
_AMOUNT = rf"(?:{_SIGN_START}[-−–](?=\d))?{_UNSIGNED}"  # "–5 °C": typeset text uses – for minus
_POWER = r"(?:[²³]|[23](?!\d))?"  # m², and m2 as typed
_PER = rf"kg|ks|km{_POWER}|ml|hod|min|g|l|m{_POWER}|h|s(?!{_HS}+[^\W\d_])"  # "Kč/m²", "m / s"; not "Kč / s DPH"
_EN_GROUPED = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?"  # "$1,234.56" after a prefixed currency symbol
_PRICE = rf"[-−]?(?:{_EN_GROUPED}|(?:{_INT})(?:[.,]\d+)?)"
_TAG_TOKEN = re.compile(rf"{_INT}|[^\W\d_]+|\S")  # "1 000" is one token: split, "000" misleads the tagger
_UNIT = rf"km/h|km{_POWER}|cm{_POWER}|mm{_POWER}|m/s|m{_POWER}|kg|g|ml|l|°C|°|%|‰|hod\.?|min\.?"
_CURRENCY = r"Kč|€|EUR|USD|\$|£"
_ROMAN = r"(?=[IVXLCDM])M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})"  # up to 3999
_NOT_LETTER_AFTER = r"(?![^\W\d_])"
_NOT_LETTER_BEFORE = r"(?<![^\W\d_])"
_ONE_LETTER_WORDS = "aikosuvz"
_INCLUSIVE_SUFFIXES = "kyně|yně|čka|čky|ka|ky|ce|a|á|é|y"  # "on/a", "Vážený/á", "student/ka", "přišli/y"
# "14.30" is a time only when hod follows, also after a second time: "15.30–16.00 hod.", "od 8.00 do 12.00 hod."
_HOUR_WORD = r"hod(?:\.|in[ay]?|ín)?"  # hod., hodin, hodiny, hodina, sk hodín
_DOT_TIME = (rf"(?=[0-5]\d(?:(?:{_HS}*[–—-]{_HS}*|{_HS}+do{_HS}+)(?:2[0-4]|[01]?\d)\.[0-5]\d)?"
             rf"{_HS}*{_HOUR_WORD}{_NOT_LETTER_AFTER})")
_SPACES = re.compile(f"{_HS}*")
# a number glued to an adjective is its first part: "25letý", "3denní", sk "5-ročný"
_ADJECTIVE_ENDINGS = "ieho|iemu|ého|ému|ých|ými|ími|ích|ém|ým|ím|om|ou|ej|ia|ie|iu|ý|á|é|í|ú"
_LETTER_BEFORE = re.compile(rf"{_NOT_LETTER_BEFORE}([^\W\d_]){_HS}+$")  # "s 2", also with a no-break space
_NUMBER_BEFORE = re.compile(rf"(?:\d\.?|[IVXLCDM]\.|\d{_HS}*{_HOUR_WORD})$")  # a dash between these reads "až"
_NUMBER_AFTER = re.compile(r"\d|[IVXLCDM]+\.")
_RANGE_AHEAD = re.compile(rf"{_HS}*[–—-]{_HS}*(?:[-−]?\d|[IVXLCDM]+\.)")


@lru_cache(maxsize=None)
def _tagger(language: str):
    import stanza

    return stanza.Pipeline(language, package=STANZA_PACKAGES[language], processors="tokenize,pos",
                           tokenize_pretokenized=True, download_method=stanza.DownloadMethod.REUSE_RESOURCES,
                           logging_level="WARN")


def _abbreviation_key(text: str) -> str:
    return re.sub(rf"\.{_HS}+", ".", re.sub(rf"{_HS}+", " ", text))


def _abbreviation_pattern(key: str) -> str:
    out = []
    for i, ch in enumerate(key):
        if ch == " ":
            out.append(f"{_HS}+")
        elif ch == "." and i < len(key) - 1:
            out.append(rf"\.{_HS}*")
        elif len(key) > 2 and ch.isalpha():
            out.append(f"[{ch}{ch.upper()}]")
        elif i == 0 and ch.isalpha():
            out.append(rf"(?:{ch}|{ch.upper()}(?={re.escape(key[1:])}{_HS}*\d))")  # "Č. 5", but "Č. Novák"
        else:
            out.append(re.escape(ch))
    return _NOT_LETTER_BEFORE + "".join(out) + (_NOT_LETTER_AFTER if key[-1].isalpha() else "")


def _items_pattern(abbreviations) -> re.Pattern:
    abbr = "|".join(_abbreviation_pattern(k) for k in sorted(abbreviations, key=len, reverse=True))
    return re.compile(
        rf"(?P<isodate>(?<![\d.,-])(?P<isoyear>\d{{4}})-(?P<isomonth>0[1-9]|1[0-2])-(?P<isoday>0[1-9]|[12]\d|3[01])(?![\d-]))"
        rf"|(?P<date>(?<!\d)(?P<day>3[01]|[12]\d|0?[1-9])\.{_HS}*(?P<month>1[0-2]|0?[1-9])\."
        rf"(?:{_HS}*(?P<year>\d{{4}})(?!\d)|{_HS}*(?P<shortyear>(?<=\.)\d{{2}}"
        rf"|(?<={_HS})(?:0\d|\d{{2}}(?!\d)(?!{_HS}*[^\W\d_])))(?!\d))?)"  # "5. 6. 05", not "5. 6. 24 lidí"
        rf"|(?P<time>(?<![\d.,:])(?P<hour>2[0-4]|[01]?\d)(?::|\.{_DOT_TIME})(?P<minute>[0-5]\d)(?::(?P<second>[0-5]\d))?(?![\d:])"
        rf"(?:{_HS}*{_HOUR_WORD}{_NOT_LETTER_AFTER})?)"
        rf"|(?P<range>(?<![\d.,])(?P<low>(?:{_SIGN_START}[-−–])?{_UNSIGNED})"
        rf"(?:{_HS}*(?:(?P<lowscale>tis|mil|mld)\.?{_NOT_LETTER_AFTER}"
        rf"(?:{_HS}+(?P<lowscaleunit>{_UNIT}|{_CURRENCY}){_NOT_LETTER_AFTER}"
        rf"(?:{_HS}*/{_HS}*(?P<lowscaleper>{_PER}){_NOT_LETTER_AFTER})?)?"
        rf"|(?P<lowunit>{_UNIT}|{_CURRENCY}){_NOT_LETTER_AFTER}"
        rf"(?:{_HS}*/{_HS}*(?P<lowper>{_PER}){_NOT_LETTER_AFTER})?))?{_HS}*[–—-]{_HS}*"
        rf"(?P<high>[-−]?{_UNSIGNED})"
        rf"(?(lowscale){_HS}*(?P<highscale>tis|mil|mld)\.?{_NOT_LETTER_AFTER}"
        rf"(?:{_HS}+(?P<highscaleunit>{_UNIT}|{_CURRENCY}){_NOT_LETTER_AFTER}"
        rf"(?:{_HS}*/{_HS}*(?P<highscaleper>{_PER}){_NOT_LETTER_AFTER})?)?"
        rf"|(?(lowunit){_HS}*(?P<highunit>{_UNIT}|{_CURRENCY}){_NOT_LETTER_AFTER}"
        rf"(?:{_HS}*/{_HS}*(?P<highper>{_PER}){_NOT_LETTER_AFTER})?"
        rf"|(?:{_HS}*(?:(?P<rangescale>tis|mil|mld)\.?{_NOT_LETTER_AFTER}"
        rf"(?:{_HS}+(?P<rangescaleunit>{_UNIT}|{_CURRENCY}){_NOT_LETTER_AFTER}"
        rf"(?:{_HS}*/{_HS}*(?P<rangescaleper>{_PER}){_NOT_LETTER_AFTER})?)?"
        rf"|(?P<rangeunit>{_UNIT}|{_CURRENCY}){_NOT_LETTER_AFTER}"
        rf"(?:{_HS}*/{_HS}*(?P<rangeper>{_PER}){_NOT_LETTER_AFTER})?))?)))"
        rf"|(?P<money>(?P<moneysign>{_SIGN_START}[-−–])?(?P<symbol>[€$£]){_HS}*(?P<price>{_PRICE})"
        rf"(?:{_HS}*/{_HS}*(?P<lowmoneyper>{_PER}){_NOT_LETTER_AFTER}{_HS}*[–—-]{_HS}*(?:(?P=symbol){_HS}*)?"
        rf"(?P<highprice>{_PRICE}){_HS}*/{_HS}*(?P<highmoneyper>{_PER}){_NOT_LETTER_AFTER}"
        rf"|(?:(?:{_HS}+(?P<lowmoneyscale>tis|mil|mld)\.?{_NOT_LETTER_AFTER})?{_HS}*[–—-]{_HS}*"
        rf"(?:(?P=symbol){_HS}*)?(?P<pricehigh>{_PRICE}))?"
        rf"(?(lowmoneyscale){_HS}+(?P<highmoneyscale>tis|mil|mld)\.?{_NOT_LETTER_AFTER}"
        rf"|(?:{_HS}+(?P<moneyscale>tis|mil|mld)\.?{_NOT_LETTER_AFTER})?)"
        rf"(?:{_HS}*/{_HS}*(?P<moneyper>{_PER}){_NOT_LETTER_AFTER})?))"
        rf"|(?P<measure>(?P<amount>{_AMOUNT})(?P<whole>,[-–—])?{_HS}*"
        rf"(?:(?P<scale>tis|mil|mld)\.?(?:{_HS}+(?P<scaleunit>{_UNIT}|{_CURRENCY}){_NOT_LETTER_AFTER}"
        rf"(?:{_HS}*/{_HS}*(?P<scaleper>{_PER}){_NOT_LETTER_AFTER})?)?"
        rf"|(?P<unit>{_UNIT}|{_CURRENCY})"
        rf"(?:{_HS}*/{_HS}*(?P<per>{_PER}){_NOT_LETTER_AFTER})?)"
        rf"{_NOT_LETTER_AFTER})"
        rf"|(?P<ordinal>(?<![\d.,])(?P<ordinalvalue>{_INT})\.(?={_HS}*(?:[^\W\d_]|[–—-]{_HS}*\d|,{_HS}*\d+\.)))"
        rf"|(?P<number>(?P<value>{_AMOUNT})(?:(?P<times>krát|x|×){_NOT_LETTER_AFTER}(?!{_HS}*\d)"
        rf"|-?(?P<compound>[^\W\d_]*?(?:{_ADJECTIVE_ENDINGS})){_NOT_LETTER_AFTER})?)"
        rf"|(?P<abbreviation>{abbr})"
        rf"|(?P<roman>{_NOT_LETTER_BEFORE}(?P<numeral>{_ROMAN})\.)"
        rf"|(?P<sign>[&+@=×±]|#(?={_HS}*\d)"
        rf"|(?:(?<=\d)|(?<=\d{_HS})|(?<=\d{_HS}{_HS})|(?<=\d{_HS}{_HS}{_HS}))x{_NOT_LETTER_AFTER})"  # "3 x 4", "3 x týdně"
        rf"|(?P<slash>(?<=[^\W\d_]{{2}})/(?P<suffix>{_INCLUSIVE_SUFFIXES}){_NOT_LETTER_AFTER}"
        rf"|(?:(?<=[^\W\d_]{{2}})|(?<={_NOT_LETTER_BEFORE}[{_ONE_LETTER_WORDS}{_ONE_LETTER_WORDS.upper()}]))"
        rf"{_HS}*/{_HS}*(?=[^\W\d_]{{2}})|(?<=\d){_HS}*/{_HS}*(?=\d))"
        rf"|(?P<dash>–|—|(?<!\S)-(?!\S)|(?<=\d)-(?=\d)|(?<=\d\.)-(?=\d)|(?<=[IVXLCDM]\.)-(?=[IVXLCDM]+\.))"
        rf"|(?P<ellipsis>\.\.\.)"
    )


def _unspeakable(text: str) -> Optional[str]:
    """The first character the G2P would raise on, if any."""
    for ch in text:
        if not (ch.isalpha() or ch.isspace() or ch in _SPEAKABLE_PUNCT or unicodedata.category(ch)[0] == "M"):
            return ch
    return None


def _parse(amount: str) -> Tuple[object, int, str]:
    """(value for num2words, absolute integer part, fraction digits without trailing zeros)."""
    s = re.sub(r"[ \u00a0\u202f]", "", amount).replace("−", "-").replace("–", "-")
    if re.fullmatch(rf"-?{_EN_AMOUNT}", s):
        s = s.replace(",", "")
    grouped = re.fullmatch(r"(-?[1-9]\d{0,2}(?:\.\d{3})+)(,\d+)?", s)  # "10.000", "10.000,50"; not "0.500"
    if grouped:
        s = grouped.group(1).replace(".", "") + (grouped.group(2) or "")
    whole, _, fraction = s.replace(".", ",").partition(",")
    fraction = fraction.rstrip("0")
    return (f"{whole},{fraction}" if fraction else int(whole)), abs(int(whole)), fraction


def _negative_zero(amount: str) -> bool:
    """"−0 °C", "-0,0": a zero written with a minus, which num2words reads as plain "nula"."""
    return re.fullmatch(r"[-−–]0+(?:[.,]0+)?", amount) is not None


def _plain_price(price: str) -> str:
    """"1,234.56" after $ or £ groups thousands with commas."""
    return price.replace(",", "") if re.fullmatch(rf"[-−]?{_EN_GROUPED}", price) else price


def _roman_value(numeral: str) -> int:
    values = [_ROMAN_VALUES[ch] for ch in numeral]
    return sum(-v if i + 1 < len(values) and v < values[i + 1] else v for i, v in enumerate(values))


def _spaced(text: str, start: int, end: int, word: str) -> str:
    """`word` in place of text[start:end], with a space on each side that lacks one."""
    left = "" if start == 0 or text[start - 1].isspace() else " "
    right = "" if end == len(text) or text[end].isspace() or text[end] in ".,;:!?…)]}“”’»›\"'" else " "
    return f"{left}{word}{right}"


def _key_of(m: re.Match) -> str:
    """The table key of a matched abbreviation: "Např." -> "např.", "t. j." -> "t.j."."""
    return _abbreviation_key(m.group(0).lower())


def _capitalise_like(source: str, words: str) -> str:
    return words[0].upper() + words[1:] if source[:1].isupper() else words


def _starts_sentence(before: str) -> bool:
    """Whether text after `before` (the paragraph up to it) starts a sentence, also direct speech
    after a colon: Řekl: „Pět…“."""
    before = _SPAN_RE.sub(lambda s: s.group(2), before).rstrip(" \t\n\r\u00a0\u202f")
    quoted = before[-1:] in "„“\"'‚‘«»‹›"
    before = before.rstrip(" \t\n\r\u00a0\u202f\"'„“”‚‘’«»‹›([—–-")
    return not before or before[-1] in ".!?…" or (quoted and before[-1] == ":")


@dataclass
class _Word:
    start: int
    end: int
    text: str
    upos: str
    feats: Dict[str, str]


class _Tags:
    """Tagger output with character offsets into the paragraph."""

    def __init__(self, words: List[_Word]):
        self.words = words
        self.starts = [w.start for w in words]

    def before(self, pos: int, skip: Tuple[str, ...] = ()) -> Optional[_Word]:
        i = bisect.bisect_left(self.starts, pos) - 1
        while i >= 0 and self.words[i].upos in skip:
            i -= 1
        return self.words[i] if i >= 0 else None

    def after(self, pos: int) -> Optional[_Word]:
        i = bisect.bisect_left(self.starts, pos)
        return next((w for w in self.words[i:] if w.upos != "PUNCT"), None)

    def head_after(self, pos: int) -> Optional[_Word]:
        """The noun that a number or an adjective at `pos` counts or agrees with; an adverb may
        modify an adjective in between: "2 velmi staré knihy", but not the noun: "Vrátil 2 zpátky knihovně"."""
        i = bisect.bisect_left(self.starts, pos)
        adverb = False
        for w in self.words[i:i + 5]:
            if w.upos in ("NOUN", "PROPN"):
                return None if adverb else w
            if w.upos in ("ADV", "PART"):
                adverb = True
            elif w.upos in ("ADJ", "DET"):
                adverb = False
            elif w.text not in _QUOTES:
                return None
        return None


class TextNormalizer:
    # v1 tables (plan v1, Prompt 2.3), extended in v2
    ABBREVIATIONS_CS = {
        "např.": "například", "tzn.": "to znamená", "atd.": "a tak dále", "apod.": "a podobně",
        "tj.": "to jest", "resp.": "respektive", "cca": "cirka", "cca.": "cirka", "č.": "číslo",
        "str.": "strana", "r.": "roku", "tis.": "tisíc", "mil.": "milion", "mld.": "miliarda",
        "aj.": "a jiné", "atp.": "a tak podobně", "mj.": "mimo jiné", "popř.": "popřípadě",
        "příp.": "případně", "zejm.": "zejména", "vč.": "včetně", "max.": "maximálně",
        "př. n. l.": "před naším letopočtem", "n. l.": "našeho letopočtu", "stol.": "století",
        "sv.": "svatý", "tzv.": "takzvaný", "a/nebo": "a nebo", "§": "paragraf", "odst.": "odstavec",
        "písm.": "písmeno",
    }
    ABBREVIATIONS_SK = {
        "napr.": "napríklad", "tzn.": "to znamená", "atď.": "a tak ďalej", "a pod.": "a podobne",
        "t.j.": "to jest", "resp.": "respektíve", "cca": "cirka", "č.": "číslo", "str.": "strana",
        "r.": "roku", "tis.": "tisíc", "mil.": "milión", "mld.": "miliarda",
        "tj.": "to jest", "atp.": "a tak podobne", "a i.": "a iné", "príp.": "prípadne",
        "vr.": "vrátane", "max.": "maximálne", "pred n. l.": "pred naším letopočtom",
        "n. l.": "nášho letopočtu", "stor.": "storočie", "sv.": "svätý", "tzv.": "takzvaný",
        "a/alebo": "a alebo", "§": "paragraf", "ods.": "odsek", "písm.": "písmeno",
    }

    def __init__(self, language: str, book_config: Optional[dict] = None):
        if language not in STANZA_PACKAGES:
            raise ValueError(f"language must be one of {sorted(STANZA_PACKAGES)}, got {language!r}")
        config = book_config or {}
        self.language = language
        self.variants = dict(config.get("num2words") or {})
        unknown = set(self.variants) - NUM2WORDS_VARIANTS[language]
        if unknown:
            raise ValueError(f"book_config['num2words'] has keywords {sorted(unknown)} that {language} "
                             f"num2words does not take; allowed: {sorted(NUM2WORDS_VARIANTS[language])}")
        self._numbers = num2words_cs if language == "cs" else num2words_sk
        table = self.ABBREVIATIONS_CS if language == "cs" else self.ABBREVIATIONS_SK
        self.abbreviations = {_abbreviation_key(k): v for k, v in table.items()}
        self._items = _items_pattern(self.abbreviations)
        phrases = sorted(filter(None, config.get("english") or []), key=len, reverse=True)
        self._english = (re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, phrases)) + r")(?!\w)")
                         if phrases else None)

    # ---- public API ----------------------------------------------------------------------
    def normalize(self, text: str) -> str:
        lines = text.split("\n")
        i = 0
        while i < len(lines):
            if lines[i].startswith("# "):
                lines[i] = "# " + self._paragraph(lines[i][2:], heading=True)
                i += 1
            elif not lines[i].strip():
                i += 1
            else:
                j = i
                while j < len(lines) and lines[j].strip() and not lines[j].startswith("# "):
                    j += 1
                lines[i:j] = self._paragraph("\n".join(lines[i:j])).split("\n")
                i = j
        return "\n".join(lines)

    # ---- paragraphs ------------------------------------------------------------------------
    def _paragraph(self, text: str, heading: bool = False) -> str:
        if self._english:
            text = self._wrap_english(text)
        spans = []
        for s in _SPAN_RE.finditer(text):
            bad = _unspeakable(s.group(2))
            if bad:
                raise ValueError(f"{bad!r} in {s.group(0)!r}: spans are kept as written, so spell it out")
            spans.append(s.span())
        items = [m for m in self._items.finditer(text)
                 if not any(s < m.end() and m.start() < e for s, e in spans)]
        tags = _Tags(self._tag(text) if any(self._needs_tags(m) for m in items) else [])
        out, pos, after_label = [], 0, -1
        for m in items:
            start, words = self._resolve(m, text, tags, after_label == m.start(), heading)
            source_case = m.lastgroup == "abbreviation" and m.group(0)[0].isalpha()
            if start == m.start() and not source_case and _starts_sentence("".join(out) + text[pos:start]):
                words = words[:1].upper() + words[1:]  # "5 lidí přišlo." -> "Pět lidí přišlo."
            if text[m.end():m.end() + 1].isalnum() and not words[-1:].isspace():
                words += " "  # "§5", "č.5", "5.díl"
            if start == m.start() and text[start - 1:start].isalnum() and words[:1].isalnum():
                words = " " + words  # "v14:30", "dne1.1.2024"
            if start == pos and out and out[-1].endswith(" ") and words.startswith(" "):
                words = words[1:]  # "3x4", "Cca5": both replacements brought a space
            out += [text[pos:start], words]
            pos = m.end()
            if m.group(0) == "#" or (m.lastgroup == "abbreviation" and _key_of(m) in LABEL_ABBREVIATIONS):
                after_label = _SPACES.match(text, pos).end()  # a label follows: "č. 5", "§ 7", "#1"
        out.append(text[pos:])
        result = "".join(out)
        bad = _unspeakable(_SPAN_RE.sub(" ", result))
        if bad:
            raise ValueError(f"no reading for {bad!r} in {text!r}")
        return result

    def _wrap_english(self, text: str) -> str:
        spans = [s.span() for s in _SPAN_RE.finditer(text)]
        out, pos = [], 0
        for m in self._english.finditer(text):
            if not any(s < m.end() and m.start() < e for s, e in spans):
                out += [text[pos:m.start()], f"<en>{m.group(0)}</en>"]
                pos = m.end()
        return "".join(out) + text[pos:]

    def _needs_tags(self, m: re.Match) -> bool:
        if m.lastgroup == "abbreviation":
            return _key_of(m) in AGREEING_ABBREVIATIONS or _key_of(m) in DECLINED_ABBREVIATIONS[self.language]
        return m.lastgroup not in ("date", "sign", "slash", "dash", "ellipsis")

    def _tag(self, text: str) -> List[_Word]:
        view = _SPAN_RE.sub(lambda s: " " * (s.start(2) - s.start()) + s.group(2) + " " * (s.end() - s.end(2)),
                            text)
        tokens = list(_TAG_TOKEN.finditer(view))
        if not tokens:
            return []
        sentence = _tagger(self.language)([[t.group() for t in tokens]]).sentences[0]
        words = []
        for t, token in zip(tokens, sentence.tokens):
            for w in token.words:
                feats = dict(f.split("=", 1) for f in (w.feats or "").split("|") if f)
                feats.update(FEATURE_FIXES[self.language].get(w.text.lower(), {}))
                upos = w.upos if any(ch.isalnum() for ch in w.text) else "PUNCT"  # CAC tags "–" as a noun at times
                words.append(_Word(t.start(), t.end(), w.text, upos, feats))
        return words

    # ---- items -----------------------------------------------------------------------------
    def _resolve(self, m: re.Match, text: str, tags: _Tags, after_label: bool, heading: bool) -> Tuple[int, str]:
        """(start of the replaced text, replacement) for one item."""
        kind, start, end = m.lastgroup, m.start(), m.end()
        if kind == "dash":
            if _NUMBER_BEFORE.search(text[:start].rstrip()) and _NUMBER_AFTER.match(text[end:].lstrip()):
                return start, _spaced(text, start, end, RANGE_WORD)  # "10:00–12:00", "XIX.–XX. století"
            return start, "—"
        if kind == "ellipsis":
            return start, "…"
        if kind == "sign":
            return start, _spaced(text, start, end, SIGNS[self.language][m.group(0)])
        if kind == "slash":
            if text[start - 1].isdigit():
                return start, _spaced(text, start, end, SLASH_WORDS[self.language]["number"])
            left, word = re.search(r"[^\W\d_]+$", text[:start]).group(), SLASH_WORDS[self.language]["word"]
            if m["suffix"]:
                feminine = self._feminine(left, m["suffix"])
                if _starts_sentence(text[:start - len(left)]):
                    feminine = feminine[0].lower() + feminine[1:]
                return start, f" {word} {feminine}"
            right = re.match(r"[^\W\d_]+", text[end:]).group()
            units = UNITS[self.language].keys() | PER_UNITS[self.language].keys()
            if ({left, right} & units) - set(_ONE_LETTER_WORDS):  # "s/bez": s is a preposition here
                return start, m.group(0)  # "Kč/kg" without a number has no reading; the check below raises
            return start, _spaced(text, start, end, word)
        if kind == "abbreviation":
            return start, self._abbreviation(m, text, tags)
        if kind == "roman":
            return self._vocalise(text, start, self._roman(m, text, tags, heading))
        if kind == "isodate":
            words = self._date(int(m["isoday"]), int(m["isomonth"]), m["isoyear"])
        elif kind == "date":
            year = m["year"] or m["shortyear"]
            words = self._date(int(m["day"]), int(m["month"]), year)
            if year is None and self._ends_sentence(text, end, tags):
                words += "."
        elif kind == "time":
            words = self._time(m, tags)
        elif kind == "range" and m["lowscale"]:  # "5 tis. Kč–10 tis. Kč", "5 tis. Kč/kg–10 tis. Kč/kg"
            words = f" {RANGE_WORD} ".join(
                self._measure(m[end_], m[end_ + "scale"], start, tags, text, end, scale_unit=m[end_ + "scaleunit"])
                + (f" {self._per(m[end_ + 'scaleper'])}" if m[end_ + "scaleper"] else "")
                for end_ in ("low", "high"))
        elif kind == "range" and m["lowunit"]:  # "5 km–10 m", "5 Kč/kg–10 Kč/kg"
            words = f" {RANGE_WORD} ".join(
                self._measure(m[end_], m[end_ + "unit"], start, tags, text, end)
                + (f" {self._per(m[end_ + 'per'])}" if m[end_ + "per"] else "")
                for end_ in ("low", "high"))
        elif kind == "range":
            words = self._range(m, m["low"], m["high"], m["rangeunit"], m["rangescale"], m["rangescaleunit"],
                                m["rangeper"] or m["rangescaleper"], text, tags, after_label)
        elif kind == "money":
            price = _plain_price(m["price"])
            sign = "" if price[0] in "-−" else (m["moneysign"] or "")  # "-$4.50", "$-4.50"
            scale = m["moneyscale"] or m["highmoneyscale"]
            if m["lowmoneyscale"] and m["lowmoneyscale"] != scale:  # "$500 tis.–$1 mil."
                words = f" {RANGE_WORD} ".join(
                    self._measure(amount, amount_scale, start, tags, text, end, scale_unit=m["symbol"])
                    for amount, amount_scale in ((sign + price, m["lowmoneyscale"]),
                                                 (_plain_price(m["pricehigh"]), scale)))
            elif m["lowmoneyper"]:  # "$4/kg–$5/kg"
                words = f" {RANGE_WORD} ".join(
                    f"{self._measure(amount, m['symbol'], start, tags, text, end)} {self._per(per)}"
                    for amount, per in ((sign + price, m["lowmoneyper"]),
                                        (_plain_price(m["highprice"]), m["highmoneyper"])))
            elif m["pricehigh"]:  # "$5–10" -> "pět až deset dolarů"
                words = self._range(m, sign + price, _plain_price(m["pricehigh"]), None if scale else m["symbol"],
                                    scale, m["symbol"] if scale else None, m["moneyper"], text, tags, after_label)
            else:  # "$5 mil." -> "pět milionů dolarů"
                words = self._measure(sign + price, scale or m["symbol"], start, tags, text, end,
                                      scale_unit=m["symbol"] if scale else None)
                if m["moneyper"]:
                    words += " " + self._per(m["moneyper"])  # "$4/kg" -> "čtyři dolary za kilogram"
        elif kind == "measure":
            words = self._measure(m["amount"], m["unit"] or m["scale"], start, tags, text, end,
                                  whole=bool(m["whole"]), scale_unit=m["scaleunit"])
            if m["per"] or m["scaleper"]:  # "100 Kč/kg" -> "sto korun za kilogram"
                words += " " + self._per(m["per"] or m["scaleper"])
        elif kind == "ordinal":
            words = self._ordinal_digits(m, text, tags, after_label, heading)
        else:
            words = self._number(m, text, tags, after_label)
        if (kind in ("range", "measure", "time", "money") and m.group(0).endswith(".")
                and self._ends_sentence(text, end, tags)):
            words += "."  # the period of "min.", "mil." or "hod." also ends the sentence
        return self._vocalise(text, start, words)

    def _abbreviation(self, m: re.Match, text: str, tags: _Tags) -> str:
        raw, key = m.group(0), _key_of(m)
        if raw.isupper() and len(key) > 2 and key in CAPITAL_ACRONYMS:
            return raw
        words = self.abbreviations[key]
        if key in AGREEING_ABBREVIATIONS:
            head = tags.head_after(m.end())
            case, gender, animacy, plural, doubt = self._agreement(head, m.start(), tags, ordinal=False)
            if case is None or gender is None:
                self._warn(text, m, words, "no noun to agree with")
            else:
                words = self._numbers.decline_ordinal(words, CASES.index(case), gender, animacy, plural)
                if doubt:
                    self._warn(text, m, words, f"{doubt}; check it")
        elif key in DECLINED_ABBREVIATIONS[self.language]:
            prep = self._reference_preposition(m.start(), tags)
            if prep is not None:
                case = (LOC if prep.text.lower() in LOCATIVE_PREPOSITIONS[self.language]
                        else _UD_CASES.get(prep.feats.get("Case")))
                if case:
                    words = DECLINED_ABBREVIATIONS[self.language][key][CASES.index(case)]
        if not raw.isupper() or _starts_sentence(text[:m.start()]):
            words = _capitalise_like(raw, words)
        if key.endswith(".") and self._ends_sentence(text, m.end(), tags,
                                                     introduces=key in NON_FINAL_ABBREVIATIONS):
            words += "."
        return words

    def _roman(self, m: re.Match, text: str, tags: _Tags, heading: bool) -> str:
        numeral, end = m["numeral"], m.end()
        nxt, prev = tags.after(end), tags.before(m.start())
        head = None
        if prev is None or prev.upos != "PROPN":
            # "XXI. století", "XIX.–XX. století", "# V. Kapitola", but "Karel IV. univerzitu" agrees with Karel
            before_noun = nxt is not None and (nxt.text[:1].islower() or nxt.upos in ("NOUN", "ADJ") or heading)
            head = (tags.head_after(end) if before_noun else None) or self._shared_head(end, tags)
        if head is None:
            if (prev is None or prev.upos not in ("NOUN", "PROPN") or _roman_value(numeral) >= 400
                    or (len(numeral) == 1 and nxt and nxt.text[:1].isupper())):
                return m.group(0)  # an initial such as "V. Havel", "Washington DC.", or no noun to agree with
            head = prev
        case, gender, animacy, plural, doubt = self._agreement(head, m.start(), tags,
                                                               following=head.start > m.start())
        words = self._ordinal(_roman_value(numeral), case or NOM, gender or "masculine", animacy or "inanimate",
                              plural)
        if case is None or gender is None:
            self._warn(text, m, words, "no noun to agree with; nominative masculine inanimate")
        elif doubt:
            self._warn(text, m, words, f"{doubt}; check it")
        if head.start < m.start() and self._ends_sentence(text, end, tags, roman=True):
            words += "."  # after "Karel IV." the period may end the sentence; before a noun it cannot
        return words

    def _date(self, day: int, month: int, year: Optional[str]) -> str:
        words = [self._ordinal(day, GEN, "masculine", "inanimate"), MONTHS_GENITIVE[self.language][month - 1]]
        if year:
            zero = f"{self._cardinal(0, NOM, 'masculine', 'inanimate')} " if year[0] == "0" else ""
            words.append(zero + self._cardinal(int(year), NOM, "masculine", "inanimate"))
        return " ".join(words)

    def _time(self, m: re.Match, tags: _Tags) -> str:
        hour, minute, second = int(m["hour"]), int(m["minute"]), m["second"]
        prep = self._preposition(m.start(), tags)
        case = None
        if prep:
            case = (TIME_PREPOSITIONS[self.language].get(prep.text.lower())
                    or _UD_CASES.get(prep.feats.get("Case")))
        if self.language == "sk" and case and prep.text.lower() in SK_CLOCK_PREPOSITIONS:  # "o štrnástej tridsať"
            words = [self._ordinal(hour, case, "feminine", "inanimate")]  # also "o nultej"
            if minute or second:
                words.append(self._minutes(minute, NOM))
            if second:
                words.append(self._minutes(int(second), NOM))
            return " ".join(words)
        case = case or NOM
        words = [self._cardinal(hour, case, "feminine", "inanimate") if hour else "nula"]
        mcase = case if self.language == "cs" and case not in (NOM, ACC) else NOM
        if minute or second:  # digital: "dvě patnáct třicet" for 2:15:30
            words.append(self._minutes(minute, mcase))
            if second:
                words.append(self._minutes(int(second), mcase))
        else:
            words.append(self._noun_phrase("hodina", hour, case))
        return " ".join(words)

    def _minutes(self, minute: int, case: str) -> str:
        words = self._cardinal(minute, case, "feminine", "inanimate")
        return f"nula {words}" if minute < 10 else words

    def _range(self, m: re.Match, low_amount: str, high_amount: str, unit: Optional[str], scale: Optional[str],
               scale_unit: Optional[str], per: Optional[str], text: str, tags: _Tags, after_label: bool) -> str:
        """"5–10 km", "5–10 tis. Kč", "$5–10": a unit or scale after the upper end serves both."""
        start, end = m.start(), m.end()
        low, value = _parse(low_amount)[0], _parse(high_amount)[0]
        decimal = isinstance(low, str) or isinstance(value, str)  # decimals are read in the nominative
        if unit or scale:
            high = self._measure(high_amount, scale or unit, start, tags, text, end, scale_unit=scale_unit)
            if unit and decimal and self._unit(unit)[0] in MINOR_UNITS[self.language]:
                words = f"{self._measure(low_amount, unit, start, tags, text, end)} {RANGE_WORD} {high}"  # "1,50–2,50 €"
            else:
                gender = SCALE_GENDERS[self.language][scale] if scale else NOUNS[self.language][self._unit(unit)[0]][0]
                case = NOM if decimal else self._preposition_case(start, tags) or NOM
                words = f"{self._signed(low_amount, self._cardinal(low, case, gender, 'inanimate'))} {RANGE_WORD} {high}"
        else:
            if decimal:
                case, gender, animacy = NOM, "masculine", "inanimate"
            else:
                case, gender, animacy = self._context(value, start, end, text, tags, after_label, m)
            words = (f"{self._signed(low_amount, self._cardinal(low, case, gender, animacy))} {RANGE_WORD} "
                     f"{self._signed(high_amount, self._cardinal(value, case, gender, animacy))}")
        return words + (f" {self._per(per)}" if per else "")

    def _measure(self, amount: str, unit: str, start: int, tags: _Tags, text: str, end: int,
                 whole: bool = False, scale_unit: Optional[str] = None) -> str:
        value, integer, fraction = _parse(amount)
        case = self._preposition_case(start, tags)
        unit = unit.rstrip(".")
        if unit in SCALES:
            if fraction:
                words = f"{self._cardinal(value, NOM, 'masculine', 'inanimate')} {SCALE_GENITIVES[self.language][unit]}"
            else:
                words = self._cardinal(value * 10 ** SCALES[unit], case or NOM, "masculine", "inanimate")
            if scale_unit:  # "5 tis. km" -> "pět tisíc kilometrů": the noun follows tisíc, milion
                noun, adjective, suffix = self._unit(scale_unit)
                count = 1000 if fraction else integer * 10 ** SCALES[unit]
                words += f" {self._noun_phrase(noun, count, NOM if fraction else case or NOM, adjective)}{suffix}"
            return words
        noun, adjective, suffix = self._unit(unit)

        def read(c: str) -> str:
            minor = MINOR_UNITS[self.language].get(noun)
            if fraction and minor and len(fraction) <= 2 and not whole:
                cents = int(fraction.ljust(2, "0"))
                words = self._counted(cents, minor, c)
                words = words if integer == 0 else f"{self._counted(integer, noun, c)} {words}"
                return f"{self._numbers.MINUS} {words}" if str(value).startswith("-") else words
            if fraction:  # "tři celé pět desetin kilometru": the noun is genitive singular
                return (f"{self._cardinal(value, NOM, 'masculine', 'inanimate')} "
                        f"{self._noun_phrase(noun, 1, GEN, adjective)}")
            return self._counted(value, noun, c, adjective)

        words = read(case or NOM)
        if case is None and words != read(ACC):
            self._warn(text, (start, end), words, "no preposition; nominative")
        if noun in MINOR_UNITS[self.language] and re.fullmatch(r"[-−–]?\d{1,3},\d{3}", amount):
            self._warn(text, (start, end), words, "comma read as decimal, not thousands; check it")
        return self._signed(amount, words) + suffix

    def _signed(self, amount: str, words: str) -> str:
        return f"{self._numbers.MINUS} {words}" if _negative_zero(amount) else words

    def _per(self, per: str) -> str:
        """The unit after "/" in the accusative singular: "za kilogram", "za metr čtvereční"."""
        power = {"2": "²", "3": "³"}.get(per[-1], per[-1]) if per[-1] in "²³23" else None
        adjective = UNIT_ADJECTIVES[self.language].get(power)
        return f"za {PER_UNITS[self.language][per[:-1] if power else per]}" + (f" {adjective}" if adjective else "")

    def _unit(self, unit: str) -> Tuple[str, Optional[str], str]:
        """(noun, agreeing adjective, suffix) of a unit symbol: "m²" -> metr, čtvereční."""
        unit = unit.rstrip(".")
        power = {"2": "²", "3": "³"}.get(unit[-1], unit[-1]) if unit[-1] in "²³23" else None
        base = unit[:-1] if power else unit
        return (UNITS[self.language][base], UNIT_ADJECTIVES[self.language].get(power),
                UNIT_SUFFIXES[self.language].get(base, ""))

    def _counted(self, value: int, noun: str, case: str, adjective: Optional[str] = None) -> str:
        number = self._cardinal(value, case, NOUNS[self.language][noun][0], "inanimate")
        return f"{number} {self._noun_phrase(noun, abs(value), case, adjective)}"

    def _noun_phrase(self, noun: str, count: int, case: str, adjective: Optional[str] = None) -> str:
        """The noun (and adjective) as `count` calls for: "pět kilometrů" is genitive plural."""
        gender, forms = NOUNS[self.language][noun]
        if self._ends_in_scale_noun(count):
            c, plural = GEN, True  # "s tisícem korun", sk "s miliónom eur"
        elif count == 1:
            c, plural = case, False
        elif case not in (NOM, ACC):
            c, plural = case, True
        else:
            form = self._count_form(count)
            c, plural = (GEN, True) if form == "gen_pl" else (case, form == "pl")
        words = forms[(6 if plural else 0) + CASES.index(c)]
        if adjective:
            words += " " + self._numbers.decline_ordinal(adjective, CASES.index(c), gender, "inanimate", plural)
        return words

    def _ordinal_digits(self, m: re.Match, text: str, tags: _Tags, after_label: bool, heading: bool) -> str:
        value = _parse(m["ordinalvalue"])[1]  # "1 000. návštěvník"
        following, nxt = text[m.end():].lstrip(" \t\u00a0\u202f"), tags.after(m.end())
        prev = tags.before(m.start())
        attributive = prev is None or prev.upos in ("ADJ", "DET", "ADP", "PUNCT")  # "Beethovenova 5. Symfonie"
        if following[:1].isupper() and not heading and (nxt is None or nxt.upos not in ("NOUN", "ADJ")
                                                         or (not attributive and self._verb_follows(m.end(), tags))):
            # "Bylo jich 5. Pak…", "Měl jen 2. Děti odešly.": a number that ends the sentence; but "5. Symfonie",
            # "# 2. Kapitola"
            case, gender, animacy = self._context(value, m.start(), m.end() - 1, text, tags, after_label, m)
            return self._cardinal(value, case, gender, animacy) + "."
        head = tags.head_after(m.end()) or self._shared_head(m.end(), tags)
        case, gender, animacy, plural, doubt = self._agreement(head, m.start(), tags)
        words = self._ordinal(value, case or NOM, gender or "masculine", animacy or "inanimate", plural)
        if case is None or gender is None:
            self._warn(text, m, words, "no noun to agree with; nominative masculine inanimate")
        elif doubt:
            self._warn(text, m, words, f"{doubt}; check it")
        return words

    def _number(self, m: re.Match, text: str, tags: _Tags, after_label: bool) -> str:
        value, _, fraction = _parse(m["value"])
        if m["compound"] and not fraction and value > 0:
            words = self._combining(value) + m["compound"]  # "dvacetipětiletý", sk "päťročný"
        elif m["compound"]:
            words = f"{self._cardinal(value, NOM, 'masculine', 'inanimate')} {m['compound']}"
            self._warn(text, m, words, "no compound form for this number")
        elif fraction:  # decimals are read in the nominative (Step 2.1)
            words = self._cardinal(value, NOM, "masculine", "inanimate")
        elif m["times"]:
            words = self._cardinal(value, NOM, "masculine", "inanimate") + "krát"
        else:
            words = self._cardinal(value, *self._context(value, m.start(), m.end(), text, tags,
                                                         after_label, m))
        words = self._signed(m["value"], words)
        if m.start() and text[m.start() - 1].isalpha():
            words = " " + words
        if m.end() < len(text) and text[m.end()].isalpha() and not re.match(r"x\d", text[m.end():m.end() + 2]):
            self._warn(text, m, words, "digits glued to a word")  # "3x4" is a multiplication, read by the sign
            words += " "
        return words

    # ---- context ---------------------------------------------------------------------------
    def _context(self, value: int, start: int, end: int, text: str, tags: _Tags, after_label: bool,
                 m) -> Tuple[str, str, str]:
        """Case, gender and animacy of a cardinal, from the noun it counts or the preposition before it."""
        if after_label or text[:start].rstrip()[-1:] in _MATH_SIGNS or text[end:].lstrip()[:1] in _MATH_SIGNS:
            return self._label(value)  # "č. 5", "#1", "tři krát čtyři", "2023/2024"
        prep_case = self._preposition_case(start, tags)
        noun = tags.head_after(end)
        case = gender = animacy = tagged_case = noun_case = None
        if noun is not None:
            gender, animacy = self._gender(noun)
            tagged_case = _UD_CASES.get(noun.feats.get("Case"))
            if noun.feats.get("Number") == "Plur" and noun.text.lower().endswith(LOCATIVE_PLURAL_ENDINGS):
                tagged_case = LOC  # "po tisíci letech"
            elif (tagged_case in PLURAL_ENDINGS and noun.feats.get("Number") == "Plur"
                    and not noun.text.lower().endswith(PLURAL_ENDINGS[tagged_case])):
                tagged_case = GEN  # "o 5 minút": a genitive plural tagged with the preposition's case
            if self._number_fits(value, noun):
                noun_case = tagged_case
        if noun_case in (DAT, INS, LOC):
            case = noun_case  # "s pěti přáteli"
        elif prep_case == GEN:
            case = GEN  # "bez pěti jablek", "do dvou hodin"
        elif prep_case and self._ends_in_scale_noun(value):
            case = prep_case  # "s tisícem lidí": after tisíc the noun is genitive in every case
        elif prep_case:
            # "ve dvě hodiny", "v pět hodin": the tagger often gives v/na/o the locative here
            case = ACC if prep_case == ACC or tagged_case in (NOM, ACC, GEN) else prep_case
        elif noun is not None:
            # no governor: "Mám pět jablek" (the noun is genitive, the number nominative);
            # Slovak "videl troch mužov" (A = G of masculine personal nouns), "dvoch psov"
            if (self.language == "sk" and noun_case == GEN and noun.text.lower().endswith("ov")
                    and self._count_form(value) != "gen_pl"):
                case = GEN
            else:
                case = ACC if noun_case == ACC or (noun_case == GEN and animacy == "personal") else NOM
        if case is None and noun is None:
            prev = tags.before(start)
            if prev is not None and prev.upos in ("NOUN", "PROPN"):
                return self._label(value)  # "kapitola 5", "v roce 2024"
        if case is None:
            self._warn(text, m, self._cardinal(value, NOM, "masculine", "inanimate"),
                       "no governing noun or preposition; nominative masculine inanimate")
            return NOM, "masculine", "inanimate"
        if gender is None:
            readings = {self._cardinal(value, case, g, a) for g, a in
                        (("masculine", "inanimate"), ("masculine", "animate"), ("masculine", "personal"),
                         ("feminine", "inanimate"), ("neuter", "inanimate"))
                        if a != "personal" or self.language == "sk"}
            if len(readings) > 1:
                self._warn(text, m, self._cardinal(value, case, "masculine", "inanimate"),
                           "no counted noun; masculine inanimate")
            return case, "masculine", "inanimate"
        return case, gender, animacy

    def _shared_head(self, pos: int, tags: _Tags) -> Optional[_Word]:
        """The noun of the next ordinal, which this one shares: "1. a 2. díl", "od XIX. do XX. století",
        "konec XIX. a začátek XX. století"."""
        i = bisect.bisect_left(tags.starts, pos)
        w = tags.words
        if i + 3 < len(w) and w[i].upos == "CCONJ" and w[i + 1].upos == "NOUN" and w[i + 1].text.isalpha():
            i += 1
        if (i + 2 < len(w) and (w[i].upos in ("CCONJ", "ADP", "NOUN") or w[i].text in ("–", "—", "-", ","))
                and (w[i + 1].text.isdigit() or re.fullmatch(_ROMAN, w[i + 1].text)) and w[i + 2].text == "."):
            return tags.head_after(w[i + 2].end) or self._shared_head(w[i + 2].end, tags)  # "2., 3. a 4. díl"
        return None

    @staticmethod
    def _chain_start(pos: int, tags: _Tags) -> int:
        """Where "1.–5." or "XIX. a XX." starts, for an ordinal at `pos`: the whole chain has one governor."""
        i, w = bisect.bisect_left(tags.starts, pos) - 1, tags.words
        while (i >= 2 and (w[i].upos == "CCONJ" or w[i].text in ("–", "—", "-", ",")) and w[i - 1].text == "."
               and (w[i - 2].text.isdigit() or re.fullmatch(_ROMAN, w[i - 2].text))):
            i -= 3
        return w[i + 1].start if i + 1 < len(w) else pos

    def _label(self, value: int) -> Tuple[str, str, str]:
        """A number that names rather than counts is nominative; Czech labels and counts with "jedna"."""
        return NOM, "feminine" if self.language == "cs" and abs(value) == 1 else "masculine", "inanimate"

    def _agreement(self, head: Optional[_Word], start: int, tags: _Tags, following: bool = True,
                   ordinal: bool = True):
        """Case, gender, animacy and plural of an adjective (an ordinal) agreeing with `head`, and a
        doubt to log when the tags look wrong."""
        if head is None:
            return None, None, None, False, None
        case = _UD_CASES.get(head.feats.get("Case"))
        gender, animacy = self._gender(head)
        plural = head.feats.get("Number") == "Plur"
        doubt = None
        if following:
            start = self._chain_start(start, tags)
            prev = tags.before(start)
            prep_case = _UD_CASES.get(prev.feats.get("Case")) if prev is not None and prev.upos == "ADP" else None
            if self.language == "cs" and head.feats.get("Gender") == "Neut" and head.text.lower().endswith("í"):
                # "století" has one form for every case but the instrumental, so the context decides
                context = prep_case or (GEN if prev is not None and prev.upos in ("NOUN", "PROPN") else NOM)
                if ordinal and context == NOM and case not in (None, NOM, ACC) and not self._clause_start(start, tags):
                    doubt = f"{head.text!r} tagged {case}, read {context}"  # "Dosáhli jsme XXI. století"
                case = context
            elif prep_case and (case == NOM or (case == GEN and prep_case != GEN)):
                case = prep_case  # "v XXI. století" tagged nominative or genitive
            elif case == GEN and head.text.lower() not in MONTHS_GENITIVE[self.language]:
                if self._clause_start(start, tags):
                    case = NOM  # nothing governs a genitive here: "XXI. století" at the start of a sentence
                elif ordinal and prev.upos in ("CCONJ", "PUNCT"):
                    doubt = "genitive noun with no governing word"  # sk "a III. diel" tagged genitive plural
            if ordinal and plural and doubt is None:
                doubt = "ordinal agrees with a plural noun"
        return case, gender, animacy, plural, doubt

    def _gender(self, word: _Word) -> Tuple[Optional[str], Optional[str]]:
        gender = _UD_GENDERS.get(word.feats.get("Gender", "").split(",")[0])
        if gender is None:
            return None, None
        if gender == "masculine" and word.feats.get("Animacy") == "Anim":
            if self.language == "cs":
                return gender, "animate"
            form = word.text.lower()
            animal = form in SK_ANIMAL_PLURALS or (word.feats.get("Number") == "Plur" and form.endswith(("y", "e")))
            return gender, "animate" if animal else "personal"  # "dvaja muži", but "dva vlci", "dva psy"
        return gender, "inanimate"

    def _preposition(self, pos: int, tags: _Tags) -> Optional[_Word]:
        word = tags.before(pos, skip=("ADV", "PART"))
        return word if word is not None and word.upos == "ADP" else None

    def _reference_preposition(self, pos: int, tags: _Tags) -> Optional[_Word]:
        """The preposition that governs a whole reference: "podle § 5 odst. 2" -> podle."""
        i = bisect.bisect_left(tags.starts, pos) - 1
        while i >= 0 and (tags.words[i].upos in ("ADV", "PART") or tags.words[i].text.isdigit()
                          or tags.words[i].text in ("§", ".", ",")
                          or f"{tags.words[i].text.lower()}." in DECLINED_ABBREVIATIONS[self.language]):
            i -= 1
        return tags.words[i] if i >= 0 and tags.words[i].upos == "ADP" else None

    def _preposition_case(self, pos: int, tags: _Tags) -> Optional[str]:
        prep = self._preposition(pos, tags)
        return _UD_CASES.get(prep.feats.get("Case")) if prep else None

    @staticmethod
    def _verb_follows(pos: int, tags: _Tags) -> bool:
        """Whether the noun phrase after `pos` is followed by a verb, as the subject of a new sentence;
        adverbs, particles and clitics may come between: "Malé děti se potom vrátily"."""
        head = tags.head_after(pos)
        if head is None:
            return False
        i = bisect.bisect_left(tags.starts, head.end)
        for w in tags.words[i:i + 4]:
            if w.upos in ("VERB", "AUX"):
                return True
            if w.upos not in ("ADV", "PART", "PRON"):
                return False
        return False

    @staticmethod
    def _clause_start(pos: int, tags: _Tags) -> bool:
        prev = tags.before(pos)
        return prev is None or (prev.upos == "PUNCT" and prev.text in ".!?…")

    def _number_fits(self, value, noun: _Word) -> bool:
        """False when the tagged number of the noun cannot follow `value` ("3 hrušky" read as singular)."""
        number = noun.feats.get("Number")
        if number is None or not isinstance(value, int):
            return True
        if abs(value) == 1:
            return number == "Sing"
        return number == "Plur" or self._count_form(value) == "sg"

    def _combining(self, n: int) -> str:
        """The number as the first part of a compound word: pěti-, dvacetipěti-; sk päť-, dvoj-."""
        if self.language == "cs":
            special = {1: "jedno", 2: "dvou", 3: "tří", 4: "čtyř", 100: "sto", 1000: "tisíci"}
            return special.get(n) or self._cardinal(n, GEN, "masculine", "inanimate").replace(" ", "")
        special = {1: "jedno", 2: "dvoj", 3: "troj", 4: "štvor"}
        return special.get(n) or self._cardinal(n, NOM, "masculine", "inanimate").replace(" ", "")

    def _ends_in_scale_noun(self, n) -> bool:
        """Whether n ends in a numeral that is a noun (Czech tisíc, milion; Slovak milión, miliarda)."""
        scale = 1000 if self.language == "cs" else 10 ** 6
        return isinstance(n, int) and n != 0 and n % scale == 0

    def _count_form(self, n: int) -> str:
        """How a counted noun follows n in the nominative and accusative: "sg", "pl" or "gen_pl"."""
        n = abs(n)
        if n == 1:
            return "sg"
        if n in (2, 3, 4):
            return "pl"
        if self.language == "sk":
            return "pl" if n > 100 and n % 100 in (2, 3, 4) else "gen_pl"  # "stodve knihy" (native review)
        if self.variants.get("construction") == "agreement" and n > 20:
            t = n % 100
            unit = t % 10 if t > 20 else (t if t < 5 else 0)
            return {1: "sg", 2: "pl", 3: "pl", 4: "pl"}.get(unit, "gen_pl")
        return "gen_pl"

    def _ends_sentence(self, text: str, end: int, tags: _Tags, roman: bool = False,
                       introduces: bool = False) -> bool:
        """Whether the period of an abbreviation, date or Roman numeral that ends at `end` also ends a
        sentence: at the end of the paragraph, or before an uppercase word (Slovak "atď. Potom", but
        not "např. Prahu")."""
        rest = _SPAN_RE.sub(lambda s: s.group(2), text[end:])
        if _RANGE_AHEAD.match(rest):
            return False  # "XIX.–XX. století"
        nxt = rest.lstrip(" \t\n\r\u00a0\u202f\"'„“”‚‘’«»‹›()[]—–-")
        if not nxt:
            return True
        if introduces or not nxt[:1].isupper():
            return False
        if roman:  # "Karel IV. Lucemburský" goes on; "Vládl Karel IV. Potom…", "…IV. Velký požár vypukl." do not
            word = tags.after(end)
            return (word is None or word.upos not in ("PROPN", "ADJ")
                    or (word.upos == "ADJ" and self._verb_follows(end, tags)))
        return True

    def _feminine(self, word: str, suffix: str) -> str:
        """The feminine form that an inclusive "on/a", "přišel/a" or sk "mohol/a" stands for."""
        lower = word.lower()
        for suffixes, endings, feminine in FEMININE_ENDINGS:
            ending = next((e for e in endings if lower.endswith(e)), None)
            if suffix in suffixes and ending:
                return _capitalise_like(word, word[:len(word) - len(ending)] + feminine)
        if (suffix == "a" and self.language == "sk" and len(lower) > 3 and lower.endswith("ol")
                and lower[-3] not in "aeiouyáéíóúýäô"):
            return word[:-2] + "la"
        return word + suffix

    def _vocalise(self, text: str, start: int, words: str) -> Tuple[int, str]:
        """"s 2 přáteli" -> "se dvěma přáteli": vocalise a one-letter preposition before the number words."""
        m = _LETTER_BEFORE.search(text[:start])
        rule = VOCALISATION[self.language].get(m.group(1).lower()) if m else None
        if rule is None or not words.startswith(rule[1]):
            return start, words
        return m.start(1), f"{_capitalise_like(m.group(1), rule[0])}{text[m.end(1):start]}{words}"

    # ---- num2words -------------------------------------------------------------------------
    def _cardinal(self, value, case: str, gender: str, animacy: str) -> str:
        return self._numbers.num2words(value, to="cardinal", gender=gender, case=case, animacy=animacy,
                                       **self.variants)

    def _ordinal(self, value: int, case: str, gender: str, animacy: str, plural: bool = False) -> str:
        return self._numbers.num2words(value, to="ordinal", gender=gender, case=case, animacy=animacy,
                                       plural=plural, **self.variants)

    def _warn(self, text: str, where, reading: str, reason: str) -> None:
        start, end = where if isinstance(where, tuple) else where.span()
        log.warning("%s: read %r as %r in %r", reason, text[start:end], reading.strip(),
                    text[max(0, start - 40):end + 40].replace("\n", " "))
