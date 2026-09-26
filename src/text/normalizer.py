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
MINOR_UNITS = {"cs": {"koruna": "haléř", "euro": "cent", "dolar": "cent"},
               "sk": {"koruna": "halier", "euro": "cent", "dolár": "cent"}}
SCALES = {"tis": 3, "mil": 6, "mld": 9}
SCALE_GENITIVES = {"cs": {"tis": "tisíce", "mil": "milionu", "mld": "miliardy"},
                   "sk": {"tis": "tisíca", "mil": "milióna", "mld": "miliardy"}}
SIGNS = {"cs": {"&": "a", "+": "plus", "@": "zavináč", "=": "rovná se", "×": "krát", "±": "plus minus",
               "#": "číslo"},
         "sk": {"&": "a", "+": "plus", "@": "zavináč", "=": "rovná sa", "×": "krát", "±": "plus mínus",
               "#": "číslo"}}
SLASH_WORDS = {"cs": {"word": "nebo", "number": "lomeno"}, "sk": {"word": "alebo", "number": "lomené"}}
RANGE_WORD = "až"

# endings every plural noun has in these cases, in both languages (hradech, ženách, dubom, mužmi)
PLURAL_ENDINGS = {DAT: ("m",), INS: ("mi", "ma", "y", "i"), LOC: ("ch",)}

# Clock times: Czech "ve čtrnáct třicet" is accusative, Slovak "o štrnástej" locative,
# whatever case the tagger gives the preposition.
TIME_PREPOSITIONS = {"cs": {"v": ACC, "ve": ACC}, "sk": {"o": LOC}}

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
_TAG_TOKEN = re.compile(r"[^\W\d_]+|\d+|\S")
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}

_HS = r"[ \t\u00a0\u202f]"  # horizontal space: no item may swallow a line break
_INT = r"\d{1,3}(?:[ \u00a0\u202f]\d{3})+(?!\d)|\d{1,3}(?:\.\d{3})+(?!\d)|\d+"  # 10 000, 10.000
_AMOUNT = rf"(?:(?<![^\s(\[])[-−](?=\d))?(?:{_INT})(?:[.,]\d+)?"
_POWER = r"(?:[²³]|[23](?!\d))?"  # m², and m2 as typed
_UNIT = rf"km/h|km{_POWER}|cm{_POWER}|mm{_POWER}|m/s|m{_POWER}|kg|g|ml|l|°C|°|%|‰|hod\.?|min\.?"
_CURRENCY = r"Kč|€|EUR|USD|\$|£"
_ROMAN = r"(?=[IVXLC])C{0,3}(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})"  # up to 399: "CD.", "DC." are acronyms
_NOT_LETTER_AFTER = r"(?![^\W\d_])"
_NOT_LETTER_BEFORE = r"(?<![^\W\d_])"
_SPACES = re.compile(f"{_HS}*")


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
            out.append(rf"\.{_HS}?")
        elif i == 0 and len(key) > 2 and ch.isalpha():
            out.append(f"[{ch}{ch.upper()}]")
        else:
            out.append(re.escape(ch))
    return _NOT_LETTER_BEFORE + "".join(out) + (_NOT_LETTER_AFTER if key[-1].isalpha() else "")


def _items_pattern(abbreviations) -> re.Pattern:
    abbr = "|".join(_abbreviation_pattern(k) for k in sorted(abbreviations, key=len, reverse=True))
    return re.compile(
        rf"(?P<date>(?<!\d)(?P<day>3[01]|[12]\d|0?[1-9])\.{_HS}?(?P<month>1[0-2]|0?[1-9])\."
        rf"(?:{_HS}?(?P<year>\d{{4}})(?!\d))?)"
        rf"|(?P<time>(?<![\d.,:])(?P<hour>2[0-4]|[01]?\d):(?P<minute>[0-5]\d)(?![\d:]))"
        rf"|(?P<range>(?<![\d.,])(?P<low>\d+){_HS}?[–—-]{_HS}?(?P<high>{_INT})"
        rf"(?:{_HS}?(?P<rangeunit>{_UNIT}|{_CURRENCY}){_NOT_LETTER_AFTER})?)"
        rf"|(?P<money>(?P<symbol>[€$£]){_HS}?(?P<price>{_AMOUNT}))"
        rf"|(?P<measure>(?P<amount>{_AMOUNT})(?P<whole>,[-–—])?{_HS}?"
        rf"(?:(?P<scale>tis|mil|mld)\.?(?:{_HS}(?P<scalecurrency>{_CURRENCY}))?|(?P<unit>{_UNIT}|{_CURRENCY}))"
        rf"{_NOT_LETTER_AFTER})"
        rf"|(?P<ordinal>(?<![\d.,])(?P<ordinalvalue>\d+)\.(?={_HS}*[^\W\d_]))"
        rf"|(?P<number>(?P<value>{_AMOUNT})(?:(?P<times>krát|x|×){_NOT_LETTER_AFTER})?)"
        rf"|(?P<abbreviation>{abbr})"
        rf"|(?P<roman>{_NOT_LETTER_BEFORE}(?P<numeral>{_ROMAN})\.)"
        rf"|(?P<sign>[&+@=×±]|#(?={_HS}?\d))"
        rf"|(?P<slash>(?<=[^\W\d_]{{2}})/(?=[^\W\d_]{{2}})|(?<=\d){_HS}?/{_HS}?(?=\d))"
        rf"|(?P<dash>–|—|(?<!\S)-(?!\S))"
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
    s = re.sub(r"[ \u00a0\u202f]", "", amount).replace("−", "-")
    grouped = re.fullmatch(r"(-?\d{1,3}(?:\.\d{3})+)(,\d+)?", s)  # "10.000", "10.000,50"
    if grouped:
        s = grouped.group(1).replace(".", "") + (grouped.group(2) or "")
    whole, _, fraction = s.replace(".", ",").partition(",")
    fraction = fraction.rstrip("0")
    return (f"{whole},{fraction}" if fraction else int(whole)), abs(int(whole)), fraction


def _roman_value(numeral: str) -> int:
    values = [_ROMAN_VALUES[ch] for ch in numeral]
    return sum(-v if i + 1 < len(values) and v < values[i + 1] else v for i, v in enumerate(values))


def _spaced(text: str, start: int, end: int, word: str) -> str:
    """`word` in place of text[start:end], with a space on each side that lacks one."""
    left = "" if start == 0 or text[start - 1].isspace() else " "
    right = "" if end == len(text) or text[end].isspace() else " "
    return f"{left}{word}{right}"


def _key_of(m: re.Match) -> str:
    """The table key of a matched abbreviation: "Např." -> "např.", "t. j." -> "t.j."."""
    return _abbreviation_key(m.group(0)[0].lower() + m.group(0)[1:])


def _capitalise_like(source: str, words: str) -> str:
    return words[0].upper() + words[1:] if source[:1].isupper() else words


def _starts_sentence(before: str) -> bool:
    """Whether text after `before` (the paragraph up to it) starts a sentence."""
    before = _SPAN_RE.sub(lambda s: s.group(2), before).rstrip(" \t\n\r\u00a0\u202f\"'„“”‚‘’«»‹›([—–-")
    return not before or before[-1] in ".!?…"


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
        """The noun that a number or an adjective at `pos` counts or agrees with."""
        i = bisect.bisect_left(self.starts, pos)
        for w in self.words[i:i + 5]:
            if w.upos in ("NOUN", "PROPN"):
                return w
            if w.upos not in ("ADJ", "DET") and w.text not in _QUOTES:
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
                lines[i] = "# " + self._paragraph(lines[i][2:])
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
    def _paragraph(self, text: str) -> str:
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
            start, words = self._resolve(m, text, tags, after_label == m.start())
            source_case = m.lastgroup == "abbreviation" and m.group(0)[0].isalpha()
            if start == m.start() and not source_case and _starts_sentence("".join(out) + text[pos:start]):
                words = words[:1].upper() + words[1:]  # "5 lidí přišlo." -> "Pět lidí přišlo."
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
        return m.lastgroup not in ("date", "sign", "dash", "ellipsis")

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
                words.append(_Word(t.start(), t.end(), w.text, w.upos, feats))
        return words

    # ---- items -----------------------------------------------------------------------------
    def _resolve(self, m: re.Match, text: str, tags: _Tags, after_label: bool) -> Tuple[int, str]:
        """(start of the replaced text, replacement) for one item."""
        kind, start, end = m.lastgroup, m.start(), m.end()
        if kind == "dash":
            if text[:start].rstrip()[-1:].isdigit() and text[end:].lstrip()[:1].isdigit():
                return start, _spaced(text, start, end, RANGE_WORD)  # "10:00–12:00"
            return start, "—"
        if kind == "ellipsis":
            return start, "…"
        if kind == "sign":
            return start, _spaced(text, start, end, SIGNS[self.language][m.group(0)])
        if kind == "slash":
            return start, _spaced(text, start, end, SLASH_WORDS[self.language][
                "number" if text[start - 1].isdigit() else "word"])
        if kind == "abbreviation":
            return start, self._abbreviation(m, text, tags)
        if kind == "roman":
            return self._vocalise(text, start, self._roman(m, text, tags))
        if kind == "date":
            words = self._date(m)
            if m["year"] is None and self._ends_sentence(text, end, tags):
                words += "."
        elif kind == "time":
            words = self._time(m, tags)
        elif kind == "range":
            words = self._range(m, text, tags, after_label)
        elif kind == "money":
            words = self._measure(m["price"], m["symbol"], start, tags, text, end)
        elif kind == "measure":
            words = self._measure(m["amount"], m["unit"] or m["scale"], start, tags, text, end,
                                  whole=bool(m["whole"]), scale_currency=m["scalecurrency"])
        elif kind == "ordinal":
            words = self._ordinal_digits(m, text, tags, after_label)
        else:
            words = self._number(m, text, tags, after_label)
        if kind in ("range", "measure") and m.group(0).endswith(".") and self._ends_sentence(text, end, tags):
            words += "."  # the period of "min." or "mil." also ends the sentence
        return self._vocalise(text, start, words)

    def _abbreviation(self, m: re.Match, text: str, tags: _Tags) -> str:
        raw, key = m.group(0), _key_of(m)
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
        words = _capitalise_like(raw, words)
        if key.endswith(".") and self._ends_sentence(text, m.end(), tags,
                                                     introduces=key in NON_FINAL_ABBREVIATIONS):
            words += "."
        return words

    def _roman(self, m: re.Match, text: str, tags: _Tags) -> str:
        numeral, end = m["numeral"], m.end()
        nxt, prev = tags.after(end), tags.before(m.start())
        head = None
        if (prev is None or prev.upos != "PROPN") and nxt and nxt.text[:1].islower():
            head = tags.head_after(end)  # "XXI. století", but "Karel IV. univerzitu" agrees with Karel
        if head is None:
            if prev is None or prev.upos not in ("NOUN", "PROPN") or (len(numeral) == 1 and nxt
                                                                      and nxt.text[:1].isupper()):
                return m.group(0)  # an initial such as "V. Havel", or no noun to agree with
            head = prev
        case, gender, animacy, plural, doubt = self._agreement(head, m.start(), tags,
                                                               following=head.start > m.start())
        words = self._ordinal(_roman_value(numeral), case or NOM, gender or "masculine", animacy or "inanimate",
                              plural)
        if case is None or gender is None:
            self._warn(text, m, words, "no noun to agree with; nominative masculine inanimate")
        elif doubt:
            self._warn(text, m, words, f"{doubt}; check it")
        if self._ends_sentence(text, end, tags, roman=True):
            words += "."
        return words

    def _date(self, m: re.Match) -> str:
        words = [self._ordinal(int(m["day"]), GEN, "masculine", "inanimate"),
                 MONTHS_GENITIVE[self.language][int(m["month"]) - 1]]
        if m["year"]:
            words.append(self._cardinal(int(m["year"]), NOM, "masculine", "inanimate"))
        return " ".join(words)

    def _time(self, m: re.Match, tags: _Tags) -> str:
        hour, minute = int(m["hour"]), int(m["minute"])
        prep = self._preposition(m.start(), tags)
        case = None
        if prep:
            case = (TIME_PREPOSITIONS[self.language].get(prep.text.lower())
                    or _UD_CASES.get(prep.feats.get("Case")))
        if self.language == "sk" and case:  # "o štrnástej tridsať": the hour is an ordinal
            words = [self._ordinal(hour, case, "feminine", "inanimate") if hour else "nula"]
            if minute:
                words.append(self._minutes(minute, NOM))
            return " ".join(words)
        case = case or NOM
        words = [self._cardinal(hour, case, "feminine", "inanimate") if hour else "nula"]
        if minute:
            words.append(self._minutes(minute, case if self.language == "cs" and case not in (NOM, ACC) else NOM))
        else:
            words.append(self._noun_phrase("hodina", hour, case))
        return " ".join(words)

    def _minutes(self, minute: int, case: str) -> str:
        words = self._cardinal(minute, case, "feminine", "inanimate")
        return f"nula {words}" if minute < 10 else words

    def _range(self, m: re.Match, text: str, tags: _Tags, after_label: bool) -> str:
        low = int(m["low"])
        if m["rangeunit"]:
            high = self._measure(m["high"], m["rangeunit"], m.start(), tags, text, m.end())
            gender = NOUNS[self.language][self._unit(m["rangeunit"])[0]][0]
            case = self._preposition_case(m.start(), tags) or NOM
            return f"{self._cardinal(low, case, gender, 'inanimate')} {RANGE_WORD} {high}"
        value = _parse(m["high"])[0]
        case, gender, animacy = self._context(value, m.start(), m.end(), text, tags, after_label, m)
        return (f"{self._cardinal(low, case, gender, animacy)} {RANGE_WORD} "
                f"{self._cardinal(value, case, gender, animacy)}")

    def _measure(self, amount: str, unit: str, start: int, tags: _Tags, text: str, end: int,
                 whole: bool = False, scale_currency: Optional[str] = None) -> str:
        value, integer, fraction = _parse(amount)
        case = self._preposition_case(start, tags)
        unit = unit.rstrip(".")
        if unit in SCALES:
            if fraction:
                words = f"{self._cardinal(value, NOM, 'masculine', 'inanimate')} {SCALE_GENITIVES[self.language][unit]}"
            else:
                words = self._cardinal(value * 10 ** SCALES[unit], case or NOM, "masculine", "inanimate")
            if scale_currency:
                words += " " + NOUNS[self.language][UNITS[self.language][scale_currency]][1][7]
            return words
        noun, adjective, suffix = self._unit(unit)

        def read(c: str) -> str:
            minor = MINOR_UNITS[self.language].get(noun)
            if fraction and minor and len(fraction) <= 2 and not whole:
                cents = int(fraction.ljust(2, "0"))
                words = self._counted(cents, minor, c)
                return words if integer == 0 else f"{self._counted(integer, noun, c)} {words}"
            if fraction:  # "tři celé pět desetin kilometru": the noun is genitive singular
                return (f"{self._cardinal(value, NOM, 'masculine', 'inanimate')} "
                        f"{self._noun_phrase(noun, 1, GEN, adjective)}")
            return self._counted(value, noun, c, adjective)

        words = read(case or NOM)
        if case is None and words != read(ACC):
            self._warn(text, (start, end), words, "no preposition; nominative")
        return words + suffix

    def _unit(self, unit: str) -> Tuple[str, Optional[str], str]:
        """(noun, agreeing adjective, suffix) of a unit symbol: "m²" -> metr, čtvereční."""
        unit = unit.rstrip(".")
        power = {"2": "²", "3": "³"}.get(unit[-1], unit[-1]) if unit[-1] in "²³23" else None
        base = unit[:-1] if power else unit
        return (UNITS[self.language][base], UNIT_ADJECTIVES[self.language].get(power),
                UNIT_SUFFIXES[self.language].get(base, ""))

    def _counted(self, value: int, noun: str, case: str, adjective: Optional[str] = None) -> str:
        gender = NOUNS[self.language][noun][0]
        return f"{self._cardinal(value, case, gender, 'inanimate')} {self._noun_phrase(noun, abs(value), case, adjective)}"

    def _noun_phrase(self, noun: str, count: int, case: str, adjective: Optional[str] = None) -> str:
        """The noun (and adjective) as `count` calls for: "pět kilometrů" is genitive plural."""
        gender, forms = NOUNS[self.language][noun]
        if count == 1:
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

    def _ordinal_digits(self, m: re.Match, text: str, tags: _Tags, after_label: bool) -> str:
        value = int(m["ordinalvalue"])
        following = text[m.end():].lstrip(" \t\u00a0\u202f")
        if following[:1].isupper():  # "Bylo jich 5. Pak…": a number that ends the sentence
            case, gender, animacy = self._context(value, m.start(), m.end() - 1, text, tags, after_label, m)
            return self._cardinal(value, case, gender, animacy) + "."
        head = tags.head_after(m.end())
        case, gender, animacy, plural, doubt = self._agreement(head, m.start(), tags)
        words = self._ordinal(value, case or NOM, gender or "masculine", animacy or "inanimate", plural)
        if case is None or gender is None:
            self._warn(text, m, words, "no noun to agree with; nominative masculine inanimate")
        elif doubt:
            self._warn(text, m, words, f"{doubt}; check it")
        return words

    def _number(self, m: re.Match, text: str, tags: _Tags, after_label: bool) -> str:
        value, _, fraction = _parse(m["value"])
        if fraction:  # decimals are read in the nominative (Step 2.1)
            words = self._cardinal(value, NOM, "masculine", "inanimate")
        elif m["times"]:
            words = self._cardinal(value, NOM, "masculine", "inanimate") + "krát"
        else:
            words = self._cardinal(value, *self._context(value, m.start(), m.end(), text, tags,
                                                         after_label, m))
        if m.start() and text[m.start() - 1].isalpha():
            words = " " + words
        if m.end() < len(text) and text[m.end()].isalpha():
            self._warn(text, m, words, "digits glued to a word")
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
            if (tagged_case in PLURAL_ENDINGS and noun.feats.get("Number") == "Plur"
                    and not noun.text.lower().endswith(PLURAL_ENDINGS[tagged_case])):
                tagged_case = GEN  # "o 5 minút": a genitive plural tagged with the preposition's case
            if self._number_fits(value, noun):
                noun_case = tagged_case
        if noun_case in (DAT, INS, LOC):
            case = noun_case  # "s pěti přáteli"
        elif prep_case == GEN:
            case = GEN  # "bez pěti jablek", "do dvou hodin"
        elif prep_case:
            # "ve dvě hodiny", "v pět hodin": the tagger often gives v/na/o the locative here
            case = ACC if prep_case == ACC or tagged_case in (NOM, ACC, GEN) else prep_case
        elif noun is not None:
            # no governor: "Mám pět jablek" (the noun is genitive, the number nominative);
            # Slovak "videl troch mužov" (A = G of masculine personal nouns)
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
            prev = tags.before(start)
            prep_case = _UD_CASES.get(prev.feats.get("Case")) if prev is not None and prev.upos == "ADP" else None
            if prep_case and (case == NOM or (case == GEN and prep_case != GEN)):
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
            return gender, "personal" if self.language == "sk" else "animate"
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
        nxt = rest.lstrip(" \t\n\r\u00a0\u202f\"'„“”‚‘’«»‹›()[]—–-")
        if not nxt:
            return True
        if introduces or not nxt[:1].isupper():
            return False
        if roman:  # "Karel IV. Lucemburský" goes on; "Vládl Karel IV. Potom…" does not
            word = tags.after(end)
            return word is None or word.upos not in ("PROPN", "ADJ")
        return True

    def _vocalise(self, text: str, start: int, words: str) -> Tuple[int, str]:
        """"s 2 přáteli" -> "se dvěma přáteli": vocalise a one-letter preposition before the number words."""
        if start < 2 or text[start - 1] != " " or (start > 2 and text[start - 3].isalpha()):
            return start, words
        prep = text[start - 2]
        rule = VOCALISATION[self.language].get(prep.lower())
        if rule is None or not words.startswith(rule[1]):
            return start, words
        return start - 2, f"{_capitalise_like(prep, rule[0])} {words}"

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
