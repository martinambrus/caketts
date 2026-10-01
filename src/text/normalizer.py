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
        "sekunda": ("feminine", _forms("sekund", _CS_ZENA)),
        "kus": ("masculine", _forms("kus", _CS_HRAD)),
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
        "sekunda": ("feminine", _forms("sek", "unda,undy,unde,undu,undou,unde,undy,únd,undám,undy,undami,undách")),
        "kus": ("masculine", _forms("kus", ",a,u,,om,e,y,ov,om,y,mi,och")),
        "libra": ("feminine", _forms("lib", "ra,ry,re,ru,rou,re,ry,ier,rám,ry,rami,rách")),
        "euro": ("neuter", _forms("eur", _SK_MESTO)),
        "percento": ("neuter", _forms("percent", _SK_MESTO)),
        "promile": ("neuter", _forms("promile", _INDECLINABLE)),
        "penny": ("feminine", _forms("pen", "ny,ny,ny,ny,ny,ny,ce,cí,ciam,ce,cami,ciach")),
    },
}

UNITS = {  # symbol -> noun in NOUNS
    "cs": {"km": "kilometr", "km/h": "kilometr", "m": "metr", "m/s": "metr", "cm": "centimetr", "mm": "milimetr",
           "kg": "kilogram", "g": "gram", "l": "litr", "L": "litr", "ml": "mililitr", "mL": "mililitr",
           "°C": "stupeň", "°": "stupeň",
           "%": "procento", "‰": "promile", "hod": "hodina", "h": "hodina", "min": "minuta", "s": "sekunda",
           "ks": "kus", "Kč": "koruna",
           "€": "euro", "EUR": "euro", "$": "dolar", "USD": "dolar", "£": "libra"},
    "sk": {"km": "kilometer", "km/h": "kilometer", "m": "meter", "m/s": "meter", "cm": "centimeter",
           "mm": "milimeter",
           "kg": "kilogram", "g": "gram", "l": "liter", "L": "liter", "ml": "mililiter", "mL": "mililiter",
           "°C": "stupeň", "°": "stupeň",
           "%": "percento", "‰": "promile", "hod": "hodina", "h": "hodina", "min": "minúta", "s": "sekunda",
           "ks": "kus", "Kč": "koruna",
           "€": "euro", "EUR": "euro", "$": "dolár", "USD": "dolár", "£": "libra"},
}
UNIT_WORDS = {language: {part.lower() for key in units for part in re.findall(r"[^\W\d_]+", key)}
              for language, units in UNITS.items()}
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
SIGNS = {"cs": {"&": "a", "+": "plus", "@": "zavináč", "=": "rovná se", "×": "krát", "x": "krát", "*": "krát",
               "±": "plus minus", "#": "číslo", "−": "mínus", "~": "přibližně", "≈": "přibližně"},
         "sk": {"&": "a", "+": "plus", "@": "zavináč", "=": "rovná sa", "×": "krát", "x": "krát", "*": "krát",
               "±": "plus mínus", "#": "číslo", "−": "mínus", "~": "približne", "≈": "približne"}}
PER_UNITS = {  # "100 Kč/kg" -> "za kilogram": the unit after "/" in the accusative singular
    "cs": {"kg": "kilogram", "g": "gram", "l": "litr", "L": "litr", "ml": "mililitr", "mL": "mililitr", "m": "metr",
           "km": "kilometr",
           "cm": "centimetr", "mm": "milimetr",
           "ks": "kus", "hod": "hodinu", "h": "hodinu", "min": "minutu", "s": "sekundu"},
    "sk": {"kg": "kilogram", "g": "gram", "l": "liter", "L": "liter", "ml": "mililiter", "mL": "mililiter", "m": "meter",
           "km": "kilometer",
           "cm": "centimeter", "mm": "milimeter",
           "ks": "kus", "hod": "hodinu", "h": "hodinu", "min": "minútu", "s": "sekundu"},
}
SLASH_WORDS = {"cs": {"word": "nebo", "number": "lomeno"}, "sk": {"word": "alebo", "number": "lomené"}}
DOT_WORDS = {"cs": "tečka", "sk": "bodka"}  # "1.2.3", "192.168.1.1": read part by part
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
# plurals of neuter dítě, oko, ucho that decline as feminines; a 1 sharing them is neuter: "jedno nebo dvě děti"
NEUTER_PLURALS = {
    "cs": frozenset("děti dětí dětem dětmi dětech oči očí očím očima očích uši uší uším ušima uších".split()),
    "sk": frozenset("deti detí deťom deťmi deťoch oči očí očiam očami očiach uši uší ušiam ušami ušiach".split()),
}

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
DAY_TIMES = ("ráno", "dopoledne", "odpoledne", "večer")  # "ve dvě večer": a clock time, not the noun counted
# a Slovak clock time after these has an ordinal hour ("o druhej", "pred druhou"); a duration keeps the
# cardinal ("za dve pätnásť")
# verbs of placing, after which a number that ends the sentence is a rank: "Skončil 2." -> "druhý"
RANK_VERBS = {"cs": ("skončil", "doběhl", "dojel", "doplaval", "umístil", "byl"),
              "sk": ("skončil", "dobeh", "doplával", "umiestnil", "bol")}
SK_CLOCK_PREPOSITIONS = frozenset({"o", "po", "pred", "okolo", "od", "do", "medzi", "k", "ku", "na"})

# Czech prepositions with the accusative of direction and the instrumental of place. A masculine plural in -y
# has one form for both ("Postavil se mezi dva stromy", "Stál mezi dvěma stromy"), so the verb decides;
# the tagger cannot. Verbs are matched by the start of the word form, as the tagger gives no lemma.
TWO_CASE_PREPOSITIONS = frozenset({"mezi", "nad", "pod", "před", "za"})
TIME_PLURALS = frozenset({"roky", "dny", "týdny"})  # "před dvěma roky" (ago), "za dva roky" (in)
# a time after a genitive verb is its object ("Dožil se devadesáti let") or a duration ("Bál se pět minut")
DURATION_GENITIVES = frozenset({"vteřin", "sekund", "minut", "hodin", "dní", "dnů", "týdnů", "měsíců", "let", "roků"})
# genitives of time, which are no object: "Pěti vítězství dosáhla minulého roku"
TIME_GENITIVES = frozenset({"dne", "dnu", "roku", "týdne", "měsíce", "večera", "rána", "dopoledne", "odpoledne", "léta",
                            "jara", "podzimu", "zimy", "noci", "času", "víkendu", "chvíle", "doby"})
# events, which one takes part in, after a number: "Zúčastní se pěti závodů"
EVENT_GENITIVES = frozenset({"akcí", "bitev", "debat", "diskusí", "etap", "festivalů", "her", "koncertů", "konferencí",
                             "kongresů", "kol", "kurzů", "mistrovství", "olympiád", "porad", "přednášek", "schůzí",
                             "seminářů", "sjezdů", "soutěží", "turnajů", "válek", "voleb", "výstav", "zápasů", "závodů"})
# nouns of time whose nominative is also their accusative of time, so no subject: "Každý rok se akce zúčastní"
TIME_NOUNS = frozenset({"rok", "den", "týden", "měsíc", "večer", "čas", "víkend", "okamžik", "moment", "podzim",
                        "život", "léto", "jaro", "ráno", "noc"})
# words that qualify a number after "a" without opening a clause of their own: "Petr a asi pět mužů", "s 2 kg a ~3 kg"
APPROXIMATORS = frozenset({"~", "≈", "asi", "přibližně", "zhruba", "skoro", "téměř", "cca", "nejméně", "nejvýše",
                           "alespoň", "aspoň", "až", "také", "též", "ještě", "jen", "pouze", "približne", "takmer",
                           "najmenej", "ešte", "tiež", "aj", "len", "iba"})
_APPROXIMATOR = "|".join(map(re.escape, sorted(APPROXIMATORS, key=len, reverse=True)))
DIRECTION_VERBS =("postav", "polož", "vlož", "hodil", "hodí", "pověs", "schoval", "schová", "klesl", "klesá",
                   "klesn", "spadl", "spadn", "padl", "padá", "vstoup", "vešel", "vejd", "vjel", "vjed", "rozděl",
                   "zařad", "stoupl", "stoupá", "vystoup", "posad", "sedl", "lehl", "lehn", "umísti", "umísť",
                   "vrátil", "zapadl", "vlezl", "vběhl", "přiš", "přijd", "dal", "dá", "šel", "šla", "šli", "jde",
                   "jdou", "jel", "jela", "jeli", "jede", "jedou")
# going with no prefix marks no destination: "jel za 2 vozy" follows them ("za dvěma vozy"); "zajel za roh" does not
GOING_VERBS = ("šel", "šla", "šli", "jde", "jdou", "jel", "jela", "jeli", "jede", "jedou")
# passives of placing keep their verb's destination: "byl položen mezi 2 svazky" -> "dva"; "postaven" (built) is a place,
# and CAC tags these participles adjectives, so the copula would be the verb
PLACED_PARTICIPLES = ("polož", "vlož", "pověš", "umístěn", "zařazen", "schován")
PLACE_VERBS = ("stál", "stoj", "lež", "seděl", "sedí", "sedě", "vis", "bydl", "žil", "žij", "zůstal", "zůstáv",
               "čekal", "čeká", "nacház", "rostl", "rost", "pracoval", "pracuj", "spal", "spí")
PLACE_FORMS = frozenset({"je", "jsou", "není", "byl", "byla", "bylo", "byli", "byly", "bude", "budou"})  # not "jel"
# Czech verbs that take the genitive (a pattern for the start of the form) with the clitic they need:
# "Dosáhli jsme XXI. století", "Bál se 2. dílu"; vzdát se, not vzdálit se (vzdálil, vzdaluje)
GENITIVE_VERBS = {"dosáh": None, "dosahov": None, "dosahuj": None, "dožil": "se", "dožij": "se", "dočkal": "se",
                  "dočká": "se", "dočkaj": "se", "vzd(?:al(?!ov|uj)|aj|á(?!l))": "se", "zúčastn": "se", "účastn": "se",
                  "bál": "se", "bojí": "se", "obával": "se", "obává": "se", "obávaj": "se", "všiml": "si", "všimn": "si",
                  "všímá": "si", "všímaj": "si", "dotkl": "se", "dotkn": "se", "dotýk": "se", "zbavil": "se", "zbav": "se",
                  "týká": "se", "týkaj": "se", "týkal": "se", "týče": "se",  # "obávají se", "dočkají se", "co se týče"
                  "(?:ze|vy|o)?pt(?:al|á|aj|ej|at)": "se", "dot(?:áz|áž|az)": "se"}
# Czech verbs with "na" and the accusative, not the locative the tagger gives "na": "Vzpomínal na XX. století"
NA_ACCUSATIVE_VERBS = ("vzpomín", "vzpomněl", "vzpomene", "myslel", "myslí", "čekal", "čeká", "těšil", "těší",
                       "zapomněl", "zapomín", "díval", "dívá", "podíval", "spoléh", "spolehl", "upozorn", "narazil",
                       "naráží", "odkaz", "odkázal")
NA_PLACE_VERBS = ("čekal", "čeká")  # "čekat na" is also "wait at": "Čekal na 2. náměstí" may be locative

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
# abbreviations that close a list, whose commas and "a" join no clauses: "PŘINESL JABLKA, HRUŠKY ATD."
LIST_END_ABBREVIATIONS = {"atd.", "apod.", "aj.", "atp.", "atď.", "a pod.", "a i."}
# keys whose capital form is an acronym, a name or initials: "TURNAJ ATP.", "TJ SOKOL", "voliči ODS."
CAPITAL_ACRONYMS = frozenset({"aj.", "atp.", "max.", "mj.", "n.l.", "ods.", "t.j.", "tj.", "vr."})
# nouns that a Roman numeral may number from behind ("díl V."), also before a genitive: "díl V. knihy"
ROMAN_LABEL_NOUNS = {"cs": frozenset({"díl", "svazek", "kapitola", "část", "oddíl", "ročník", "kniha", "sešit"}),
                     "sk": frozenset({"diel", "zväzok", "kapitola", "časť", "oddiel", "ročník", "kniha", "zošit"})}
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
_MATH_SIGNS = frozenset("×*=+±/−")
# no sound, so dropped or replaced on input: soft hyphen, zero-width space, word joiner, BOM, hyphen variants
_TYPOGRAPHY = str.maketrans({"\u00ad": None, "\u200b": " ", "\u2060": None, "\ufeff": None,
                             "\u2010": "-", "\u2011": "-", "\u2012": "–", "\u2015": "—", "\u2044": "/"})
_SPAN_RE = re.compile(r"<(cs|sk|en)>(.*?)</\1>", re.DOTALL)
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}

_HS = r"[ \t\u00a0\u202f]"  # horizontal space: no item may swallow a line break
_INT = r"[1-9]\d{0,2}(?:[ \u00a0\u202f]\d{3})+(?!\d)|[1-9]\d{0,2}(?:\.\d{3})+(?!\d|\.\d)|\d+"  # 10 000, 10.000
# a minus sign starts after a space, bracket, quote or operator: „-5 °C“, "=-5"; not after a letter,
# digit or period: "COVID-19", "5-3", "1.-5."
_SIGN_START = r"(?:(?<=\dx)|(?<![^\s(\[{\"'„“”‚‘’«»‹›=:×*~≈/+]))"  # also "3x-4", "3*-4", "~-5", but not "Max-5"
_EN_AMOUNT = r"\d{1,3}(?:(?:,\d{3}){2,}(?:\.\d+)?|,\d{3}\.\d+)(?!\d)"  # "1,234.56 USD": never a Czech decimal
_UNSIGNED = rf"(?:{_EN_AMOUNT}|(?:{_INT})(?:[.,]\d+)?)"
_AMOUNT = rf"(?:{_SIGN_START}[-−–](?=\d))?{_UNSIGNED}"  # "–5 °C": typeset text uses – for minus
_POWER = r"(?:[²³]|[23](?!\d))?"  # m², and m2 as typed
# what ~, ≈ or * is read before: "~5", "≈ -$5", "3*$4", "3 * + $4"; a spaced hyphen is a dash, not a minus
_OPERAND = rf"{_HS}*(?:[+±]{_HS}*|[-−–])?(?:[$€£]{_HS}*(?:[+±]{_HS}*|[-−–])?)?\d"
# "Kč/m²", "m / s"; not "Kč / s DPH"; symbols of two letters or more also in capitals: "KG", "KM/H"
_PER = rf"(?i:kg|ks|km{_POWER}|cm{_POWER}|mm{_POWER}|ml|hod|min)|g|l|L|m{_POWER}|h|s(?!{_HS}+[^\W\d_])"
_CAPITAL_PER = rf"|G|M{_POWER}|H|S(?!{_HS}+[^\W\d_])"  # one-letter symbols in capitals, in all-caps text only
_EN_GROUPED = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?"  # "$1,234.56" after a prefixed currency symbol
_PRICE = rf"[-−–+]?(?:{_EN_GROUPED}|(?:{_INT})(?:[.,]\d+)?)"
_TAG_TOKEN = re.compile(rf"{_INT}|[^\W\d_]+|\S")  # "1 000" is one token: split, "000" misleads the tagger
_UNIT = (rf"(?i:km/h|km{_POWER}|cm{_POWER}|mm{_POWER}|m/s|kg|ks\.?|ml|hod\.?|min\.?)|m{_POWER}|g|l|L|°C|°|%|‰|h\.?"
         rf"|s(?!{_HS}+[^\W\d_])\.?")  # "5 s.", but "Mám 5 s sebou"
# in a paragraph in capitals only: SI writes these lowercase, and in mixed text a capital is a prefix or a name
_CAPITAL_UNIT = rf"|M(?:[²³]|[23](?!\d))|(?<={_HS})(?:M{_POWER}|G|H\.?|S(?!{_HS}+[^\W\d_])\.?)"  # "5 G", "60M2"; "5G" a name
_CURRENCY = r"(?i:kč)|€|EUR|USD|\$|£"
_ROMAN = r"(?=[IVXLCDM])M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})"  # up to 3999
_NOT_LETTER_AFTER = r"(?![^\W\d_])"
_NOT_LETTER_BEFORE = r"(?<![^\W\d_])"
_ONE_LETTER_WORDS = "aikosuvz"
_INCLUSIVE_SUFFIXES = "kyně|yně|čka|čky|ka|ky|ce|a|á|é|y"  # "on/a", "Vážený/á", "student/ka", "přišli/y"
# "14.30" is a time only when hod follows, also after more times: "15.30–16.00 hod.", "v 8.30 a 9.30 hod."
_HOUR_WORD = r"(?:hod(?:\.|in[ay]?|ín)?|h\.?)"  # hod., hodin, hodiny, hodina, h., sk hodín
_DOT_TIME = (rf"(?=[0-5]\d(?:(?:{_HS}*[–—,-]{_HS}*|{_HS}+(?:do|až|a|nebo|alebo){_HS}+)(?:2[0-4]|[01]?\d)[.:][0-5]\d)*"
             rf"{_HS}*{_HOUR_WORD}{_NOT_LETTER_AFTER})")
# a range written with "až" ("1 až 2 °C"), but not before a time ("8 až 9.30 hod."), which the time item reads
_TIME_AHEAD = rf"(?:2[0-4]|[01]?\d)(?::|\.{_DOT_TIME})[0-5]\d"
_SPACES = re.compile(r"\s*")  # to the next item; inside a paragraph a line break is a space too
# a number glued to an adjective is its first part: "25letý", "3denní", sk "5-ročný"
_ADJECTIVE_ENDINGS = "ieho|iemu|ého|ému|ých|ými|ími|ích|ém|ým|ím|om|ou|ej|ia|ie|iu|ý|á|é|í|ú"
_LETTER_BEFORE = re.compile(rf"{_NOT_LETTER_BEFORE}([^\W\d_]){_HS}+$")  # "s 2", also with a no-break space
_NUMBER_BEFORE = re.compile(rf"(?:\d\.?|[IVXLCDM]\.|\d[:.]\d\d{_HS}*{_HOUR_WORD})$")  # a dash between these reads "až"
_NUMBER_AFTER = re.compile(r"\d|[IVXLCDM]+\.")
_DATE_BEFORE = re.compile(rf"(?<![\d.])(?:3[01]|[12]\d|0?[1-9])\.{_HS}*(?:1[0-2]|0?[1-9])\.{_HS}+$")  # "5. 6. "
_RANGE_AHEAD = re.compile(rf"{_HS}*[–—-]{_HS}*(?:[-−–+]?\d|[IVXLCDM]+\.)")


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


def _items_pattern(abbreviations, capitals: bool = False) -> re.Pattern:
    """capitals: for a paragraph in capitals, where "5 M" and "5 G" are units too."""
    abbr = "|".join(_abbreviation_pattern(k) for k in sorted(abbreviations, key=len, reverse=True))
    unit = _UNIT + (_CAPITAL_UNIT if capitals else "")
    per = _PER + (_CAPITAL_PER if capitals else "")
    length = rf"(?i:km|cm|mm){_POWER}|(?i:ml)|m{_POWER}|l|L" + (rf"|(?<={_HS})M{_POWER}|M(?:[²³]|[23](?!\d))"
                                                                if capitals else "")
    # nor does the upper end after "až" run into a word ("2 až 3krát", "2 až 3letý") or an ordinal's noun ("5 až
    # 6. den"); a capital after the period starts a sentence ("1 až 2. Potom"), but in capitals it may be "6. DEN"
    after_period = r"[^\W\d_]" if capitals else "[a-záäčďéěíĺľňóôŕřšťúůýž]"
    range_end = rf"(?![^\W_]|[.,:]\d|\.{_HS}*{after_period})"

    def unit_per(name: str) -> str:  # after a length or volume, "/ s" is a second also before a word: "5 m / s a pak"
        return (rf"(?P<{name}unit>(?P<{name}length>{length})(?![\w²³]|/(?!(?i:s){_NOT_LETTER_AFTER}))|{unit}|{_CURRENCY})"
                rf"{_NOT_LETTER_AFTER}"  # a length before "/s", not "km/h" or "m/s", which are units of their own
                rf"(?:{_HS}*/{_HS}*(?P<{name}per>{per}|(?({name}length)(?i:s)|(?!))){_NOT_LETTER_AFTER}\.?)?")

    return re.compile(
        rf"(?P<isodate>(?<![\d.,-])(?P<isoyear>\d{{4}})-(?P<isomonth>0[1-9]|1[0-2])-(?P<isoday>0[1-9]|[12]\d|3[01])(?![\d-]))"
        rf"|(?P<date>(?<!\d)(?P<day>3[01]|[12]\d|0?[1-9])\.{_HS}*(?P<month>1[0-2]|0?[1-9])\."
        rf"(?:{_HS}*(?P<year>\d{{4}})(?!\d)|{_HS}*(?P<shortyear>(?<=\.)\d{{2}}"
        rf"|(?<={_HS})(?:0\d|\d{{2}}(?!\d)(?!{_HS}*[^\W\d_])))(?!\d))?(?!\d))"  # "5. 6. 05", not "5. 6. 24 lidí"
        rf"|(?P<dotted>(?<![\d.,])(?![1-9]\d{{0,2}}(?:\.\d{{3}})+(?!\d|\.\d))\d+(?:\.\d+){{2,}}(?!\d))"  # "1.2.3"
        rf"|(?P<commas>(?P<commalist>(?<![\d.,])(?!{_EN_AMOUNT})\d+(?:,\d+){{2,}}(?!\d))"  # "1,2,3": no number has two commas
        rf"(?:{_HS}*(?:(?P<commascale>(?i:tis|mil|mld))\.?{_NOT_LETTER_AFTER}(?:{_HS}+{unit_per('commascale')})?"
        rf"|{unit_per('comma')}))?)"  # "1,2,3 kg", "1,2,3 Kč/kg", "1,2,3 tis. Kč"
        rf"|(?P<time>(?<![\d.,:])(?P<hour>2[0-4]|[01]?\d)(?::|\.{_DOT_TIME})(?P<minute>[0-5]\d)(?::(?P<second>[0-5]\d))?(?![\d:])"
        rf"(?:{_HS}*{_HOUR_WORD}{_NOT_LETTER_AFTER})?)"
        rf"|(?P<range>(?<![\d.,])(?P<low>(?:{_SIGN_START}[-−–])?{_UNSIGNED})"
        rf"(?:{_HS}*(?:(?P<lowscale>(?i:tis|mil|mld))\.?{_NOT_LETTER_AFTER}"
        rf"(?:{_HS}+{unit_per('lowscale')})?"
        rf"|{unit_per('low')}))?(?:{_HS}*[–—-]{_HS}*|{_HS}+(?P<rangeword>až|AŽ){_HS}+(?!{_TIME_AHEAD}))"
        rf"(?P<high>[-−–+]?{_UNSIGNED})(?(rangeword){range_end})"
        rf"(?(lowscale){_HS}*(?P<highscale>(?i:tis|mil|mld))\.?{_NOT_LETTER_AFTER}"
        rf"(?:{_HS}+{unit_per('highscale')})?"
        rf"|(?(lowunit){_HS}*{unit_per('high')}"
        rf"|(?:{_HS}*(?:(?P<rangescale>(?i:tis|mil|mld))\.?{_NOT_LETTER_AFTER}"
        rf"(?:{_HS}+{unit_per('rangescale')})?"
        rf"|{unit_per('range')}))?)))"
        rf"|(?P<money>(?P<moneysign>{_SIGN_START}[-−–])?(?P<symbol>[€$£]){_HS}*(?P<price>{_PRICE})"
        rf"(?:{_HS}*/{_HS}*(?P<lowmoneyper>{per}){_NOT_LETTER_AFTER}\.?{_HS}*[–—-]{_HS}*(?P<highpersign>[-−–])?"
        rf"(?:(?P=symbol){_HS}*)?"
        rf"(?P<highprice>{_PRICE}){_HS}*/{_HS}*(?P<highmoneyper>{per}){_NOT_LETTER_AFTER}\.?"
        rf"|(?:(?:{_HS}+(?P<lowmoneyscale>(?i:tis|mil|mld))\.?{_NOT_LETTER_AFTER})?"
        rf"(?:{_HS}*[–—-]{_HS}*|{_HS}+(?P<moneyword>až|AŽ){_HS}+(?!{_TIME_AHEAD}))"
        rf"(?P<highsign>[-−–])?(?:(?P=symbol){_HS}*)?(?P<pricehigh>{_PRICE})(?(moneyword){range_end}))?"  # "-$5–-$10"
        rf"(?(lowmoneyscale){_HS}+(?P<highmoneyscale>(?i:tis|mil|mld))\.?{_NOT_LETTER_AFTER}"
        rf"|(?:{_HS}+(?P<moneyscale>(?i:tis|mil|mld))\.?{_NOT_LETTER_AFTER})?)"
        rf"(?:{_HS}*/{_HS}*(?P<moneyper>{per}){_NOT_LETTER_AFTER}\.?)?))"
        rf"|(?P<measure>(?P<amount>{_AMOUNT})(?P<whole>,[-–—])?{_HS}*"
        rf"(?:(?P<scale>(?i:tis|mil|mld))\.?(?:{_HS}+{unit_per('scale')})?"
        rf"|{unit_per('')})"
        rf"{_NOT_LETTER_AFTER})"
        rf"|(?P<ordinal>(?<![\d.,])(?P<ordinalvalue>{_INT})\.(?={_HS}*(?:[^\W\d_]|[–—-]{_HS}*\d|,{_HS}*\d+\.)))"
        rf"|(?P<number>(?P<value>{_AMOUNT})(?:(?P<times>krát|x|×){_NOT_LETTER_AFTER}(?!{_HS}*[-−–+]?\d)"
        rf"|-?(?P<compound>[^\W\d_]*?(?:{_ADJECTIVE_ENDINGS})){_NOT_LETTER_AFTER})?)"
        rf"|(?P<abbreviation>{abbr})"
        rf"|(?P<roman>{_NOT_LETTER_BEFORE}(?P<numeral>{_ROMAN})\.)"
        rf"|(?P<sign>[&+@=×±−]|#(?={_HS}*\d)"
        rf"|(?<=\d){_HS}*x{_NOT_LETTER_AFTER}{_HS}*"  # "3 x 4", "3    x 4", "3 x týdně"
        rf"|(?<=\d){_HS}*\*{_HS}*(?={_OPERAND})|[~≈](?={_OPERAND}))"  # "3*4", "~5 km", "≈$5", "3≈4"
        rf"|(?P<slash>(?<=[^\W\d_]{{2}})/(?P<suffix>{_INCLUSIVE_SUFFIXES}){_NOT_LETTER_AFTER}"
        rf"|(?:(?<=[^\W\d_]{{2}})|(?<={_NOT_LETTER_BEFORE}[{_ONE_LETTER_WORDS}{_ONE_LETTER_WORDS.upper()}]))"
        rf"{_HS}*/{_HS}*(?=[^\W\d_]{{2}}|[{_ONE_LETTER_WORDS}{_ONE_LETTER_WORDS.upper()}]{_NOT_LETTER_AFTER})"
        rf"|(?<={_NOT_LETTER_BEFORE}[{_ONE_LETTER_WORDS}{_ONE_LETTER_WORDS.upper()}]){_HS}*/{_HS}*"
        rf"(?=[{_ONE_LETTER_WORDS}{_ONE_LETTER_WORDS.upper()}]{_NOT_LETTER_AFTER})"  # "a/i", "v/z"
        rf"|(?<=\d){_HS}*/{_HS}*(?=[-−–+]?\d))"
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
    s = re.sub(r"[ \u00a0\u202f]", "", amount).replace("−", "-").replace("–", "-").removeprefix("+")
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
    return price.replace(",", "") if re.fullmatch(rf"[-−–+]?{_EN_GROUPED}", price) else price


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


def _clean_typography(text: str) -> str:
    """Drop invisible characters and normalise hyphen variants outside <cs>/<sk>/<en> spans, which stay as written."""
    out, pos = [], 0
    for s in _SPAN_RE.finditer(text):
        out += [text[pos:s.start()].translate(_TYPOGRAPHY), s.group(0)]
        pos = s.end()
    return "".join(out) + text[pos:].translate(_TYPOGRAPHY)


def _all_capitals(text: str) -> bool:
    """Whether a paragraph is written in capitals: no lowercase letter outside <cs>/<sk>/<en> spans."""
    letters = [ch for ch in _SPAN_RE.sub(" ", text) if ch.isalpha()]
    return len(letters) > 1 and not any(ch.islower() for ch in letters)


def _canonical(symbol: str, keys) -> str:
    """The table key of a unit symbol written in capitals: "KČ" -> "Kč", "KM/H" -> "km/h"."""
    return symbol if symbol in keys else next((k for k in keys if k.lower() == symbol.lower()), symbol)


def _agree(a: "_Word", b: "_Word", values: bool = False) -> bool:
    """The same number, gender and animacy where both words have one: "dosáhli" agrees with "muži", not "cíle";
    a present verb has no gender ("dosáhnou"). With `values` one shared value of a multi-valued tag is enough:
    "dosáhla" (Fem,Neut and Plur,Sing) agrees with "družstva"."""
    return all(None in (a.feats.get(f), b.feats.get(f))
               or (set(a.feats[f].split(",")) & set(b.feats[f].split(",")) if values else a.feats[f] == b.feats[f])
               for f in ("Number", "Gender", "Animacy"))


def _plain_nominative(w: "_Word") -> bool:
    """A word tagged nominative in a form no genitive singular shares, and no accusative of time ("každý rok", "akce"):
    a masculine animate ("Petr", "Jiří"; not "soudce"), a feminine in -a or a consonant ("účast", "píseň"; not "inflace"),
    a singular "stroj" or "auto", "on", "kdo"."""
    form, feats = w.text.lower(), w.feats
    singular, gender = feats.get("Number") == "Sing", feats.get("Gender")
    return feats.get("Case") == "Nom" and form.isalpha() and (
        form in ("on", "ono", "oni", "ony", "kdo", "někdo", "nikdo")
        or (gender == "Masc" and feats.get("Animacy") == "Anim" and not form.endswith(("e", "ě"))
            and not (form.endswith("í") and feats.get("Number") == "Plur"))
        or (gender == "Fem" and singular and (form.endswith("a") or form[-1] not in "aáeéěiíoóuúůyý")
            and form not in TIME_NOUNS)
        or (w.upos in ("NOUN", "PROPN") and singular and form not in TIME_NOUNS
            and ((gender == "Masc" and feats.get("Animacy") == "Inan") or (gender == "Neut" and form.endswith("o")))))


def _affirmative(form: str) -> str:
    """A verb form without its negation, for the tables of verb stems: "nedosáhl" -> "dosáhl"; "nesl" stays."""
    return form[2:] if form.startswith("ne") and len(form) > 4 else form


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

    def __init__(self, words: List[_Word], capitals: bool = False):
        self.words, self.capitals = words, capitals
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
        adverb = modifier = False
        for w in self.words[i:]:
            if w.upos in ("NOUN", "PROPN"):
                return None if adverb else w
            if w.upos == "CCONJ" or w.text == ",":
                if not modifier:
                    return None  # "Zůstali 2, ale staré ženy odešly"; but "2 červené a modré knihy"
                adverb = True
            elif w.upos in ("ADV", "PART"):  # "2 velmi staré knihy"
                adverb = True
            elif w.upos in ("ADJ", "DET"):
                adverb, modifier = False, True
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
        self._items_in_capitals = _items_pattern(self.abbreviations, capitals=True)
        books = sorted(filter(None, config.get("verse_references") or []), key=len, reverse=True)
        numbered = [book for book in books if re.search(r"\d", book)]
        if numbered:
            raise ValueError(f"book_config['verse_references'] has {numbered}, whose digits would stay as written; "
                             f"list the name alone ('Jan' for '1 Jan'), and the number before it is read as usual")
        # "Jan 3,16", "Mt 5,3–12", "Jan 3:16", "Mt 5,3–7,29", "Jan 3,16.18": chapter and verse, not a decimal or
        # a time; the book may be in its span ("<en>John</en> 3,16"); atomic, so "Jan 3,16–18a" is not cut short
        names = '|'.join(map(re.escape, books))
        self._verses = (re.compile(rf"(?<!\w)(?P<book><(?P<booklang>en|cs|sk)>(?:{names})</(?P=booklang)>|(?:{names}))"
                                   rf"{_HS}+(?P<chapter>\d+)[,:]"
                                   rf"(?P<verse>\d+)(?>(?:{_HS}*[–—-]{_HS}*(?P<last>\d+)(?:,(?P<lastverse>\d+))?)?"
                                   rf"(?P<more>(?:\.\d+(?:{_HS}*[–—-]{_HS}*\d+)?)*))(?!\d|[,.:]\d|[^\W\d_])")
                        if books else None)
        phrases = sorted((p.translate(_TYPOGRAPHY) for p in config.get("english") or [] if p), key=len, reverse=True)
        self._english = (re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, phrases)) + r")(?!\w)")
                         if phrases else None)

    # ---- public API ----------------------------------------------------------------------
    def normalize(self, text: str) -> str:
        lines = _clean_typography(text).split("\n")
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
        pattern = self._items_in_capitals if _all_capitals(text) else self._items
        verses = list(self._verses.finditer(text)) if self._verses else []
        items = sorted(verses + [m for m in pattern.finditer(text)
                                 if not any(v.start() < m.end() and m.start() < v.end() for v in verses)],
                       key=lambda m: m.start())
        items = [m for m in items if not any((s < m.end() and m.start() < e)  # a verse may hold its book's span
                                             and not (m.re is self._verses and m.start() <= s and e <= m.end())
                                             for s, e in spans)]
        tags = _Tags(self._tag(text) if any(self._needs_tags(m) for m in items) else [], _all_capitals(text))
        out, pos, after_label, goes_on = [], 0, -1, -1
        for m in items:
            start, words = self._resolve(m, text, tags, after_label == m.start(), heading)
            source_case = m.lastgroup == "abbreviation" and m.group(0)[0].isalpha()
            if (start == m.start() and not source_case and start != goes_on
                    and _starts_sentence("".join(out) + text[pos:start])):
                words = words[:1].upper() + words[1:]  # "5 lidí přišlo." -> "Pět lidí přišlo."
            if text[m.end():m.end() + 1].isalnum() and not words[-1:].isspace():
                words += " "  # "§5", "č.5", "5.díl"
            if start == m.start() and text[start - 1:start].isalnum() and words[:1].isalnum():
                words = " " + words  # "v14:30", "dne1.1.2024"
            if start == pos and out and out[-1].endswith(" ") and words.startswith(" "):
                words = words[1:]  # "3x4", "Cca5": both replacements brought a space
            out += [text[pos:start], words]
            pos = m.end()
            if (m.lastgroup == "abbreviation" and words == m.group(0) and words.endswith(".") and tags.words
                    and not self._ends_sentence(text, m.start(), m.end(), tags) and not self._verb_before(m.start(), tags, False)):
                goes_on = _SPACES.match(text, pos).end()  # "Turnaj ATP. 500 začal": a kept acronym's period, no verb yet
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

    def _verse(self, v: re.Match) -> str:
        """"Jan 3,16" -> "Jan tři, šestnáct", "Mt 5,3–12" -> "Mt pět, tři až dvanáct", "Jan 3,16.18" -> "Jan
        tři, šestnáct a osmnáct", for the books of book_config["verse_references"]."""
        def number(digits: str) -> str:
            return self._cardinal(int(digits), *self._label(int(digits)))

        words = f"{v['book']} {number(v['chapter'])}, {number(v['verse'])}"
        if v["last"]:
            words += f" {RANGE_WORD} {number(v['last'])}"
        if v["lastverse"]:
            words += f", {number(v['lastverse'])}"
        more = re.findall(rf"\.(\d+)(?:{_HS}*[–—-]{_HS}*(\d+))?", v["more"])
        for i, (first, last) in enumerate(more):
            words += (" a " if i == len(more) - 1 else ", ") + number(first) + (f" {RANGE_WORD} {number(last)}" if last else "")
        return words

    def _needs_tags(self, m: re.Match) -> bool:
        if m.re is self._verses:
            return False
        if m.lastgroup in ("abbreviation", "date") and m.group(0).endswith(".") and _all_capitals(m.string):
            return True  # in capitals only the tags tell whether the period ends the sentence: "ATD. ODEŠEL"
        if m.lastgroup == "abbreviation":
            return _key_of(m) in AGREEING_ABBREVIATIONS or _key_of(m) in DECLINED_ABBREVIATIONS[self.language]
        if m.lastgroup == "commas":
            # the case of "s 1,2,3 kg", and the noun of "s 1,2,3 přáteli"
            return bool(m["commaunit"] or m["commascale"] or re.match(rf"{_HS}*[^\W\d_]", m.string[m.end():]))
        return m.lastgroup not in ("isodate", "date", "dotted", "sign", "slash", "dash", "ellipsis")

    def _tag(self, text: str) -> List[_Word]:
        view = _SPAN_RE.sub(lambda s: " " * (s.start(2) - s.start()) + s.group(2) + " " * (s.end() - s.end(2)),
                            text)
        tokens = list(_TAG_TOKEN.finditer(view))
        if not tokens:
            return []
        texts = [t.group() for t in tokens]
        letters = [i for i, s in enumerate(texts) if s.isalpha() and not re.fullmatch("[IVXLCDM]+", s)]
        capitals = {i for i in letters if texts[i].isupper()}
        # the tagger reads every word of an all-caps run as a caseless proper noun; an acronym alone stays
        lowered = {i for pair in zip(letters, letters[1:]) if set(pair) <= capitals for i in pair}
        sentence = _tagger(self.language)([[s.lower() if i in lowered else s for i, s in enumerate(texts)]]).sentences[0]
        words = []
        for i, (t, token) in enumerate(zip(tokens, sentence.tokens)):
            for w in token.words:
                feats = dict(f.split("=", 1) for f in (w.feats or "").split("|") if f)
                feats.update(FEATURE_FIXES[self.language].get(w.text.lower(), {}))
                # CAC tags "–" as a noun at times; "~" and "≈" read "přibližně", so a preposition reaches past them
                upos = "ADV" if texts[i] in ("~", "≈") else (w.upos if any(ch.isalnum() for ch in w.text) else "PUNCT")
                words.append(_Word(t.start(), t.end(), texts[i] if i in lowered else w.text, upos, feats))
        return words

    # ---- items -----------------------------------------------------------------------------
    def _resolve(self, m: re.Match, text: str, tags: _Tags, after_label: bool, heading: bool) -> Tuple[int, str]:
        """(start of the replaced text, replacement) for one item."""
        kind, start, end = m.lastgroup, m.start(), m.end()
        if m.re is self._verses:
            return start, self._verse(m)
        if kind == "dash":
            if _NUMBER_BEFORE.search(text[:start].rstrip()) and _NUMBER_AFTER.match(text[end:].lstrip()):
                return start, _spaced(text, start, end, RANGE_WORD)  # "10:00–12:00", "XIX.–XX. století"
            return start, "—"
        if kind == "ellipsis":
            return start, "…"
        if kind == "dotted":
            return start, f" {DOT_WORDS[self.language]} ".join(
                self._cardinal(int(part), *self._label(int(part))) for part in m.group(0).split("."))
        if kind == "sign":
            return start, _spaced(text, start, end, SIGNS[self.language][m.group(0).strip()])
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
            if year is None and self._ends_sentence(text, start, end, tags):
                words += "."
        elif kind == "commas":
            parts = m["commalist"].split(",")
            if not (m["commaunit"] or m["commascale"]):
                noun = tags.head_after(end)
                if noun is not None and noun.start > end:  # "s 1,2,3 přáteli": the values agree with the noun they count
                    form = self._context(int(parts[-1]), start, end, text, tags, after_label, m)
                    words = ", ".join(self._cardinal(int(part), *form) for part in parts)
                    # the noun's form is the last value's: "Viděl 1,2,5 mužů" may want "jednoho" as an object
                    if form[0] == NOM and words != ", ".join(self._cardinal(int(part), ACC, *form[1:]) for part in parts):
                        self._warn(text, m, words, "list read in the nominative; as an object it takes the accusative")
                else:
                    words = ", ".join(self._cardinal(int(part), *self._label(int(part))) for part in parts)
                self._check_glued(text, m, words)
            else:  # "1,2,3 kg" -> "jeden, dva, tři kilogramy": the unit follows the last value, all agree with it
                gender = (SCALE_GENDERS[self.language][m["commascale"].lower()] if m["commascale"]
                          else NOUNS[self.language][self._unit(m["commaunit"])[0]][0])  # "1,2,3 tis. Kč": tisíc's
                case = self._governing_case(start, tags) or self._unit_verb_case(
                    parts[-1], m["commascale"] or m["commaunit"], start, end, text, tags, log=False) or NOM
                if (m["commascale"] or "").lower() == "tis" and self.language == "sk":  # "dvetisíc": one word each
                    head = [self._measure(part, m["commascale"], start, tags, text, end, comma=False) for part in parts[:-1]]
                else:
                    head = [self._cardinal(int(part), case, gender, "inanimate") for part in parts[:-1]]
                words = ", ".join(head + [self._measure(parts[-1], m["commascale"] or m["commaunit"], start, tags, text,
                                                        end, scale_unit=m["commascaleunit"])])
                if m["commaper"] or m["commascaleper"]:
                    words += " " + self._per(m["commaper"] or m["commascaleper"])
        elif kind == "time":
            words = self._time(m, tags)
        elif kind == "range" and m["lowscale"]:  # "5 tis. Kč–10 tis. Kč", "5 tis. Kč/kg–10 tis. Kč/kg"
            words = f" {RANGE_WORD} ".join(
                self._measure(m[end_], m[end_ + "scale"], start, tags, text, end, scale_unit=m[end_ + "scaleunit"],
                              comma=False, verb_log=end_ == "high")
                + (f" {self._per(m[end_ + 'scaleper'])}" if m[end_ + "scaleper"] else "")
                for end_ in ("low", "high"))
            self._check_comma(text, (start, end), words, m["low"], m["high"])
        elif kind == "range" and m["lowunit"]:  # "5 km–10 m", "5 Kč/kg–10 Kč/kg"
            words = f" {RANGE_WORD} ".join(
                self._measure(m[end_], m[end_ + "unit"], start, tags, text, end, comma=False, verb_log=end_ == "high")
                + (f" {self._per(m[end_ + 'per'])}" if m[end_ + "per"] else "")
                for end_ in ("low", "high"))
            self._check_comma(text, (start, end), words, m["low"], m["high"])
        elif kind == "range":
            words = self._range(m, m["low"], m["high"], m["rangeunit"], m["rangescale"], m["rangescaleunit"],
                                m["rangeper"] or m["rangescaleper"], text, tags, after_label)
        elif kind == "money":
            plus = m["price"].startswith("+")  # "$+5" -> "plus pět dolarů"
            price = _plain_price(m["price"][1:] if plus else m["price"])
            sign = "" if price[0] in "-−–" else (m["moneysign"] or "")  # "-$4.50", "$-4.50"
            scale = (m["moneyscale"] or m["highmoneyscale"] or "").lower() or None
            if m["lowmoneyscale"] and m["lowmoneyscale"].lower() != scale:  # "$500 tis.–$1 mil."
                words = f" {RANGE_WORD} ".join(
                    self._measure(amount, amount_scale, start, tags, text, end, scale_unit=m["symbol"], verb_log=high)
                    for amount, amount_scale, high in (
                            (sign + price, m["lowmoneyscale"], False),
                            ((m["highsign"] or "") + _plain_price(m["pricehigh"]), scale, True)))
            elif m["lowmoneyper"]:  # "$4/kg–$5/kg"
                words = f" {RANGE_WORD} ".join(
                    f"{self._measure(amount, m['symbol'], start, tags, text, end, verb_log=high)} {self._per(per)}"
                    for amount, per, high in (
                            (sign + price, m["lowmoneyper"], False),
                            ((m["highpersign"] or "") + _plain_price(m["highprice"]), m["highmoneyper"], True)))
            elif m["pricehigh"]:  # "$5–10" -> "pět až deset dolarů"
                words = self._range(m, sign + price, (m["highsign"] or "") + _plain_price(m["pricehigh"]),
                                    None if scale else m["symbol"],
                                    scale, m["symbol"] if scale else None, m["moneyper"], text, tags, after_label)
            else:  # "$5 mil." -> "pět milionů dolarů"
                words = self._measure(sign + price, scale or m["symbol"], start, tags, text, end,
                                      scale_unit=m["symbol"] if scale else None)
                if m["moneyper"]:
                    words += " " + self._per(m["moneyper"])  # "$4/kg" -> "čtyři dolary za kilogram"
            if plus:
                words = f"{SIGNS[self.language]['+']} {words}"
        elif kind == "measure":
            words = self._measure(m["amount"], m["unit"] or m["scale"], start, tags, text, end,
                                  whole=bool(m["whole"]), scale_unit=m["scaleunit"])
            if m["per"] or m["scaleper"]:  # "100 Kč/kg" -> "sto korun za kilogram"
                words += " " + self._per(m["per"] or m["scaleper"])
        elif kind == "ordinal":
            words = self._ordinal_digits(m, text, tags, after_label, heading)
        else:
            words = self._number(m, text, tags, after_label)
        if (kind in ("range", "measure", "time", "money", "commas") and m.group(0).endswith(".")
                and self._ends_sentence(text, start, end, tags)):
            words += "."  # the period of "min.", "mil." or "hod." also ends the sentence
        return self._vocalise(text, start, words)

    def _abbreviation(self, m: re.Match, text: str, tags: _Tags) -> str:
        raw, key = m.group(0), _key_of(m)
        if raw.isupper() and len(key) > 2 and key in CAPITAL_ACRONYMS:
            after = text[m.end():].lstrip(" \t  ")
            before = text[:m.start()].rstrip(" \t  ")
            # MAX. and ODS. introduce an amount ("MAX. ≈ 5", "ODS. 2"), N. L. follows a year ("300 N. L."); the
            # others keep their capital meaning also before a number: "TURNAJ ATP. 500"
            if not ((key in ("max.", "ods.") and re.match(r"(?:[-−–+±~≈$€£][ \t\u00a0\u202f]*)*\d", after))
                    or (key == "n.l." and before[-1:].isdigit())):
                if _all_capitals(text):
                    self._warn(text, m, raw, "acronym or abbreviation in capitals, kept as written; check it")
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
        if key.endswith(".") and self._ends_sentence(text, m.start(), m.end(), tags,
                                                     introduces=key in NON_FINAL_ABBREVIATIONS,
                                                     list_end=key in LIST_END_ABBREVIATIONS):
            words += "."
        return words

    def _roman(self, m: re.Match, text: str, tags: _Tags, heading: bool) -> str:
        numeral, end = m["numeral"], m.end()
        nxt, prev = tags.after(end), tags.before(m.start())
        head = None
        if prev is None or prev.upos != "PROPN":
            # "XXI. století", "XIX.–XX. století", "# V. Kapitola", but "Karel IV. univerzitu" agrees with Karel
            before_noun = nxt is not None and (nxt.text[:1].islower() or heading or (
                nxt.upos in ("NOUN", "ADJ") and (len(numeral) > 1 or numeral in "IVX")
                and not self._verb_follows(end, tags)))  # not "D. Kapitola byla…", "Příloha C. Varianta D."
            head = (tags.head_after(end) if before_noun else None) or self._shared_head(end, tags)
        if head is None:
            # "Vyšel díl V. Kniha byla…", "…V. Nové vydání…": after a thing, not a person, and before a sentence
            # whose subject is a noun, V is a numeral
            lead = next((w for w in tags.words[bisect.bisect_left(tags.starts, end):] if w.upos not in ("ADV", "PART")),
                        None)  # past an adverb: "Vyšel díl V. Poté kniha uspěla."
            subject = tags.head_after(lead.start) if lead is not None and lead.upos in ("NOUN", "ADJ", "DET") else lead
            # a thing by its tag or as a known label: CAC gives feminine nouns no animacy, so "Paní V." has none
            numbers_a_thing = (prev is not None and prev.upos == "NOUN"
                               and (prev.feats.get("Animacy") == "Inan" or prev.text.lower() in ROMAN_LABEL_NOUNS[self.language])
                               and subject is not None and subject.upos in ("NOUN", "PRON")
                               and self._verb_follows(lead.start, tags))
            if (prev is None or prev.upos not in ("NOUN", "PROPN") or _roman_value(numeral) >= 400
                    or (len(numeral) == 1 and (numeral not in "IVX"
                                               or (nxt and nxt.text[:1].isupper() and not numbers_a_thing)))):
                return m.group(0)  # "V. Havel", "Washington DC.", "Příloha C.", "Velikost L.", or no noun to agree with
            head = prev
        case, gender, animacy, plural, doubt = self._agreement(head, m.start(), tags,
                                                               following=head.start > m.start())
        words = self._ordinal(_roman_value(numeral), case or NOM, gender or "masculine", animacy or "inanimate",
                              plural)
        if case is None or gender is None:
            self._warn(text, m, words, "no noun to agree with; nominative masculine inanimate")
        elif doubt:
            self._warn(text, m, words, f"{doubt}; check it")
        elif (prev is not None and prev.text.lower() in ROMAN_LABEL_NOUNS[self.language] and head.start > m.start()
              and case == GEN):
            # "díl V. knihy": "díl páté knihy" (of the fifth book), or "díl pátý knihy" (volume five of the book)
            self._warn(text, m, words, f"numeral after {prev.text!r} may number it instead; check it")
        if head.start < m.start() and self._ends_sentence(text, m.start(), end, tags, roman=True):
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
        prep = self._preposition(m.start(), tags) or self._shared_preposition(m.start(), tags)
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
        words = [self._cardinal(hour, case, "feminine", "inanimate") if hour or case not in (NOM, ACC) else "nula"]
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
            high = self._measure(high_amount, scale or unit, start, tags, text, end, scale_unit=scale_unit, comma=False)
            if unit and decimal and self._unit(unit)[0] in MINOR_UNITS[self.language]:  # "1,50–2,50 €"
                low_words = self._measure(low_amount, unit, start, tags, text, end, comma=False, verb_log=False)
                words = f"{low_words} {RANGE_WORD} {high}"
            elif scale and scale.lower() == "tis" and self.language == "sk":  # "dvetisíc až tritisíc": one word each
                words = f"{self._measure(low_amount, scale, start, tags, text, end, comma=False)} {RANGE_WORD} {high}"
            else:
                gender = (SCALE_GENDERS[self.language][scale.lower()] if scale
                          else NOUNS[self.language][self._unit(unit)[0]][0])
                case = NOM if decimal else (self._governing_case(start, tags) or self._unit_verb_case(
                    high_amount, scale or unit, start, end, text, tags, log=False) or NOM)
                words = f"{self._signed(low_amount, self._cardinal(low, case, gender, 'inanimate'))} {RANGE_WORD} {high}"
        else:
            if decimal:
                case, gender, animacy = NOM, "masculine", "inanimate"
            else:
                case, gender, animacy = self._context(value, start, end, text, tags, after_label, m)
            words = (f"{self._signed(low_amount, self._cardinal(low, case, gender, animacy))} {RANGE_WORD} "
                     f"{self._signed(high_amount, self._cardinal(value, case, gender, animacy))}")
        self._check_comma(text, (start, end), words, low_amount, high_amount)  # one check for both ends
        self._check_glued(text, m, words)
        return words + (f" {self._per(per)}" if per else "")

    def _measure(self, amount: str, unit: str, start: int, tags: _Tags, text: str, end: int,
                 whole: bool = False, scale_unit: Optional[str] = None, comma: bool = True,
                 verb_log: bool = True) -> str:
        """comma: log a "2,000" read as a decimal (a range checks both ends itself); verb_log: log a genitive
        verb's doubt (a range logs it at its upper end only)."""
        value, integer, fraction = _parse(amount)
        case = self._governing_case(start, tags) or self._unit_verb_case(amount, unit, start, end, text, tags,
                                                                         log=verb_log)
        unit = unit.rstrip(".")
        if unit.lower() in SCALES:  # also "5 TIS. Kč"
            unit = unit.lower()
            if fraction:
                words = f"{self._cardinal(value, NOM, 'masculine', 'inanimate')} {SCALE_GENITIVES[self.language][unit]}"
            else:
                words = self._cardinal(value * 10 ** SCALES[unit], case or NOM, "masculine", "inanimate")
            if scale_unit:  # "5 tis. km" -> "pět tisíc kilometrů": the noun follows tisíc, milion
                noun, adjective, suffix = self._unit(scale_unit)
                count = 1000 if fraction else integer * 10 ** SCALES[unit]
                words += f" {self._noun_phrase(noun, count, NOM if fraction else case or NOM, adjective)}{suffix}"
            if comma:
                self._check_comma(text, (start, end), words, amount)
            return self._signed(amount, words)
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
        if comma:
            self._check_comma(text, (start, end), words, amount)
        return self._signed(amount, words) + suffix

    def _check_glued(self, text: str, m: re.Match, words: str) -> bool:
        """Log a word glued after the last digit: "5G", "1,2,3G", "5–10G"; not the "x" of "3x4", read as a sign."""
        if text[m.end():m.end() + 1].isalpha() and not re.match(f"x{_NOT_LETTER_AFTER}", text[m.end():]):
            self._warn(text, m, words, "digits glued to a word")
            return True
        return False

    def _check_comma(self, text: str, where, words: str, *amounts: str) -> None:
        """"2,000" is read with a decimal comma, as Czech and Slovak write it; English means two thousand."""
        if any(re.fullmatch(r"[-−–+]?\d{1,3},\d{3}", amount) for amount in amounts):
            self._warn(text, where, words, "comma read as decimal, not thousands; check it")

    def _signed(self, amount: str, words: str) -> str:
        if amount.startswith("+"):
            return f"{SIGNS[self.language]['+']} {words}"  # "$5–$+10"
        return f"{self._numbers.MINUS} {words}" if _negative_zero(amount) else words

    def _per(self, per: str) -> str:
        """The unit after "/" in the accusative singular: "za kilogram", "za metr čtvereční"."""
        power = {"2": "²", "3": "³"}.get(per[-1], per[-1]) if per[-1] in "²³23" else None
        adjective = UNIT_ADJECTIVES[self.language].get(power)
        key = _canonical(per[:-1] if power else per, PER_UNITS[self.language])
        return f"za {PER_UNITS[self.language][key]}" + (f" {adjective}" if adjective else "")

    def _unit(self, unit: str) -> Tuple[str, Optional[str], str]:
        """(noun, agreeing adjective, suffix) of a unit symbol: "m²" -> metr, čtvereční."""
        unit = unit.rstrip(".")
        power = {"2": "²", "3": "³"}.get(unit[-1], unit[-1]) if unit[-1] in "²³23" else None
        base = _canonical(unit[:-1] if power else unit, UNITS[self.language])
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
        # "Beethovenova 5. Symfonie"; in capitals also after a conjunction ("A 2. DÍLY VYŠLY"), but not one between
        # cardinals ("V LETECH 1914 A 1918. VÁLKA SKONČILA")
        attributive = prev is None or prev.upos in ("ADJ", "DET", "ADP", "PUNCT") or (
            prev.upos == "CCONJ" and tags.capitals and not getattr(tags.before(prev.start), "text", "").isdigit())
        # after a preposition, a capitalised noun in its case is the ordinal's noun even when tagged a name
        # ("V 5. Symfonii"); a nominative one before a verb starts a sentence ("Přišel v 5. Symfonie začala.")
        after_preposition = prev is not None and prev.upos == "ADP" and nxt is not None
        governed = (after_preposition and nxt.upos in ("NOUN", "PROPN") and nxt.feats.get("Animacy") != "Anim"
                    and self._agrees_with_preposition(prev, nxt))
        subject = after_preposition and nxt.feats.get("Case") == "Nom" and self._verb_follows(m.end(), tags)
        # in capitals a chain is no sentence end ("1. AŽ 5. LEDNA", "1. A 2. DÍL"); in mixed text a chain's "a" is
        # lowercase, so a capital "A" starts a sentence: "Bylo jich 1. A 2. díl vyšel."
        if following[:1].isupper() and not heading and not governed and (
                not _all_capitals(text) or self._shared_head(m.end(), tags) is None) and (
                nxt is None or nxt.upos not in ("NOUN", "ADJ") or subject
                or (not attributive and self._verb_follows(m.end(), tags))):
            # "Bylo jich 5. Pak…", "Měl jen 2. Děti odešly.": a number that ends the sentence; but "5. Symfonie",
            # "# 2. Kapitola"
            verb = self._rank_verb(m.start(), tags)
            if verb is not None:
                return self._ordinal(value, NOM, *self._gender(verb)) + "."  # "Skončil 2. Pak…"
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
        elif (len(m["value"]) == 2 and m["value"].isdigit() and _DATE_BEFORE.search(text[:m.start()])
              and tags.head_after(m.end()) is None):
            words = self._cardinal(value, NOM, "masculine", "inanimate")  # "5. 6. 24 v Praze": the year of the date
        elif text[m.end():m.end() + 1] == "." and self._rank_verb(m.start(), tags) is not None:
            words = self._ordinal(value, NOM, *self._gender(self._rank_verb(m.start(), tags)))  # "Skončil 2."
        else:
            words = self._cardinal(value, *self._context(value, m.start(), m.end(), text, tags,
                                                         after_label, m))
        words = self._signed(m["value"], words)
        self._check_comma(text, m, words, m["value"])
        if m.start() and text[m.start() - 1].isalpha():
            words = " " + words
        if self._check_glued(text, m, words):
            words += " "
        return words

    # ---- context ---------------------------------------------------------------------------
    def _context(self, value: int, start: int, end: int, text: str, tags: _Tags, after_label: bool,
                 m) -> Tuple[str, str, str]:
        """Case, gender and animacy of a cardinal, from the noun it counts or the preposition before it."""
        before, after = text[:start].rstrip(), text[end:].lstrip()
        if (after_label or before[-1:] in _MATH_SIGNS or after[:1] in _MATH_SIGNS
                or re.search(rf"\d{_HS}*[~≈]$", before) or re.match(f"[~≈]{_OPERAND}", after)):
            return self._label(value)  # "č. 5", "#1", "tři krát čtyři", "2023/2024", "1≈2"; not "s ≈5 lidmi"
        prep_case = self._preposition_case(start, tags)
        own = noun = tags.head_after(end)
        if noun is None:
            od = getattr(self._preposition(start, tags), "text", "").lower() in ("od", "ode")  # also "od asi 1 do 2 h"
            link = (rf"(?:{_HS}*,{_HS}*|{_HS}+(?i:a|nebo|alebo{'|do' if od else ''}){_HS}+)"
                    rf"(?:(?i:{_APPROXIMATOR}){_HS}*)?(?P<later>[-−–+]?{_UNSIGNED})")  # "1 nebo asi 2 h"
            unit = _UNIT + (_CAPITAL_UNIT if tags.capitals else "")
            later_unit = re.match(rf"(?:{link})+{_HS}*(?P<unit>{unit}|{_CURRENCY}){_NOT_LETTER_AFTER}", text[end:])
            if later_unit:  # "1 nebo 2 h", "od 1 do 2 °C": the unit after a later number counts this one too
                unit_gender = NOUNS[self.language][self._unit(later_unit["unit"])[0]][0]
                unit_case = self._governing_case(start, tags) or self._unit_verb_case(
                    later_unit["later"], later_unit["unit"], end + later_unit.start("later"),
                    end + later_unit.end("unit"), text, tags, log=False)
                words = self._cardinal(value, NOM, unit_gender, "inanimate")
                if unit_case is None and (words != self._cardinal(value, ACC, unit_gender, "inanimate")
                                          or (self.language == "cs" and self._verb_case(start, None, tags) == GEN)):
                    self._warn(text, m, words, "no preposition; nominative")
                return unit_case or NOM, unit_gender, "inanimate"
            noun = self._shared_count_head(end, tags)
        case = gender = animacy = tagged_case = noun_case = None
        if noun is not None:
            gender, animacy = self._gender(noun)
            tagged_case = _UD_CASES.get(noun.feats.get("Case"))
            if noun.feats.get("Number") == "Plur" and noun.text.lower().endswith(LOCATIVE_PLURAL_ENDINGS):
                tagged_case = LOC  # "po tisíci letech"
            elif (tagged_case in PLURAL_ENDINGS and noun.feats.get("Number") == "Plur"
                    and not noun.text.lower().endswith(PLURAL_ENDINGS[tagged_case])):
                tagged_case = GEN  # "o 5 minút": a genitive plural tagged with the preposition's case
            approximator = tags.before(start)
            if (approximator is not None and approximator.text.lower() in APPROXIMATORS and tagged_case != NOM
                    and noun.feats.get("Number") == "Plur"):
                # "s 5 lidmi a ~3 psy": past "~" CAC loses the case that "a" shares with the earlier noun; but after
                # a clause's verb, a verb after the amount opens a new clause ("a skoro 2 hodiny tam zůstal")
                link = tags.before(approximator.start)
                earlier = tags.before(link.start) if link is not None and link.upos == "CCONJ" else None
                if (earlier is not None and earlier.upos == "NOUN"
                        and not (self._verb_follows(end, tags) and self._verb_before(earlier.start, tags, True))):
                    shared_case = _UD_CASES.get(earlier.feats.get("Case"))
                    if shared_case in PLURAL_ENDINGS and noun.text.lower().endswith(PLURAL_ENDINGS[shared_case]):
                        tagged_case = shared_case
            if (self.language == "cs" and noun.feats.get("Gender") == "Neut" and noun.text.lower().endswith("í")
                    and self._count_form(value) == "gen_pl"):
                tagged_case = noun_case = GEN  # "pět vítězství": one form for most cases, after 5 a genitive plural
            elif self._number_fits(value, noun):
                noun_case = tagged_case
        prep = self._preposition(start, tags)
        if (self.language == "cs" and prep is not None and prep.text.lower() in TWO_CASE_PREPOSITIONS
                and noun is not None and noun.feats.get("Gender") == "Masc" and noun.feats.get("Number") == "Plur"
                and noun.text.lower().endswith("y")):
            decided = self._two_case(prep, noun, start, tags)
            verb = self._clause_verb(start, tags)
            if (decided == INS and prep.text.lower() == "za" and verb is not None
                    and _affirmative(verb.text.lower()).startswith(GOING_VERBS)):  # "Šel za 2 stromy": after them, or behind them
                self._warn(text, m, self._cardinal(value, INS, gender, animacy),
                           "za after going read as following (instrumental); a place behind is the accusative; check it")
            if decided:
                return decided, gender, animacy
        if noun_case in (DAT, INS, LOC):
            case = noun_case  # "s pěti přáteli"
        elif prep_case == GEN:
            case = GEN  # "bez pěti jablek", "do dvou hodin"
        elif prep_case and self._ends_in_scale_noun(value):
            case = prep_case  # "s tisícem lidí": after tisíc the noun is genitive in every case
        elif (prep is not None and prep.text.lower() in ("v", "ve") and self.language == "cs" and 0 <= value <= 24
              and (noun is None or noun.text.lower() in DAY_TIMES) and not re.match(rf"{_HS}+ze?{_HS}", text[end:])):
            # after "v" a bare number is a clock time ("Přišel v pět", "ve dvě večer"); an age would be "v pěti"
            if not re.match(rf"{_HS}+(?:{'|'.join(DAY_TIMES)}|v noci){_NOT_LETTER_AFTER}", text[end:]):
                self._warn(text, m, self._cardinal(value, ACC, "feminine", "inanimate"),
                           "time (v pět) or age (v pěti), read as a time; check it")
            return ACC, "feminine", "inanimate"
        elif prep_case:
            # "ve dvě hodiny", "v pět hodin": the tagger often gives v/na/o the locative here
            case = ACC if prep_case == ACC or tagged_case in (NOM, ACC, GEN) else prep_case
        elif noun is not None:
            # no governor: "Mám pět jablek" (the noun is genitive, the number nominative);
            # Slovak "videl troch mužov" (A = G of masculine personal nouns), "dvoch psov"
            if (self.language == "sk" and noun_case == GEN and noun.text.lower().endswith("ov")
                    and self._count_form(value) != "gen_pl"):
                case = GEN
            elif (noun_case == GEN and 0 < abs(value) < 5 and "," not in text[start:end]
                  and (noun is own or self._governed(noun, start, prep, tags))
                  and not (abs(value) > 1 and noun.text.lower().endswith(("a", "á", "e", "ě", "é", "i", "o", "y")))):
                # 1–4 agree with their noun ("dva body"), so a genitive one is governed: "Dosáhl dvou bodů"; a genitive
                # plural never ends like -a, -e, -i, -o, -y, so "skoro 2 hodiny" tagged genitive is not one
                case = GEN
            elif (noun_case == GEN and (abs(value) >= 5 or value == 0) and self.language == "cs"
                  and self._verb_case(start, prep, tags) == GEN):
                # from 5, and at 0, the noun is genitive anyway, so the verb tells: a numeral subject takes a neuter
                # singular verb ("Zúčastnilo se padesát lidí"), any other is the verb's object ("Dosáhl pěti bodů")
                case = self._genitive_verb_case(value, start, noun, text, m, tags)
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
        if (i + 2 < len(w) and (w[i].upos in ("CCONJ", "ADP", "NOUN") or w[i].text.lower() in ("–", "—", "-", ",", "až"))
                and (w[i + 1].text.isdigit() or re.fullmatch(_ROMAN, w[i + 1].text)) and w[i + 2].text == "."):
            return tags.head_after(w[i + 2].end) or self._shared_head(w[i + 2].end, tags)  # "2., 3. a 4. díl"
        return None

    def _governed(self, noun: _Word, pos: int, prep: Optional[_Word], tags: _Tags) -> bool:
        """Whether the genitive of a later number's noun, which the number at `pos` shares, is governed: after 2–4
        it is ("Dosáhl 2 a 3 bodů"), after 5 only a genitive verb tells ("Bál se 2 nebo 5 psů", not "2 nebo 5 piv")."""
        later = next((w for w in reversed(tags.words[:bisect.bisect_left(tags.starts, noun.start)])
                      if w.text.replace(" ", "").isdigit()), None)
        count = int(later.text.replace(" ", "")) if later is not None else 0
        return noun.feats.get("Number") == "Plur" and (
            1 < count < 5 or (self.language == "cs" and self._verb_case(pos, prep, tags) == GEN))

    def _shared_count_head(self, pos: int, tags: _Tags) -> Optional[_Word]:
        """The noun of a later number that this one shares: "2 nebo 3 knihy", sk "2 alebo 3 muži", "od 1 do 2 hodin"."""
        i, w = bisect.bisect_left(tags.starts, pos), tags.words
        od = i > 0 and getattr(tags.before(w[i - 1].start, skip=("ADV", "PART")), "text", "").lower() in ("od", "ode")
        j = i + 2 if i + 1 < len(w) and w[i + 1].text.lower() in APPROXIMATORS else i + 1  # "1 nebo asi 2 knihy"
        if (j < len(w) and (w[i].upos == "CCONJ" or w[i].text in (",", "–", "—", "-")
                            or (od and w[i].text.lower() == "do"))
                and w[j].text.replace(" ", "").isdigit()):
            return tags.head_after(w[j].end) or self._shared_count_head(w[j].end, tags)
        return None

    @staticmethod
    def _chain_start(pos: int, tags: _Tags) -> int:
        """Where "1.–5." or "XIX. a XX." starts, for an ordinal at `pos`: the whole chain has one governor."""
        i, w = bisect.bisect_left(tags.starts, pos) - 1, tags.words
        while (i >= 2 and (w[i].upos == "CCONJ" or w[i].text.lower() in ("–", "—", "-", ",", "až")) and w[i - 1].text == "."
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
            verb_case = self._verb_case(start, prev, tags) if self.language == "cs" else None
            if verb_case == ACC:
                prep_case = ACC  # "na" with "vzpomínat": "na dvacáté století"
            if self.language == "cs" and head.feats.get("Gender") == "Neut" and head.text.lower().endswith("í"):
                # "století" has one form for every case but the instrumental, so the context decides
                context = prep_case or verb_case or (GEN if prev is not None and prev.upos in ("NOUN", "PROPN") else NOM)
                if verb_case == GEN and self._subject(head, start, tags):
                    context = NOM  # "Cíle dosáhlo 2. sdružení", not "Dožil se 2. tisíciletí", "Dosáhlo to 2. výročí"
                if ordinal and context == NOM and case not in (None, NOM, ACC) and not self._clause_start(start, tags):
                    doubt = f"{head.text!r} tagged {case}, read {context}"  # "Dosáhli jsme XXI. století"
                elif ordinal and verb_case == ACC and _affirmative(self._clause_verb(start, tags).text.lower()).startswith(NA_PLACE_VERBS):
                    doubt = f"na {head.text!r} read as the accusative (waiting for); the locative (waiting at) fits too"
                case = context
            # a singular nominative or accusative is the subject ("Cíle dosáhl 2. muž"), and so is an agreeing
            # plural nominative ("Cíle dosáhli 2. muži")
            elif verb_case == GEN and (plural or case not in (NOM, ACC)) and not self._subject(head, start, tags):
                case = GEN  # "Dosáhli 5. místa", "Bál se 2. dílu": a genitive singular that looks plural or dative
                plural = plural and not head.text.lower().endswith(("a", "y", "e", "ě", "u"))
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
        if word.feats.get("Number") == "Plur" and word.text.lower() in NEUTER_PLURALS[self.language]:
            return "neuter", "inanimate"
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

    def _rank_verb(self, pos: int, tags: _Tags) -> Optional[_Word]:
        """The placement verb before a number that ends a sentence, which makes the number a rank:
        "Skončil 2." -> "druhý"; a count keeps its cardinal: "Koupil 5.", "Bylo jich 5."."""
        verb = tags.before(pos, skip=("PRON",))  # "umístil se 3."
        if (verb is None or verb.upos not in ("VERB", "AUX") or "Sing" not in verb.feats.get("Number", "Sing")
                or self._gender(verb)[0] not in ("masculine", "feminine")):
            return None
        return verb if _affirmative(verb.text.lower()).startswith(RANK_VERBS[self.language]) else None

    @staticmethod
    def _clause_verb(pos: int, tags: _Tags) -> Optional[_Word]:
        """The verb nearest to `pos` in its clause, which a comma or a sentence end closes, and before `pos` also
        a conjunction ("Cíle dosáhl a 2. místo obsadil"); an auxiliary ("jsme", "by") is skipped, a copula
        ("je", "byl") counts. Only the nearest verb governs: in "Dosáhl cíle a obsadil 2. místo" it is "obsadil".
        A clause with no verb of its own shares the one before its conjunction: "Dosáhl cíle a 2. místa"."""
        i, w, found = bisect.bisect_left(tags.starts, pos), tags.words, []

        def closes(j: int) -> bool:  # an ordinal's period is no sentence end: "a 2. místo obsadil", "A 2. DÍLY VYŠLY"
            return w[j].text in (",", ".", "!", "?", "…", ";", ":") and not (
                w[j].text == "." and 0 < j < len(w) - 1 and w[j - 1].end == w[j].start
                and (w[j - 1].text.isdigit() or re.fullmatch(_ROMAN, w[j - 1].text))
                and (w[j + 1].text[:1].islower() or (tags.capitals and j >= 2 and w[j - 2].upos == "CCONJ"
                                                    and not (j >= 3 and w[j - 3].text.isdigit()))))

        for step, across in ((-1, False), (1, False), (-1, True)):
            if across and found:
                break
            j = i - 1 if step < 0 else i
            while 0 <= j < len(w) and not closes(j) and not (step < 0 and not across and w[j].upos == "CCONJ"):
                if w[j].upos == "VERB" or (w[j].upos == "AUX" and _affirmative(w[j].text.lower()) in PLACE_FORMS):
                    found.append((abs(j - i), w[j]))
                    break
                j += step
        return min(found, key=lambda f: f[0])[1] if found else None

    @staticmethod
    def _clitic_of(verb: _Word, tags: _Tags, clitic: str) -> bool:
        """Whether `clitic` ("se", "si") belongs to `verb`: it stands in the verb's own clause, which a comma,
        a sentence end, a conjunction or another verb closes ("Smál se a vzdal 2. kolo": "se" is smál's)."""
        i, w = bisect.bisect_left(tags.starts, verb.start), tags.words
        for step in (-1, 1):
            j = i + step
            while (0 <= j < len(w) and w[j].text not in (",", ".", "!", "?", "…", ";", ":")
                   and w[j].upos not in ("CCONJ", "VERB")):
                if w[j].text.lower() == clitic:
                    return True
                j += step
        return False

    def _two_case(self, prep: _Word, noun: Optional[_Word], pos: int, tags: _Tags) -> Optional[str]:
        """The case after mezi, nad, pod, před or za when the noun has one form for both: the accusative
        after a verb of direction, the instrumental after one of place; mezi and před are mostly places."""
        word = prep.text.lower()
        if noun is not None and noun.text.lower() in TIME_PLURALS:
            return {"před": INS, "za": ACC}.get(word)
        verb = self._clause_verb(pos, tags)
        form = _affirmative(verb.text.lower()) if verb is not None else ""
        clause = self._clause_before(pos, tags) + self._clause_after(pos, tags)  # "Mezi 2 svazky byl dopis položen"
        if (any(w.upos in ("ADJ", "VERB") and w.feats.get("VerbForm") == "Part"
                and _affirmative(w.text.lower()).startswith(PLACED_PARTICIPLES) for w in clause)
                and not any(w.text.lower() in ("je", "jsou", "není", "nejsou") for w in clause)):
            return ACC  # the event, "byl položen mezi dva svazky"; the state is a place: "je pověšen nad dvěma stoly"
        if verb is not None and verb.feats.get("Voice") == "Pass":
            return INS  # "Dům je postaven mezi dvěma stromy": a passive is the state, not the motion
        if form and form.startswith(DIRECTION_VERBS) and form not in PLACE_FORMS:
            return INS if word == "za" and form.startswith(GOING_VERBS) else ACC
        if form and (form in PLACE_FORMS or form.startswith(PLACE_VERBS)):
            return INS
        return INS if word in ("mezi", "před") and noun is not None else None

    def _verb_case(self, pos: int, prev: Optional[_Word], tags: _Tags) -> Optional[str]:
        """The case a Czech verb gives a noun phrase at `pos`: the genitive after "dosáhnout", "bát se";
        the accusative after "na" with "vzpomínat", "čekat"."""
        verb = self._clause_verb(pos, tags)
        form = _affirmative(verb.text.lower()) if verb is not None else ""
        if prev is not None and prev.upos == "ADP":
            return ACC if prev.text.lower() == "na" and form and form.startswith(NA_ACCUSATIVE_VERBS) else None
        clitic = next((c for stem, c in GENITIVE_VERBS.items() if re.match(stem, form)), False) if form else False
        return GEN if clitic is None or (clitic and self._clitic_of(verb, tags, clitic)) else None

    def _agrees_with_preposition(self, prep: _Word, noun: _Word) -> bool:
        case = _UD_CASES.get(noun.feats.get("Case"))
        if prep.text.lower() in LOCATIVE_PREPOSITIONS[self.language]:
            return case in (ACC, LOC)  # the tagger gives v, na, o, po either case
        return case is not None and case == _UD_CASES.get(prep.feats.get("Case"))

    def _genitive_verb_case(self, value: int, start: int, noun: _Word, text: str, where, tags: _Tags) -> str:
        """Subject (NOM) or object (GEN) of a Czech genitive verb for a number counting `noun`; where=None logs nothing."""
        gender, animacy = self._gender(noun)

        def warn(reading: str, reason: str) -> None:
            if where is not None:
                self._warn(text, where, reading, reason)

        verb = self._clause_verb(start, tags)
        feats = verb.feats
        if (value == 0 and "Fem" in (feats.get("Gender") or "") and "Sing" in (feats.get("Number") or "")
                and self._no_subject_in_sight(verb, tags)):
            # an "-la" verb also agrees with "nula": "Zúčastnila se nula lidí"; like a present verb
            taking_part = (_affirmative(verb.text.lower()).startswith(("zúčastn", "účastn"))
                           and noun.feats.get("VerbForm") != "Vnoun")
            case = NOM if start < verb.start or taking_part else GEN
            warn(self._cardinal(0, case, "feminine", "inanimate"), "subject or object of a verb that may agree with "
                 "zero, read as the " + ("subject" if case == NOM else "object") + "; check it")
            return case
        conjunction = tags.before(start)
        if conjunction is not None and conjunction.text.lower() in APPROXIMATORS:  # "Petr a asi 5 mužů"
            conjunction = tags.before(conjunction.start)
        first = tags.before(conjunction.start) if conjunction is not None and conjunction.upos == "CCONJ" else None
        nominative = (first is not None and first.feats.get("Case") == "Nom" and first.text.isalpha()
                      and first.text.lower() not in UNIT_WORDS[self.language])
        # a subject in the verb's clause, also after the number ("Dosáhne pěti bodů právě Petr") or before an
        # "a" joining it to another predicate; a number before the verb opens a clause: "Pěti chyb si nikdo"
        subject = shared = fronted = False
        if verb.start < start:
            words = self._clause_before(verb.start, tags)
            stop = tags.before(words[-1].start if words else verb.start)
            # an object put first: a genitive ("Úspěchu"), or to CAC a nominative ("Cíle") that a Czech past verb
            # shows is no subject (a Slovak "-li" shows no gender)
            fronted = (stop is None or stop.upos != "CCONJ") and any(w.upos == "NOUN" and (
                (w.feats.get("Case") == "Nom" and self.language == "cs" and feats.get("Gender") is not None
                 and not _agree(verb, w, True) and w.text.lower() not in TIME_NOUNS | NEUTER_PLURALS[self.language])
                or (w.feats.get("Case") == "Gen" and w.text.lower() not in TIME_GENITIVES
                    and w.text.lower() not in MONTHS_GENITIVE[self.language]
                    and getattr(tags.before(w.start, skip=("ADJ", "DET", "ADV", "PART")), "upos", None)
                    not in ("ADP", "NOUN", "PROPN", "NUM"))) for w in words)
            subject = any(_plain_nominative(w) and _agree(verb, w) for w in words + [
                w for w in self._clause_after(verb.end, tags) if w.start < start and w is not first]
                + self._clause_after(noun.end, tags))
            if not subject and stop is not None and stop.upos == "CCONJ" and all(
                    w.upos in ("AUX", "ADV", "PART") or w.text.lower() in ("se", "si") for w in words):
                subject = shared = any(_plain_nominative(w) and _agree(verb, w)
                                       for w in self._clause_before(stop.start, tags))
        else:
            subject = any(_plain_nominative(w) and _agree(verb, w)
                          for w in [w for w in self._clause_before(verb.start, tags) if w.start >= noun.end]
                          + self._clause_after(verb.end, tags))
        # "Báli jsme se 5 psů": an auxiliary in the first or second person is the subject; so is "to" ("Dosáhlo to
        # pěti bodů"), but not of taking part ("Zúčastnilo se to padesát lidí") or after a question word ("Čeho se to")
        # after "procent" the noun it counts follows: "Pět procent studentů dosáhli cíle", "pěti procent hlasování"
        counted = tags.head_after(noun.end) if noun.text.lower() in ("procent", "promile", "milionů", "miliard") else None
        heads = [noun] + ([counted] if counted is not None and counted.feats.get("Case") == "Gen" else [])
        taking_part = (_affirmative(verb.text.lower()).startswith(("zúčastn", "účastn"))
                       and all(h.feats.get("VerbForm") != "Vnoun"  # an event: "padesáti jednání"
                               and h.text.lower() not in EVENT_GENITIVES for h in heads))
        clause = self._clause_before(verb.start, tags) + self._clause_after(verb.end, tags)
        pronouns = [w.start for w in clause if w.upos == "PRON" and w.text.lower() not in ("se", "si")]
        subject = subject or any((w.upos == "AUX" and w.feats.get("Person") in ("1", "2"))
                                 or (w.text.lower() in ("to", "toto", "tohle") and w.feats.get("Case") == "Nom"
                                     and not taking_part and not any(p < w.start for p in pronouns)) for w in clause)
        # the verb agrees with the number in the plural, as people say: "Pět mužů dosáhli cíle"
        # "-la" (Fem,Neut and Plur,Sing) agrees as a neuter plural with a number before it when its object follows:
        # "Pět procent voličů se zúčastnila voleb", but "Pěti vítězství dosáhla (minulého roku)": a feminine subject
        after = tags.head_after(verb.end)
        la = feats.get("Number") == "Plur,Sing" and (fronted or (
            start < verb.start and after is not None and after.feats.get("Case") == "Gen"
            and after.text.lower() not in TIME_GENITIVES))
        colloquial = any((feats.get("Number") == "Plur" or (la and h.feats.get("Gender") == "Neut"))
                         and h.feats.get("Gender") in (feats.get("Gender") or "").split(",")
                         and (h.feats.get("Gender") != "Masc" or h.feats.get("Animacy") == feats.get("Animacy"))
                         for h in heads if h is noun or h.feats.get("Number") == "Plur")
        if subject:
            case = GEN
        elif fronted and (first is not None or colloquial or feats.get("Gender") is None):
            case = NOM  # "Cíle dosáhl vůdce a pět mužů", "Cíle dosáhli pět mužů"
        elif nominative and _plain_nominative(first):
            case = NOM  # "a" joins like cases: "Cíle dosáhla Eva a pět žen"; "vůdce" may be a genitive too
        elif feats.get("VerbForm") == "Inf" or feats.get("Person") in ("1", "2"):
            case = GEN  # "Chce dosáhnout pěti bodů": no numeral subject
        elif feats.get("Gender") is None:
            # a present verb shows no gender: with no subject the number is its subject before it ("Pět mužů
            # dosáhne cíle") and its object after it ("Dosáhne pěti bodů"), but "Zúčastní se padesát lidí"
            case = GEN if verb.start < start and not taking_part else NOM
        elif colloquial and start < verb.start:
            case = NOM  # before its verb the number is its subject also when the verb agrees in the plural
            warn(self._cardinal(value, NOM, gender or "masculine", animacy or "inanimate"),
                 "subject of a plural verb, as people say, or an object put first; check it")
        else:
            case = NOM if (feats.get("Gender"), feats.get("Number")) == ("Neut", "Sing") else GEN
        if (case == GEN and noun.text.lower() in DURATION_GENITIVES and self._after_own_noun(start, tags)
                and getattr(tags.after(noun.end), "text", "").lower() in ("před", "po")):
            return ACC  # after the verb's object a time before or after something: "Dosáhl cíle pět minut před ostatními"
        if case == GEN and noun.text.lower() in DURATION_GENITIVES:
            warn(self._cardinal(value, GEN, gender or "masculine", animacy or "inanimate"),
                 "time after a genitive verb, read as its object; a duration is the accusative; check it")
        elif (feats.get("Gender") is None and feats.get("VerbForm") != "Inf"
              and feats.get("Person") not in ("1", "2") and (case == NOM or not subject)):
            warn(self._cardinal(value, case, gender or "masculine", animacy or "inanimate"),
                 "subject or object of a present genitive verb, read as the "
                 + ("subject" if case == NOM else "object") + "; check it")
        elif case == GEN and nominative and not subject:  # a nominative in a form a genitive shares
            warn(self._cardinal(value, GEN, gender or "masculine", animacy or "inanimate"),
                 "conjunct of a nominative subject or the verb's object, read as the object; check it")
        elif case == GEN and shared and feats.get("Gender") is None:  # or "a" opens the number's clause
            warn(self._cardinal(value, GEN, gender or "masculine", animacy or "inanimate"),
                 "object of a subject shared across \"a\", or a new clause's subject; check it")
        elif case == GEN and colloquial and not subject:  # "Cíle dosáhli pět mužů" is how people say it too
            warn(self._cardinal(value, GEN, gender or "masculine", animacy or "inanimate"),
                 "object, or the subject of a plural verb, as people say; check it")
        return case

    def _no_subject_in_sight(self, verb: _Word, tags: _Tags) -> bool:
        """No plain nominative before `verb` and no first- or second-person auxiliary in its clause."""
        before = self._clause_before(verb.start, tags)
        return not any(_plain_nominative(w) for w in before) and not any(
            w.upos == "AUX" and w.feats.get("Person") in ("1", "2") for w in before + self._clause_after(verb.end, tags))

    def _unit_verb_case(self, amount: str, unit: str, start: int, end: int, text: str, tags: _Tags,
                        log: bool = True) -> Optional[str]:
        """After a Czech genitive verb an amount with a unit is its subject or object as a counted noun is."""
        value, integer, fraction = _parse(amount)
        unit = unit.rstrip(".")
        scale = unit.lower() in SCALES
        name = None if scale else self._unit(unit)[0]
        if (self.language != "cs" or self._after_own_noun(start, tags)  # "Dosáhl rychlosti 120 km/h"
                or (fraction and (scale or len(fraction) > 2 or name not in MINOR_UNITS["cs"]))
                or self._verb_case(start, self._preposition(start, tags), tags) != GEN):
            return None
        count = (integer or int(fraction.ljust(2, "0"))) if fraction else value * 10 ** SCALES.get(unit.lower(), 0)
        gender = SCALE_GENDERS["cs"][unit.lower()] if scale else NOUNS["cs"][name][0]
        verb = self._clause_verb(start, tags)
        if (1 < integer < 5 and gender == "neuter" and "Neut" in (verb.feats.get("Gender") or "")
                and "Plur" in (verb.feats.get("Number") or "") and self._no_subject_in_sight(verb, tags)):
            # an "-la" verb is also a neuter plural: "Voleb se zúčastnila dvě procenta voličů"
            taking_part = _affirmative(verb.text.lower()).startswith(("zúčastn", "účastn"))
            case = NOM if start < verb.start or taking_part else GEN
            if log:
                self._warn(text, (start, end), self._cardinal(count, case, "neuter", "inanimate"),
                           "subject or object of a verb that may be neuter plural, read as the "
                           + ("subject" if case == NOM else "object") + "; check it")
            return case
        noun = _Word(start, end, unit.lower() if scale else NOUNS["cs"][name][1][7], "NOUN",
                     {"Gender": {v: k for k, v in _UD_GENDERS.items()}[gender], "Animacy": "Inan"})
        return self._genitive_verb_case(count, start, noun, text, (start, end) if log else None, tags)

    @staticmethod
    def _after_own_noun(pos: int, tags: _Tags) -> bool:
        """Whether a genitive noun of the verb's own, no attribute and after no preposition, stands right before `pos`."""
        before = tags.before(pos, skip=("ADV", "PART"))
        return (before is not None and before.upos == "NOUN" and before.feats.get("Case") == "Gen"
                and getattr(tags.before(before.start, skip=("ADJ", "DET", "ADV", "PART")), "upos", None)
                not in ("ADP", "NOUN", "PROPN"))

    def _governing_case(self, pos: int, tags: _Tags) -> Optional[str]:
        """The case a preposition gives an amount at `pos`, also one shared with an earlier amount:
        "s 2 kg a 3 kg", "s 1 kg a 2–3 kg"."""
        prep = self._preposition(pos, tags) or self._shared_preposition(pos, tags)
        if prep is None:
            return None
        if (self.language == "cs" and prep.text.lower() in ("pod", "nad")
                and self._two_case(prep, None, prep.start, tags) == ACC):
            return ACC  # "Teplota klesla pod pět stupňů", but "je pod pěti stupni"
        return _UD_CASES.get(prep.feats.get("Case"))

    def _shared_preposition(self, pos: int, tags: _Tags) -> Optional[_Word]:
        """The preposition of an earlier time or amount that this one shares: sk "o 8.30 a 9.30 hod.",
        "od 2:00–3:00", "s 2 kg a 3 kg"."""
        i, w = bisect.bisect_left(tags.starts, pos) - 1, tags.words
        # "Pracoval s 2 kg a asi 3 kg zůstaly": a verb after the amount, in a clause with its verb, opens a new one
        opens = (i >= 0 and w[i].text.lower() in APPROXIMATORS and i + 1 < len(w)
                 and self._verb_follows(w[i + 1].end, tags))
        while i >= 0 and w[i].text.lower() in APPROXIMATORS:  # "s 2 kg a ~3 kg"
            i -= 1
        if i < 0 or w[i].text.lower() not in ("a", "nebo", "alebo", ",", "–", "—", "-"):
            return None
        while i >= 0:
            text = w[i].text
            if w[i].upos == "ADP" and not (i > 0 and w[i - 1].text == "/"):  # the "s" of "m/s" is no preposition
                return None if opens and self._verb_before(w[i].start, tags, True) else w[i]
            if text in (".", "!", "?", "…") and not (text == "." and 0 < i < len(w) - 1 and (
                    (w[i - 1].text.isdigit() and w[i + 1].text.isdigit() and w[i + 1].start == w[i].end)  # "8.30"
                    or ((re.fullmatch(_HOUR_WORD, w[i - 1].text.lower()) or w[i - 1].text.lower() in SCALES)
                        and not w[i + 1].text[:1].isupper()))):  # "hod.–3:00", but "s 2 kg. A 3 kg"
                return None
            if not (text.isdigit() or not any(ch.isalnum() for ch in text) or text.lower() in ("a", "nebo", "alebo")
                    or re.fullmatch(_HOUR_WORD, text.lower()) or text.lower() in UNIT_WORDS[self.language]
                    or text.lower() in SCALES):
                return None
            i -= 1
        return None

    def _preposition(self, pos: int, tags: _Tags) -> Optional[_Word]:
        word = tags.before(pos, skip=("ADV", "PART"))
        return word if word is not None and word.upos == "ADP" else None

    def _reference_preposition(self, pos: int, tags: _Tags) -> Optional[_Word]:
        """The preposition that governs a whole reference: "podle § 5 odst. 2" -> podle."""
        i = bisect.bisect_left(tags.starts, pos) - 1
        while i >= 0 and (tags.words[i].upos in ("ADV", "PART") or tags.words[i].text.isdigit()
                          or tags.words[i].text in ("§", ",") or tags.words[i].text.lower() in ("a", "nebo", "alebo")
                          or (tags.words[i].text == "." and i > 0  # "odst. 2", but not the end of "§ 5. § 6"
                              and f"{tags.words[i - 1].text.lower()}." in DECLINED_ABBREVIATIONS[self.language])
                          or f"{tags.words[i].text.lower()}." in DECLINED_ABBREVIATIONS[self.language]):
            i -= 1
        return tags.words[i] if i >= 0 and tags.words[i].upos == "ADP" else None

    def _preposition_case(self, pos: int, tags: _Tags) -> Optional[str]:
        prep = self._preposition(pos, tags)
        return _UD_CASES.get(prep.feats.get("Case")) if prep else None

    def _subject(self, head: _Word, pos: int, tags: _Tags) -> bool:
        """Whether a noun tagged nominative after its verb is the subject: it agrees with the verb, and nothing
        before it in the clause does ("Cíle dosáhli 2. muži", but "Dosáhlo to 2. výročí"). These verbs have a
        person for a subject, so an inanimate word does not take it from an animate one: "Cíle dosáhnou 2. muži"."""
        verb = self._clause_verb(pos, tags)
        rivals = [w for w in self._clause_before(pos, tags) if w.feats.get("Case") == "Nom" and _agree(verb, w)
                  and not (w.feats.get("Animacy") == "Inan" and head.feats.get("Animacy") == "Anim")] if verb else []
        # a present verb shows no gender, and by number alone a genitive singular that looks plural would be its
        # subject ("Obávají se 2. kola"): a plural needs a person, a singular does not ("Cíle dosáhne 2. sdružení")
        person = verb is not None and (verb.feats.get("Gender") is not None or head.feats.get("Number") != "Plur"
                                       or head.feats.get("Animacy") == "Anim")
        return head.feats.get("Case") == "Nom" and person and _agree(verb, head) and not rivals

    @staticmethod
    def _clause_before(pos: int, tags: _Tags, clause: bool = True) -> List[_Word]:
        """The words before `pos`, nearest first, back to the sentence start; with `clause`, back to a comma or a
        conjunction too."""
        words = []
        for w in reversed(tags.words[:bisect.bisect_left(tags.starts, pos)]):
            if w.text in (".", "!", "?", "…", ";", ":") or (clause and (w.text == "," or w.upos == "CCONJ")):
                break
            words.append(w)
        return words

    @staticmethod
    def _clause_after(pos: int, tags: _Tags) -> List[_Word]:
        """The words from `pos` to the end of its clause, which a comma, a sentence end or a conjunction closes."""
        words = []
        for w in tags.words[bisect.bisect_left(tags.starts, pos):]:
            if w.text in (",", ".", "!", "?", "…", ";", ":") or w.upos == "CCONJ":
                break
            words.append(w)
        return words

    @staticmethod
    def _verb_before(pos: int, tags: _Tags, clause: bool) -> bool:
        """Whether the sentence before `pos` already has its verb; with `clause`, only the clause, which a comma
        or a conjunction also closes: a name, a date or an amount is in its own ("KDYŽ PŘIŠEL, KAREL IV.",
        "PŘIŠEL A DNE 5. 6."), the end of a list is not ("PŘINESL JABLKA, HRUŠKY ATD."), but a subordinate clause
        before the list does not count: "KDYŽ PŘIŠEL, JABLKA, HRUŠKY ATD. LEŽELY"."""
        words = TextNormalizer._clause_before(pos, tags, clause)
        if clause:
            return any(w.upos in ("VERB", "AUX") for w in words)
        stretch = []  # the words between two commas, nearest first, so its first word is the last one
        for w in words + [None]:
            if w is not None and w.text != ",":
                stretch.append(w)
            elif any(x.upos in ("VERB", "AUX") for x in stretch) and stretch[-1].upos != "SCONJ":
                return True
            else:
                stretch = []
        return False

    @staticmethod
    def _verb_follows(pos: int, tags: _Tags) -> bool:
        """Whether the noun phrase after `pos` is followed by a verb, as the subject of a new sentence;
        adverbs, particles and clitics may come between: "Malé děti se potom vrátily"; or a pronoun: "On uspěl"."""
        head = tags.head_after(pos)
        if head is None:
            head = tags.after(pos)
            if head is None or head.upos != "PRON":
                return False
        i = bisect.bisect_left(tags.starts, head.end)
        for w in tags.words[i:]:
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

    def _ends_sentence(self, text: str, start: int, end: int, tags: _Tags, roman: bool = False,
                       introduces: bool = False, list_end: bool = False) -> bool:
        """Whether the period of an abbreviation, date or Roman numeral at `start`-`end` also ends a
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
        word, capitals = tags.after(end), _all_capitals(text)
        verb = next((w for w in tags.words[bisect.bisect_left(tags.starts, end):] if w.upos not in ("PRON", "ADV", "PART")),
                    None)  # past a clitic or an adverb: "KAREL IV. SE NARODIL"
        # after a name a conjunction goes on, as a lowercase "a" does: "KAREL IV. A VÁCLAV IV.", "VLÁDL KAREL IV. A POTOM";
        # after an abbreviation, whose period may end the sentence, only when the clause has no verb yet
        if capitals and verb is not None and ((verb.upos == "CCONJ" and roman)
                                              or (verb.upos in ("VERB", "AUX", "CCONJ")
                                                  and not self._verb_before(start, tags, not list_end))):
            return False  # "ROKU 300 N. L. VLÁDL", "KAREL IV. ZALOŽIL": the verb is theirs; "BYL TAM ATD. ODEŠEL" ends
        if roman:  # "Karel IV. Lucemburský" goes on; "Vládl Karel IV. Potom…", "…IV. Velký požár vypukl." do not
            return (word is None or word.upos not in ("PROPN", "ADJ")
                    or (not capitals and len(word.text) > 1 and word.text.isupper())  # "…IV. USA vznikly" is no name
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
