# -*- coding: utf-8 -*-
"""
Czech Number to Words Converter - Complete Version with Full Case & Gender Declension

Follows official Czech grammar rules (Pravidla českého pravopisu):
- Full support for all 6 grammatical cases (vocative omitted as typically = nominative)
- Full support for all 3 genders (masculine, feminine, neuter)
- Proper compound forms for numbers

Cases: nominative, genitive, dative, accusative, instrumental, locative
Genders: masculine, feminine, neuter

Key differences from Slovak:
- Different vowels: Czech ě, ů vs Slovak ä, ô
- Different number words: čtyři vs štyri, pět vs päť, etc.
- Hundreds: dvěstě, třista, čtyřista, pětset vs dvesto, tristo
"""

# =============================================================================
# CONSTANTS
# =============================================================================

CASES = ['nominative', 'genitive', 'dative', 'accusative', 'instrumental', 'locative']
CASE_INDICES = {case: i for i, case in enumerate(CASES)}

ZERO = 'nula'
MINUS = 'mínus'
POINT_WORD = 'celých'

# =============================================================================
# ONES (1-9) - Structure: {num: ((masc_cases), (fem_cases), (neut_cases))}
# Each gender tuple has 6 cases: nom, gen, dat, acc, ins, loc
# =============================================================================
ONES = {
    1: (
        ('jeden', 'jednoho', 'jednomu', 'jednoho', 'jedním', 'jednom'),     # masculine
        ('jedna', 'jedné', 'jedné', 'jednu', 'jednou', 'jedné'),            # feminine
        ('jedno', 'jednoho', 'jednomu', 'jedno', 'jedním', 'jednom'),       # neuter
    ),
    2: (
        ('dva', 'dvou', 'dvěma', 'dva', 'dvěma', 'dvou'),                   # masculine
        ('dvě', 'dvou', 'dvěma', 'dvě', 'dvěma', 'dvou'),                   # feminine
        ('dvě', 'dvou', 'dvěma', 'dvě', 'dvěma', 'dvou'),                   # neuter
    ),
    3: (
        ('tři', 'tří', 'třem', 'tři', 'třemi', 'třech'),                    # masculine
        ('tři', 'tří', 'třem', 'tři', 'třemi', 'třech'),                    # feminine
        ('tři', 'tří', 'třem', 'tři', 'třemi', 'třech'),                    # neuter
    ),
    4: (
        ('čtyři', 'čtyř', 'čtyřem', 'čtyři', 'čtyřmi', 'čtyřech'),          # masculine
        ('čtyři', 'čtyř', 'čtyřem', 'čtyři', 'čtyřmi', 'čtyřech'),          # feminine
        ('čtyři', 'čtyř', 'čtyřem', 'čtyři', 'čtyřmi', 'čtyřech'),          # neuter
    ),
    # 5-9: gender-neutral, only case matters
    5: (
        ('pět', 'pěti', 'pěti', 'pět', 'pěti', 'pěti'),
        ('pět', 'pěti', 'pěti', 'pět', 'pěti', 'pěti'),
        ('pět', 'pěti', 'pěti', 'pět', 'pěti', 'pěti'),
    ),
    6: (
        ('šest', 'šesti', 'šesti', 'šest', 'šesti', 'šesti'),
        ('šest', 'šesti', 'šesti', 'šest', 'šesti', 'šesti'),
        ('šest', 'šesti', 'šesti', 'šest', 'šesti', 'šesti'),
    ),
    7: (
        ('sedm', 'sedmi', 'sedmi', 'sedm', 'sedmi', 'sedmi'),
        ('sedm', 'sedmi', 'sedmi', 'sedm', 'sedmi', 'sedmi'),
        ('sedm', 'sedmi', 'sedmi', 'sedm', 'sedmi', 'sedmi'),
    ),
    8: (
        ('osm', 'osmi', 'osmi', 'osm', 'osmi', 'osmi'),
        ('osm', 'osmi', 'osmi', 'osm', 'osmi', 'osmi'),
        ('osm', 'osmi', 'osmi', 'osm', 'osmi', 'osmi'),
    ),
    9: (
        ('devět', 'devíti', 'devíti', 'devět', 'devíti', 'devíti'),
        ('devět', 'devíti', 'devíti', 'devět', 'devíti', 'devíti'),
        ('devět', 'devíti', 'devíti', 'devět', 'devíti', 'devíti'),
    ),
}

# =============================================================================
# TEENS (10-19) - Gender-neutral, 6 cases
# =============================================================================
TEENS = {
    0: ('deset', 'deseti', 'deseti', 'deset', 'deseti', 'deseti'),
    1: ('jedenáct', 'jedenácti', 'jedenácti', 'jedenáct', 'jedenácti', 'jedenácti'),
    2: ('dvanáct', 'dvanácti', 'dvanácti', 'dvanáct', 'dvanácti', 'dvanácti'),
    3: ('třináct', 'třinácti', 'třinácti', 'třináct', 'třinácti', 'třinácti'),
    4: ('čtrnáct', 'čtrnácti', 'čtrnácti', 'čtrnáct', 'čtrnácti', 'čtrnácti'),
    5: ('patnáct', 'patnácti', 'patnácti', 'patnáct', 'patnácti', 'patnácti'),
    6: ('šestnáct', 'šestnácti', 'šestnácti', 'šestnáct', 'šestnácti', 'šestnácti'),
    7: ('sedmnáct', 'sedmnácti', 'sedmnácti', 'sedmnáct', 'sedmnácti', 'sedmnácti'),
    8: ('osmnáct', 'osmnácti', 'osmnácti', 'osmnáct', 'osmnácti', 'osmnácti'),
    9: ('devatenáct', 'devatenácti', 'devatenácti', 'devatenáct', 'devatenácti', 'devatenácti'),
}

# =============================================================================
# TENS (20-90) - Gender-neutral, 6 cases
# =============================================================================
TENS = {
    2: ('dvacet', 'dvaceti', 'dvaceti', 'dvacet', 'dvaceti', 'dvaceti'),
    3: ('třicet', 'třiceti', 'třiceti', 'třicet', 'třiceti', 'třiceti'),
    4: ('čtyřicet', 'čtyřiceti', 'čtyřiceti', 'čtyřicet', 'čtyřiceti', 'čtyřiceti'),
    5: ('padesát', 'padesáti', 'padesáti', 'padesát', 'padesáti', 'padesáti'),
    6: ('šedesát', 'šedesáti', 'šedesáti', 'šedesát', 'šedesáti', 'šedesáti'),
    7: ('sedmdesát', 'sedmdesáti', 'sedmdesáti', 'sedmdesát', 'sedmdesáti', 'sedmdesáti'),
    8: ('osmdesát', 'osmdesáti', 'osmdesáti', 'osmdesát', 'osmdesáti', 'osmdesáti'),
    9: ('devadesát', 'devadesáti', 'devadesáti', 'devadesát', 'devadesáti', 'devadesáti'),
}

# Compound forms for tens (when followed by ones in oblique cases)
TENS_COMPOUND = {
    2: ('dvacet', 'dvacátého', 'dvacátému', 'dvacet', 'dvacátým', 'dvacátém'),
    3: ('třicet', 'třicátého', 'třicátému', 'třicet', 'třicátým', 'třicátém'),
    4: ('čtyřicet', 'čtyřicátého', 'čtyřicátému', 'čtyřicet', 'čtyřicátým', 'čtyřicátém'),
    5: ('padesát', 'padesátého', 'padesátému', 'padesát', 'padesátým', 'padesátém'),
    6: ('šedesát', 'šedesátého', 'šedesátému', 'šedesát', 'šedesátým', 'šedesátém'),
    7: ('sedmdesát', 'sedmdesátého', 'sedmdesátému', 'sedmdesát', 'sedmdesátým', 'sedmdesátém'),
    8: ('osmdesát', 'osmdesátého', 'osmdesátému', 'osmdesát', 'osmdesátým', 'osmdesátém'),
    9: ('devadesát', 'devadesátého', 'devadesátému', 'devadesát', 'devadesátým', 'devadesátém'),
}

# Compound forms for ones 1-9 when used in compound numbers
ONES_COMPOUND = {
    1: (
        ('jeden', 'jednoho', 'jednomu', 'jednoho', 'jedním', 'jednom'),
        ('jedna', 'jedné', 'jedné', 'jednu', 'jednou', 'jedné'),
        ('jedno', 'jednoho', 'jednomu', 'jedno', 'jedním', 'jednom'),
    ),
    2: (
        ('dva', 'druhého', 'druhému', 'dva', 'druhým', 'druhém'),
        ('dvě', 'druhé', 'druhé', 'dvě', 'druhou', 'druhé'),
        ('dvě', 'druhého', 'druhému', 'dvě', 'druhým', 'druhém'),
    ),
    3: (
        ('tři', 'třetího', 'třetímu', 'tři', 'třetím', 'třetím'),
        ('tři', 'třetí', 'třetí', 'tři', 'třetí', 'třetí'),
        ('tři', 'třetího', 'třetímu', 'tři', 'třetím', 'třetím'),
    ),
    4: (
        ('čtyři', 'čtvrtého', 'čtvrtému', 'čtyři', 'čtvrtým', 'čtvrtém'),
        ('čtyři', 'čtvrté', 'čtvrté', 'čtyři', 'čtvrtou', 'čtvrté'),
        ('čtyři', 'čtvrtého', 'čtvrtému', 'čtyři', 'čtvrtým', 'čtvrtém'),
    ),
    5: (
        ('pět', 'pátého', 'pátému', 'pět', 'pátým', 'pátém'),
        ('pět', 'páté', 'páté', 'pět', 'pátou', 'páté'),
        ('pět', 'pátého', 'pátému', 'pět', 'pátým', 'pátém'),
    ),
    6: (
        ('šest', 'šestého', 'šestému', 'šest', 'šestým', 'šestém'),
        ('šest', 'šesté', 'šesté', 'šest', 'šestou', 'šesté'),
        ('šest', 'šestého', 'šestému', 'šest', 'šestým', 'šestém'),
    ),
    7: (
        ('sedm', 'sedmého', 'sedmému', 'sedm', 'sedmým', 'sedmém'),
        ('sedm', 'sedmé', 'sedmé', 'sedm', 'sedmou', 'sedmé'),
        ('sedm', 'sedmého', 'sedmému', 'sedm', 'sedmým', 'sedmém'),
    ),
    8: (
        ('osm', 'osmého', 'osmému', 'osm', 'osmým', 'osmém'),
        ('osm', 'osmé', 'osmé', 'osm', 'osmou', 'osmé'),
        ('osm', 'osmého', 'osmému', 'osm', 'osmým', 'osmém'),
    ),
    9: (
        ('devět', 'devátého', 'devátému', 'devět', 'devátým', 'devátém'),
        ('devět', 'deváté', 'deváté', 'devět', 'devátou', 'deváté'),
        ('devět', 'devátého', 'devátému', 'devět', 'devátým', 'devátém'),
    ),
}

# =============================================================================
# HUNDREDS - Czech uses: sto, dvěstě, třista, čtyřista, pětset, etc.
# Structure: {hundreds: ((masc_cases), (fem_cases), (neut_cases))}
# =============================================================================
HUNDREDS = {
    1: (
        ('sto', 'stého', 'stému', 'sto', 'stým', 'stém'),
        ('stá', 'sté', 'sté', 'stou', 'stou', 'sté'),
        ('sté', 'stého', 'stému', 'sté', 'stým', 'stém'),
    ),
    2: (
        ('dvěstě', 'dvoustého', 'dvoustému', 'dvěstě', 'dvoustým', 'dvoustém'),
        ('dvěstě', 'dvousté', 'dvousté', 'dvěstě', 'dvoustou', 'dvousté'),
        ('dvěstě', 'dvoustého', 'dvoustému', 'dvěstě', 'dvoustým', 'dvoustém'),
    ),
    3: (
        ('třista', 'třístého', 'třístému', 'třista', 'třístým', 'třístém'),
        ('třista', 'třísté', 'třísté', 'třista', 'třístou', 'třísté'),
        ('třista', 'třístého', 'třístému', 'třista', 'třístým', 'třístém'),
    ),
    4: (
        ('čtyřista', 'čtyřstého', 'čtyřstému', 'čtyřista', 'čtyřstým', 'čtyřstém'),
        ('čtyřista', 'čtyřsté', 'čtyřsté', 'čtyřista', 'čtyřstou', 'čtyřsté'),
        ('čtyřista', 'čtyřstého', 'čtyřstému', 'čtyřista', 'čtyřstým', 'čtyřstém'),
    ),
    5: (
        ('pětset', 'pětistého', 'pětistému', 'pětset', 'pětistým', 'pětistém'),
        ('pětset', 'pětisté', 'pětisté', 'pětset', 'pětistou', 'pětisté'),
        ('pětset', 'pětistého', 'pětistému', 'pětset', 'pětistým', 'pětistém'),
    ),
    6: (
        ('šestset', 'šestistého', 'šestistému', 'šestset', 'šestistým', 'šestistém'),
        ('šestset', 'šestisté', 'šestisté', 'šestset', 'šestistou', 'šestisté'),
        ('šestset', 'šestistého', 'šestistému', 'šestset', 'šestistým', 'šestistém'),
    ),
    7: (
        ('sedmset', 'sedmistého', 'sedmistému', 'sedmset', 'sedmistým', 'sedmistém'),
        ('sedmset', 'sedmisté', 'sedmisté', 'sedmset', 'sedmistou', 'sedmisté'),
        ('sedmset', 'sedmistého', 'sedmistému', 'sedmset', 'sedmistým', 'sedmistém'),
    ),
    8: (
        ('osmset', 'osmistého', 'osmistému', 'osmset', 'osmistým', 'osmistém'),
        ('osmset', 'osmisté', 'osmisté', 'osmset', 'osmistou', 'osmisté'),
        ('osmset', 'osmistého', 'osmistému', 'osmset', 'osmistým', 'osmistém'),
    ),
    9: (
        ('devětset', 'devítistého', 'devítistému', 'devětset', 'devítistým', 'devítistém'),
        ('devětset', 'devítisté', 'devítisté', 'devětset', 'devítistou', 'devítisté'),
        ('devětset', 'devítistého', 'devítistému', 'devětset', 'devítistým', 'devítistém'),
    ),
}

# =============================================================================
# THOUSANDS
# =============================================================================
THOUSAND_WORD = 'tisíc'

# Compound thousands declension with gender
TISIC_COMPOUND = (
    ('tisíc', 'tisícího', 'tisícímu', 'tisíc', 'tisícím', 'tisícím'),    # masculine
    ('tisíc', 'tisící', 'tisící', 'tisíc', 'tisící', 'tisící'),          # feminine
    ('tisíc', 'tisícího', 'tisícímu', 'tisíc', 'tisícím', 'tisícím'),    # neuter
)

# Prefixes for compound thousands (2000 = dvoutisíc, etc.)
THOUSAND_PREFIXES = {
    2: 'dvou',
    3: 'tří',
    4: 'čtyř',
    5: 'pěti',
    6: 'šesti',
    7: 'sedmi',
    8: 'osmi',
    9: 'devíti',
}

# =============================================================================
# SCALES - Compound declension forms for millions and above
# =============================================================================

# Nominative forms for scales by plural index (0=singular, 1=2-4, 2=5+)
SCALE_NOM_FORMS = {
    2: ('milion', 'miliony', 'milionů'),
    3: ('miliarda', 'miliardy', 'miliard'),
    4: ('bilion', 'biliony', 'bilionů'),
    5: ('biliarda', 'biliardy', 'biliard'),
    6: ('trilion', 'triliony', 'trilionů'),
}

# Compound "t" suffix forms for oblique cases
SCALE_T_COMPOUND = {
    2: (  # milion-t-
        ('milion', 'miliontého', 'miliontému', 'milion', 'miliontým', 'miliontém'),
        ('milion', 'milionté', 'milionté', 'milion', 'miliontou', 'milionté'),
        ('milion', 'miliontého', 'miliontému', 'milion', 'miliontým', 'miliontém'),
    ),
    3: (  # miliard-t-
        ('miliarda', 'miliardtého', 'miliardtému', 'miliarda', 'miliardtým', 'miliardtém'),
        ('miliarda', 'miliardté', 'miliardté', 'miliarda', 'miliardtou', 'miliardté'),
        ('miliarda', 'miliardtého', 'miliardtému', 'miliarda', 'miliardtým', 'miliardtém'),
    ),
    4: (  # bilion-t-
        ('bilion', 'biliontého', 'biliontému', 'bilion', 'biliontým', 'biliontém'),
        ('bilion', 'bilionté', 'bilionté', 'bilion', 'biliontou', 'bilionté'),
        ('bilion', 'biliontého', 'biliontému', 'bilion', 'biliontým', 'biliontém'),
    ),
    5: (  # biliard-t-
        ('biliarda', 'biliardtého', 'biliardtému', 'biliarda', 'biliardtým', 'biliardtém'),
        ('biliarda', 'biliardté', 'biliardté', 'biliarda', 'biliardtou', 'biliardté'),
        ('biliarda', 'biliardtého', 'biliardtému', 'biliarda', 'biliardtým', 'biliardtém'),
    ),
    6: (  # trilion-t-
        ('trilion', 'triliontého', 'triliontému', 'trilion', 'triliontým', 'triliontém'),
        ('trilion', 'trilionté', 'trilionté', 'trilion', 'triliontou', 'trilionté'),
        ('trilion', 'triliontého', 'triliontému', 'trilion', 'triliontým', 'triliontém'),
    ),
}

# Prefixes for compound scales - masculine forms
SCALE_PREFIXES_MASC = {
    2: 'dva',
    3: 'tři',
    4: 'čtyři',
    5: 'pět',
    6: 'šest',
    7: 'sedm',
    8: 'osm',
    9: 'devět',
}

# Prefixes for compound scales - feminine forms (for miliarda, biliarda)
SCALE_PREFIXES_FEM = {
    2: 'dvě',
    3: 'tři',
    4: 'čtyři',
    5: 'pět',
    6: 'šest',
    7: 'sedm',
    8: 'osm',
    9: 'devět',
}

SCALE_VALUES = {
    2: 10**6, 3: 10**9, 4: 10**12, 5: 10**15, 6: 10**18,
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_gender_index(gender) -> int:
    """Get index for gender: 0=masculine, 1=feminine, 2=neuter"""
    if gender in ('masculine', 'm', 0):
        return 0
    elif gender in ('feminine', 'f', 1):
        return 1
    return 2  # neuter


def get_case_index(case) -> int:
    """Get index for case."""
    if isinstance(case, int):
        return case
    return CASE_INDICES.get(case, 0)


def get_plural_index(count: int) -> int:
    """Determine plural form index based on count."""
    if count == 1:
        return 0
    last_two = abs(count) % 100
    if 11 <= last_two <= 14:
        return 2
    last_digit = abs(count) % 10
    if 2 <= last_digit <= 4:
        return 1
    return 2


SCALE_GENDERS = {
    2: 'm',  # milion
    3: 'f',  # miliarda
    4: 'm',  # bilion
    5: 'f',  # biliarda
    6: 'm',  # trilion
}


def get_scale_gender(scale_idx: int) -> str:
    """Get gender of scale word."""
    return SCALE_GENDERS.get(scale_idx, 'm')


# =============================================================================
# CORE CONVERSION FUNCTIONS
# =============================================================================

def convert_ones(n: int, gender_idx: int, case_idx: int) -> str:
    """Convert 1-9 with gender and case."""
    if n == 0:
        return ''
    return ONES[n][gender_idx][case_idx]


def convert_teens(n: int, case_idx: int) -> str:
    """Convert 10-19 with case (gender-neutral)."""
    return TEENS[n][case_idx]


def convert_tens(t: int, case_idx: int) -> str:
    """Convert 20, 30, ... 90 with case (gender-neutral)."""
    return TENS[t][case_idx]


def convert_0_99(n: int, gender_idx: int, case_idx: int) -> str:
    """Convert 0-99 with gender and case."""
    if n == 0:
        return ''
    if n < 10:
        return convert_ones(n, gender_idx, case_idx)
    if n < 20:
        return convert_teens(n - 10, case_idx)
    
    tens_digit = n // 10
    ones_digit = n % 10
    
    if ones_digit == 0:
        return convert_tens(tens_digit, case_idx)
    else:
        if case_idx in (0, 3):  # nominative or accusative
            tens_word = TENS[tens_digit][case_idx]
            ones_word = convert_ones(ones_digit, gender_idx, case_idx)
        else:
            tens_word = TENS_COMPOUND[tens_digit][case_idx]
            ones_word = ONES_COMPOUND[ones_digit][gender_idx][case_idx]
        
        return tens_word + ones_word


def convert_0_999(n: int, gender_idx: int, case_idx: int) -> str:
    """Convert 0-999 with gender and case."""
    if n == 0:
        return ''
    
    hundreds = n // 100
    remainder = n % 100
    
    if hundreds == 0:
        return convert_0_99(remainder, gender_idx, case_idx)
    
    if remainder == 0:
        return HUNDREDS[hundreds][gender_idx][case_idx]
    else:
        hundreds_word = HUNDREDS[hundreds][gender_idx][0]  # nominative
        remainder_word = convert_0_99(remainder, gender_idx, case_idx)
        return hundreds_word + remainder_word


def convert_thousands_compound(n: int, gender_idx: int, case_idx: int, has_remainder: bool = False) -> str:
    """Convert 1-999 thousands with gender and case declension."""
    if n == 0:
        return ''
    
    if n == 1:
        if has_remainder:
            return THOUSAND_WORD
        else:
            return TISIC_COMPOUND[gender_idx][case_idx]
    
    if n < 10:
        prefix = THOUSAND_PREFIXES[n]
        suffix = TISIC_COMPOUND[gender_idx][case_idx]
        return prefix + suffix
    
    prefix = convert_0_999(n, 0, 0)
    suffix = TISIC_COMPOUND[gender_idx][case_idx]
    return prefix + suffix


def convert_below_million(n: int, gender_idx: int, case_idx: int) -> str:
    """Convert number below 1 million."""
    if n == 0:
        return ''
    if n < 1000:
        return convert_0_999(n, gender_idx, case_idx)
    
    thousands = n // 1000
    remainder = n % 1000
    
    result = convert_thousands_compound(thousands, gender_idx, case_idx, has_remainder=(remainder > 0))
    
    if remainder > 0:
        result += convert_0_999(remainder, gender_idx, case_idx)
    
    return result


def convert_scale_compound(count: int, scale_idx: int, gender_idx: int, case_idx: int, has_remainder: bool = False) -> str:
    """Convert a count of a scale (milion, miliarda, etc.) as compound form."""
    scale_gender = get_scale_gender(scale_idx)
    is_feminine_scale = (scale_gender == 'f')
    plural_idx = get_plural_index(count)
    
    if count == 1:
        if has_remainder or case_idx in (0, 3):
            return SCALE_NOM_FORMS[scale_idx][0]
        else:
            return SCALE_T_COMPOUND[scale_idx][gender_idx][case_idx]
    
    if count < 10:
        if is_feminine_scale:
            prefix = SCALE_PREFIXES_FEM.get(count, '')
        else:
            prefix = SCALE_PREFIXES_MASC.get(count, '')
        if not prefix:
            prefix = convert_ones(count, 0, 0)
    elif count < 1000:
        prefix = convert_0_999(count, 0, 0)
    elif count < 1_000_000:
        prefix = convert_below_million(count, 0, 0)
    else:
        prefix = int_to_cardinal_compound(count, 'masculine', 'nominative')
    
    if has_remainder or case_idx in (0, 3):
        suffix = SCALE_NOM_FORMS[scale_idx][plural_idx]
    else:
        suffix = SCALE_T_COMPOUND[scale_idx][gender_idx][case_idx]
    
    return prefix + suffix


def int_to_cardinal_compound(n: int, gender: str, case: str) -> str:
    """Convert integer to Czech cardinal number as compound (no spaces)."""
    gender_idx = get_gender_index(gender)
    case_idx = get_case_index(case)
    
    if n < 1_000_000:
        return convert_below_million(n, gender_idx, case_idx)
    
    result = ''
    remaining = n
    
    for scale_idx in sorted(SCALE_VALUES.keys(), reverse=True):
        scale_value = SCALE_VALUES[scale_idx]
        
        if remaining >= scale_value:
            count = remaining // scale_value
            remaining = remaining % scale_value
            
            has_remainder = (remaining > 0)
            scale_part = convert_scale_compound(count, scale_idx, gender_idx, case_idx, has_remainder)
            result += scale_part
    
    if remaining > 0:
        result += convert_below_million(remaining, gender_idx, case_idx)
    
    return result


# =============================================================================
# ORDINAL NUMBERS
# =============================================================================

ORDINAL_ONES = {
    1: (  # první (soft pattern)
        ('první', 'prvního', 'prvnímu', 'prvního', 'prvním', 'prvním'),
        ('první', 'první', 'první', 'první', 'první', 'první'),
        ('první', 'prvního', 'prvnímu', 'první', 'prvním', 'prvním'),
    ),
    2: (  # druhý (hard pattern)
        ('druhý', 'druhého', 'druhému', 'druhého', 'druhým', 'druhém'),
        ('druhá', 'druhé', 'druhé', 'druhou', 'druhou', 'druhé'),
        ('druhé', 'druhého', 'druhému', 'druhé', 'druhým', 'druhém'),
    ),
    3: (  # třetí (soft pattern)
        ('třetí', 'třetího', 'třetímu', 'třetího', 'třetím', 'třetím'),
        ('třetí', 'třetí', 'třetí', 'třetí', 'třetí', 'třetí'),
        ('třetí', 'třetího', 'třetímu', 'třetí', 'třetím', 'třetím'),
    ),
    4: (  # čtvrtý (hard pattern)
        ('čtvrtý', 'čtvrtého', 'čtvrtému', 'čtvrtého', 'čtvrtým', 'čtvrtém'),
        ('čtvrtá', 'čtvrté', 'čtvrté', 'čtvrtou', 'čtvrtou', 'čtvrté'),
        ('čtvrté', 'čtvrtého', 'čtvrtému', 'čtvrté', 'čtvrtým', 'čtvrtém'),
    ),
    5: (  # pátý (hard pattern)
        ('pátý', 'pátého', 'pátému', 'pátého', 'pátým', 'pátém'),
        ('pátá', 'páté', 'páté', 'pátou', 'pátou', 'páté'),
        ('páté', 'pátého', 'pátému', 'páté', 'pátým', 'pátém'),
    ),
    6: (  # šestý (hard pattern)
        ('šestý', 'šestého', 'šestému', 'šestého', 'šestým', 'šestém'),
        ('šestá', 'šesté', 'šesté', 'šestou', 'šestou', 'šesté'),
        ('šesté', 'šestého', 'šestému', 'šesté', 'šestým', 'šestém'),
    ),
    7: (  # sedmý (hard pattern)
        ('sedmý', 'sedmého', 'sedmému', 'sedmého', 'sedmým', 'sedmém'),
        ('sedmá', 'sedmé', 'sedmé', 'sedmou', 'sedmou', 'sedmé'),
        ('sedmé', 'sedmého', 'sedmému', 'sedmé', 'sedmým', 'sedmém'),
    ),
    8: (  # osmý (hard pattern)
        ('osmý', 'osmého', 'osmému', 'osmého', 'osmým', 'osmém'),
        ('osmá', 'osmé', 'osmé', 'osmou', 'osmou', 'osmé'),
        ('osmé', 'osmého', 'osmému', 'osmé', 'osmým', 'osmém'),
    ),
    9: (  # devátý (hard pattern)
        ('devátý', 'devátého', 'devátému', 'devátého', 'devátým', 'devátém'),
        ('devátá', 'deváté', 'deváté', 'devátou', 'devátou', 'deváté'),
        ('deváté', 'devátého', 'devátému', 'deváté', 'devátým', 'devátém'),
    ),
}

ORDINAL_TEENS = {
    0: (  # desátý (10th)
        ('desátý', 'desátého', 'desátému', 'desátého', 'desátým', 'desátém'),
        ('desátá', 'desáté', 'desáté', 'desátou', 'desátou', 'desáté'),
        ('desáté', 'desátého', 'desátému', 'desáté', 'desátým', 'desátém'),
    ),
    1: (  # jedenáctý (11th)
        ('jedenáctý', 'jedenáctého', 'jedenáctému', 'jedenáctého', 'jedenáctým', 'jedenáctém'),
        ('jedenáctá', 'jedenácté', 'jedenácté', 'jedenáctou', 'jedenáctou', 'jedenácté'),
        ('jedenácté', 'jedenáctého', 'jedenáctému', 'jedenácté', 'jedenáctým', 'jedenáctém'),
    ),
    2: (  # dvanáctý (12th)
        ('dvanáctý', 'dvanáctého', 'dvanáctému', 'dvanáctého', 'dvanáctým', 'dvanáctém'),
        ('dvanáctá', 'dvanácté', 'dvanácté', 'dvanáctou', 'dvanáctou', 'dvanácté'),
        ('dvanácté', 'dvanáctého', 'dvanáctému', 'dvanácté', 'dvanáctým', 'dvanáctém'),
    ),
    3: (  # třináctý (13th)
        ('třináctý', 'třináctého', 'třináctému', 'třináctého', 'třináctým', 'třináctém'),
        ('třináctá', 'třinácté', 'třinácté', 'třináctou', 'třináctou', 'třinácté'),
        ('třinácté', 'třináctého', 'třináctému', 'třinácté', 'třináctým', 'třináctém'),
    ),
    4: (  # čtrnáctý (14th)
        ('čtrnáctý', 'čtrnáctého', 'čtrnáctému', 'čtrnáctého', 'čtrnáctým', 'čtrnáctém'),
        ('čtrnáctá', 'čtrnácté', 'čtrnácté', 'čtrnáctou', 'čtrnáctou', 'čtrnácté'),
        ('čtrnácté', 'čtrnáctého', 'čtrnáctému', 'čtrnácté', 'čtrnáctým', 'čtrnáctém'),
    ),
    5: (  # patnáctý (15th)
        ('patnáctý', 'patnáctého', 'patnáctému', 'patnáctého', 'patnáctým', 'patnáctém'),
        ('patnáctá', 'patnácté', 'patnácté', 'patnáctou', 'patnáctou', 'patnácté'),
        ('patnácté', 'patnáctého', 'patnáctému', 'patnácté', 'patnáctým', 'patnáctém'),
    ),
    6: (  # šestnáctý (16th)
        ('šestnáctý', 'šestnáctého', 'šestnáctému', 'šestnáctého', 'šestnáctým', 'šestnáctém'),
        ('šestnáctá', 'šestnácté', 'šestnácté', 'šestnáctou', 'šestnáctou', 'šestnácté'),
        ('šestnácté', 'šestnáctého', 'šestnáctému', 'šestnácté', 'šestnáctým', 'šestnáctém'),
    ),
    7: (  # sedmnáctý (17th)
        ('sedmnáctý', 'sedmnáctého', 'sedmnáctému', 'sedmnáctého', 'sedmnáctým', 'sedmnáctém'),
        ('sedmnáctá', 'sedmnácté', 'sedmnácté', 'sedmnáctou', 'sedmnáctou', 'sedmnácté'),
        ('sedmnácté', 'sedmnáctého', 'sedmnáctému', 'sedmnácté', 'sedmnáctým', 'sedmnáctém'),
    ),
    8: (  # osmnáctý (18th)
        ('osmnáctý', 'osmnáctého', 'osmnáctému', 'osmnáctého', 'osmnáctým', 'osmnáctém'),
        ('osmnáctá', 'osmnácté', 'osmnácté', 'osmnáctou', 'osmnáctou', 'osmnácté'),
        ('osmnácté', 'osmnáctého', 'osmnáctému', 'osmnácté', 'osmnáctým', 'osmnáctém'),
    ),
    9: (  # devatenáctý (19th)
        ('devatenáctý', 'devatenáctého', 'devatenáctému', 'devatenáctého', 'devatenáctým', 'devatenáctém'),
        ('devatenáctá', 'devatenácté', 'devatenácté', 'devatenáctou', 'devatenáctou', 'devatenácté'),
        ('devatenácté', 'devatenáctého', 'devatenáctému', 'devatenácté', 'devatenáctým', 'devatenáctém'),
    ),
}

ORDINAL_TENS = {
    2: (  # dvacátý (20th)
        ('dvacátý', 'dvacátého', 'dvacátému', 'dvacátého', 'dvacátým', 'dvacátém'),
        ('dvacátá', 'dvacáté', 'dvacáté', 'dvacátou', 'dvacátou', 'dvacáté'),
        ('dvacáté', 'dvacátého', 'dvacátému', 'dvacáté', 'dvacátým', 'dvacátém'),
    ),
    3: (  # třicátý (30th)
        ('třicátý', 'třicátého', 'třicátému', 'třicátého', 'třicátým', 'třicátém'),
        ('třicátá', 'třicáté', 'třicáté', 'třicátou', 'třicátou', 'třicáté'),
        ('třicáté', 'třicátého', 'třicátému', 'třicáté', 'třicátým', 'třicátém'),
    ),
    4: (  # čtyřicátý (40th)
        ('čtyřicátý', 'čtyřicátého', 'čtyřicátému', 'čtyřicátého', 'čtyřicátým', 'čtyřicátém'),
        ('čtyřicátá', 'čtyřicáté', 'čtyřicáté', 'čtyřicátou', 'čtyřicátou', 'čtyřicáté'),
        ('čtyřicáté', 'čtyřicátého', 'čtyřicátému', 'čtyřicáté', 'čtyřicátým', 'čtyřicátém'),
    ),
    5: (  # padesátý (50th)
        ('padesátý', 'padesátého', 'padesátému', 'padesátého', 'padesátým', 'padesátém'),
        ('padesátá', 'padesáté', 'padesáté', 'padesátou', 'padesátou', 'padesáté'),
        ('padesáté', 'padesátého', 'padesátému', 'padesáté', 'padesátým', 'padesátém'),
    ),
    6: (  # šedesátý (60th)
        ('šedesátý', 'šedesátého', 'šedesátému', 'šedesátého', 'šedesátým', 'šedesátém'),
        ('šedesátá', 'šedesáté', 'šedesáté', 'šedesátou', 'šedesátou', 'šedesáté'),
        ('šedesáté', 'šedesátého', 'šedesátému', 'šedesáté', 'šedesátým', 'šedesátém'),
    ),
    7: (  # sedmdesátý (70th)
        ('sedmdesátý', 'sedmdesátého', 'sedmdesátému', 'sedmdesátého', 'sedmdesátým', 'sedmdesátém'),
        ('sedmdesátá', 'sedmdesáté', 'sedmdesáté', 'sedmdesátou', 'sedmdesátou', 'sedmdesáté'),
        ('sedmdesáté', 'sedmdesátého', 'sedmdesátému', 'sedmdesáté', 'sedmdesátým', 'sedmdesátém'),
    ),
    8: (  # osmdesátý (80th)
        ('osmdesátý', 'osmdesátého', 'osmdesátému', 'osmdesátého', 'osmdesátým', 'osmdesátém'),
        ('osmdesátá', 'osmdesáté', 'osmdesáté', 'osmdesátou', 'osmdesátou', 'osmdesáté'),
        ('osmdesáté', 'osmdesátého', 'osmdesátému', 'osmdesáté', 'osmdesátým', 'osmdesátém'),
    ),
    9: (  # devadesátý (90th)
        ('devadesátý', 'devadesátého', 'devadesátému', 'devadesátého', 'devadesátým', 'devadesátém'),
        ('devadesátá', 'devadesáté', 'devadesáté', 'devadesátou', 'devadesátou', 'devadesáté'),
        ('devadesáté', 'devadesátého', 'devadesátému', 'devadesáté', 'devadesátým', 'devadesátém'),
    ),
}

ORDINAL_HUNDREDS = {
    1: (  # stý (100th)
        ('stý', 'stého', 'stému', 'stého', 'stým', 'stém'),
        ('stá', 'sté', 'sté', 'stou', 'stou', 'sté'),
        ('sté', 'stého', 'stému', 'sté', 'stým', 'stém'),
    ),
    2: (  # dvoustý (200th)
        ('dvoustý', 'dvoustého', 'dvoustému', 'dvoustého', 'dvoustým', 'dvoustém'),
        ('dvoustá', 'dvousté', 'dvousté', 'dvoustou', 'dvoustou', 'dvousté'),
        ('dvousté', 'dvoustého', 'dvoustému', 'dvousté', 'dvoustým', 'dvoustém'),
    ),
    3: (  # třístý (300th)
        ('třístý', 'třístého', 'třístému', 'třístého', 'třístým', 'třístém'),
        ('třístá', 'třísté', 'třísté', 'třístou', 'třístou', 'třísté'),
        ('třísté', 'třístého', 'třístému', 'třísté', 'třístým', 'třístém'),
    ),
    4: (  # čtyřstý (400th)
        ('čtyřstý', 'čtyřstého', 'čtyřstému', 'čtyřstého', 'čtyřstým', 'čtyřstém'),
        ('čtyřstá', 'čtyřsté', 'čtyřsté', 'čtyřstou', 'čtyřstou', 'čtyřsté'),
        ('čtyřsté', 'čtyřstého', 'čtyřstému', 'čtyřsté', 'čtyřstým', 'čtyřstém'),
    ),
    5: (  # pětistý (500th)
        ('pětistý', 'pětistého', 'pětistému', 'pětistého', 'pětistým', 'pětistém'),
        ('pětistá', 'pětisté', 'pětisté', 'pětistou', 'pětistou', 'pětisté'),
        ('pětisté', 'pětistého', 'pětistému', 'pětisté', 'pětistým', 'pětistém'),
    ),
    6: (  # šestistý (600th)
        ('šestistý', 'šestistého', 'šestistému', 'šestistého', 'šestistým', 'šestistém'),
        ('šestistá', 'šestisté', 'šestisté', 'šestistou', 'šestistou', 'šestisté'),
        ('šestisté', 'šestistého', 'šestistému', 'šestisté', 'šestistým', 'šestistém'),
    ),
    7: (  # sedmistý (700th)
        ('sedmistý', 'sedmistého', 'sedmistému', 'sedmistého', 'sedmistým', 'sedmistém'),
        ('sedmistá', 'sedmisté', 'sedmisté', 'sedmistou', 'sedmistou', 'sedmisté'),
        ('sedmisté', 'sedmistého', 'sedmistému', 'sedmisté', 'sedmistým', 'sedmistém'),
    ),
    8: (  # osmistý (800th)
        ('osmistý', 'osmistého', 'osmistému', 'osmistého', 'osmistým', 'osmistém'),
        ('osmistá', 'osmisté', 'osmisté', 'osmistou', 'osmistou', 'osmisté'),
        ('osmisté', 'osmistého', 'osmistému', 'osmisté', 'osmistým', 'osmistém'),
    ),
    9: (  # devítistý (900th)
        ('devítistý', 'devítistého', 'devítistému', 'devítistého', 'devítistým', 'devítistém'),
        ('devítistá', 'devítisté', 'devítisté', 'devítistou', 'devítistou', 'devítisté'),
        ('devítisté', 'devítistého', 'devítistému', 'devítisté', 'devítistým', 'devítistém'),
    ),
}

ORDINAL_TISICI = (
    ('tisící', 'tisícího', 'tisícímu', 'tisícího', 'tisícím', 'tisícím'),
    ('tisící', 'tisící', 'tisící', 'tisící', 'tisící', 'tisící'),
    ('tisící', 'tisícího', 'tisícímu', 'tisící', 'tisícím', 'tisícím'),
)

ORDINAL_THOUSAND_PREFIXES = {
    2: 'dvou',
    3: 'tří',
    4: 'čtyř',
    5: 'pěti',
    6: 'šesti',
    7: 'sedmi',
    8: 'osmi',
    9: 'devíti',
}

ORDINAL_SCALE = {
    2: (  # miliontý
        ('miliontý', 'miliontého', 'miliontému', 'miliontého', 'miliontým', 'miliontém'),
        ('miliontá', 'milionté', 'milionté', 'miliontou', 'miliontou', 'milionté'),
        ('milionté', 'miliontého', 'miliontému', 'milionté', 'miliontým', 'miliontém'),
    ),
    3: (  # miliardtý
        ('miliardtý', 'miliardtého', 'miliardtému', 'miliardtého', 'miliardtým', 'miliardtém'),
        ('miliardtá', 'miliardté', 'miliardté', 'miliardtou', 'miliardtou', 'miliardté'),
        ('miliardté', 'miliardtého', 'miliardtému', 'miliardté', 'miliardtým', 'miliardtém'),
    ),
    4: (  # biliontý
        ('biliontý', 'biliontého', 'biliontému', 'biliontého', 'biliontým', 'biliontém'),
        ('biliontá', 'bilionté', 'bilionté', 'biliontou', 'biliontou', 'bilionté'),
        ('bilionté', 'biliontého', 'biliontému', 'bilionté', 'biliontým', 'biliontém'),
    ),
    5: (  # biliardtý
        ('biliardtý', 'biliardtého', 'biliardtému', 'biliardtého', 'biliardtým', 'biliardtém'),
        ('biliardtá', 'biliardté', 'biliardté', 'biliardtou', 'biliardtou', 'biliardté'),
        ('biliardté', 'biliardtého', 'biliardtému', 'biliardté', 'biliardtým', 'biliardtém'),
    ),
    6: (  # triliontý
        ('triliontý', 'triliontého', 'triliontému', 'triliontého', 'triliontým', 'triliontém'),
        ('triliontá', 'trilionté', 'trilionté', 'triliontou', 'triliontou', 'trilionté'),
        ('trilionté', 'triliontého', 'triliontému', 'trilionté', 'triliontým', 'triliontém'),
    ),
}

ORDINAL_SCALE_PREFIXES = {
    2: 'dvou',
    3: 'tří',
    4: 'čtyř',
    5: 'pěti',
    6: 'šesti',
    7: 'sedmi',
    8: 'osmi',
    9: 'devíti',
}


# =============================================================================
# ORDINAL CONVERSION FUNCTIONS
# =============================================================================

def ordinal_ones(n: int, gender_idx: int, case_idx: int) -> str:
    return ORDINAL_ONES[n][gender_idx][case_idx]

def ordinal_teens(n: int, gender_idx: int, case_idx: int) -> str:
    return ORDINAL_TEENS[n][gender_idx][case_idx]

def ordinal_tens(t: int, gender_idx: int, case_idx: int) -> str:
    return ORDINAL_TENS[t][gender_idx][case_idx]

def ordinal_hundreds(h: int, gender_idx: int, case_idx: int) -> str:
    return ORDINAL_HUNDREDS[h][gender_idx][case_idx]

def ordinal_0_99(n: int, gender_idx: int, case_idx: int) -> str:
    if n == 0:
        return ''
    if n < 10:
        return ordinal_ones(n, gender_idx, case_idx)
    if n < 20:
        return ordinal_teens(n - 10, gender_idx, case_idx)
    
    tens_digit = n // 10
    ones_digit = n % 10
    
    if ones_digit == 0:
        return ordinal_tens(tens_digit, gender_idx, case_idx)
    else:
        tens_word = ORDINAL_TENS[tens_digit][gender_idx][case_idx]
        ones_word = ordinal_ones(ones_digit, gender_idx, case_idx)
        return tens_word + ones_word

def ordinal_0_999(n: int, gender_idx: int, case_idx: int) -> str:
    if n == 0:
        return ''
    
    hundreds = n // 100
    remainder = n % 100
    
    if hundreds == 0:
        return ordinal_0_99(remainder, gender_idx, case_idx)
    
    if remainder == 0:
        return ordinal_hundreds(hundreds, gender_idx, case_idx)
    else:
        hundreds_word = ORDINAL_HUNDREDS[hundreds][gender_idx][0]
        remainder_word = ordinal_0_99(remainder, gender_idx, case_idx)
        return hundreds_word + remainder_word

def ordinal_thousands(n: int, gender_idx: int, case_idx: int, has_remainder: bool = False) -> str:
    if n == 0:
        return ''
    
    if n == 1:
        if has_remainder:
            return ORDINAL_TISICI[gender_idx][0]
        return ORDINAL_TISICI[gender_idx][case_idx]
    
    if n < 10:
        prefix = ORDINAL_THOUSAND_PREFIXES.get(n, '')
        if not prefix:
            prefix = convert_0_999(n, 0, 0)
    else:
        prefix = convert_0_999(n, 0, 0)
    
    if has_remainder:
        suffix = ORDINAL_TISICI[gender_idx][0]
    else:
        suffix = ORDINAL_TISICI[gender_idx][case_idx]
    
    return prefix + suffix

def ordinal_below_million(n: int, gender_idx: int, case_idx: int) -> str:
    if n == 0:
        return ''
    if n < 1000:
        return ordinal_0_999(n, gender_idx, case_idx)
    
    thousands = n // 1000
    remainder = n % 1000
    
    result = ordinal_thousands(thousands, gender_idx, case_idx, has_remainder=(remainder > 0))
    
    if remainder > 0:
        result += ordinal_0_999(remainder, gender_idx, case_idx)
    
    return result

def ordinal_scale_word(count: int, scale_idx: int, gender_idx: int, case_idx: int, has_remainder: bool = False) -> str:
    scale_ordinal = ORDINAL_SCALE[scale_idx]
    
    if count == 1:
        if has_remainder:
            return scale_ordinal[gender_idx][0]
        return scale_ordinal[gender_idx][case_idx]
    
    if count < 10:
        prefix = ORDINAL_SCALE_PREFIXES.get(count, '')
        if not prefix:
            prefix = convert_0_999(count, 0, 0)
    elif count < 1_000_000:
        prefix = convert_below_million(count, 0, 0)
    else:
        prefix = int_to_cardinal_compound(count, 'masculine', 'nominative')
    
    if has_remainder:
        suffix = scale_ordinal[gender_idx][0]
    else:
        suffix = scale_ordinal[gender_idx][case_idx]
    
    return prefix + suffix

def int_to_ordinal(n: int, gender: str = 'masculine', case: str = 'nominative') -> str:
    if n <= 0:
        raise ValueError("Ordinal numbers must be positive integers")
    
    gender_idx = get_gender_index(gender)
    case_idx = get_case_index(case)
    
    if n < 1_000_000:
        return ordinal_below_million(n, gender_idx, case_idx)
    
    result = ''
    remaining = n
    
    for scale_idx in sorted(SCALE_VALUES.keys(), reverse=True):
        scale_value = SCALE_VALUES[scale_idx]
        
        if remaining >= scale_value:
            count = remaining // scale_value
            remaining = remaining % scale_value
            
            has_remainder = (remaining > 0)
            result += ordinal_scale_word(count, scale_idx, gender_idx, case_idx, has_remainder)
    
    if remaining > 0:
        result += ordinal_below_million(remaining, gender_idx, case_idx)
    
    return result


# =============================================================================
# MAIN CONVERSION FUNCTION
# =============================================================================

def int_to_cardinal(n: int, gender: str = 'masculine', case: str = 'nominative') -> str:
    if n < 0:
        return MINUS + ' ' + int_to_cardinal(abs(n), gender, case)
    if n == 0:
        return ZERO
    return int_to_cardinal_compound(n, gender, case)

def float_to_cardinal(n: float, gender: str = 'masculine', case: str = 'nominative') -> str:
    if n < 0:
        return MINUS + ' ' + float_to_cardinal(abs(n), gender, case)
    
    str_n = str(n)
    if '.' in str_n:
        int_part, dec_part = str_n.split('.')
    else:
        return int_to_cardinal(int(n), gender, case)
    
    int_words = int_to_cardinal(int(int_part), gender, case)
    
    if not dec_part or int(dec_part) == 0:
        return int_words
    
    leading_zeros = len(dec_part) - len(dec_part.lstrip('0'))
    zero_words = (ZERO + ' ') * leading_zeros
    
    dec_value = int(dec_part.lstrip('0')) if dec_part.lstrip('0') else 0
    dec_words = int_to_cardinal(dec_value, gender, case) if dec_value else ''
    
    return int_words + ' ' + POINT_WORD + ' ' + zero_words + dec_words


# =============================================================================
# API CLASS
# =============================================================================

class Num2Word_CS:
    """Czech number to words converter."""
    
    def __init__(self):
        self.negword = MINUS
        self.pointword = POINT_WORD
    
    def to_cardinal(self, number, **kwargs) -> str:
        gender = kwargs.get('gender', 'masculine')
        case = kwargs.get('case', 'nominative')
        if isinstance(number, float):
            return float_to_cardinal(number, gender, case)
        return int_to_cardinal(int(number), gender, case)
    
    def to_ordinal(self, number, **kwargs) -> str:
        gender = kwargs.get('gender', 'masculine')
        case = kwargs.get('case', 'nominative')
        return int_to_ordinal(int(number), gender, case)


def num2words(number, to: str = 'cardinal', **kwargs) -> str:
    """Convert number to Czech words."""
    converter = Num2Word_CS()
    if to == 'ordinal':
        return converter.to_ordinal(number, **kwargs)
    return converter.to_cardinal(number, **kwargs)


if __name__ == '__main__':
    print("=" * 80)
    print("CZECH NUM2WORDS - Full Gender + Case Declension Test")
    print("=" * 80)
    
    print("\n--- Cardinal 1-10 (masculine nominative) ---")
    for n in range(1, 11):
        result = num2words(n, gender='masculine', case='nominative')
        print(f"  {n}: {result}")
    
    print("\n--- Ordinal 1st-10th (masculine nominative) ---")
    for n in range(1, 11):
        result = num2words(n, to='ordinal', gender='masculine', case='nominative')
        print(f"  {n}: {result}")
