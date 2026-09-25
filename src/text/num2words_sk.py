# -*- coding: utf-8 -*-
"""
Slovak Number to Words Converter - Complete Version with Full Case & Gender Declension

Follows official Slovak grammar rules (Pravidlá slovenského pravopisu):
- Full support for all 6 grammatical cases
- Full support for all 3 genders (masculine, feminine, neuter)
- Proper compound forms: dvetisíc, tritisíc, päťtisíc
- Separate words for millions+: dva milióny, päť miliónov

Cases: nominative, genitive, dative, accusative, instrumental, locative
Genders: masculine, feminine, neuter

Key rules:
- HUNDREDS decline by both gender AND case: sto/stého/stému (masc), stá/stej/stej (fem)
- ONES 1-4 decline by gender, 5-9 are gender-neutral but case-sensitive
- TENS decline by case only
- Compound thousands (dvetisíc) are indeclinable
- Millions+ are separate words with full declension
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
        ('jeden', 'jedného', 'jednému', 'jedného', 'jedným', 'jednom'),     # masculine
        ('jedna', 'jednej', 'jednej', 'jednu', 'jednou', 'jednej'),         # feminine
        ('jedno', 'jedného', 'jednému', 'jedno', 'jedným', 'jednom'),       # neuter
    ),
    2: (
        ('dva', 'dvoch', 'dvom', 'dva', 'dvoma', 'dvoch'),                  # masculine
        ('dve', 'dvoch', 'dvom', 'dve', 'dvomi', 'dvoch'),                  # feminine
        ('dve', 'dvoch', 'dvom', 'dve', 'dvoma', 'dvoch'),                  # neuter
    ),
    3: (
        ('tri', 'troch', 'trom', 'troch', 'troma', 'troch'),               # masculine (acc=gen for animate)
        ('tri', 'troch', 'trom', 'tri', 'tromi', 'troch'),                 # feminine
        ('tri', 'troch', 'trom', 'tri', 'troma', 'troch'),                 # neuter
    ),
    4: (
        ('štyri', 'štyroch', 'štyrom', 'štyroch', 'štyrmi', 'štyroch'),    # masculine
        ('štyri', 'štyroch', 'štyrom', 'štyri', 'štyrmi', 'štyroch'),      # feminine
        ('štyri', 'štyroch', 'štyrom', 'štyri', 'štyrmi', 'štyroch'),      # neuter
    ),
    # 5-9: gender-neutral, only case matters (same forms for all genders)
    5: (
        ('päť', 'piatich', 'piatim', 'päť', 'piatimi', 'piatich'),
        ('päť', 'piatich', 'piatim', 'päť', 'piatimi', 'piatich'),
        ('päť', 'piatich', 'piatim', 'päť', 'piatimi', 'piatich'),
    ),
    6: (
        ('šesť', 'šiestich', 'šiestim', 'šesť', 'šiestimi', 'šiestich'),
        ('šesť', 'šiestich', 'šiestim', 'šesť', 'šiestimi', 'šiestich'),
        ('šesť', 'šiestich', 'šiestim', 'šesť', 'šiestimi', 'šiestich'),
    ),
    7: (
        ('sedem', 'siedmich', 'siedmim', 'sedem', 'siedmimi', 'siedmich'),
        ('sedem', 'siedmich', 'siedmim', 'sedem', 'siedmimi', 'siedmich'),
        ('sedem', 'siedmich', 'siedmim', 'sedem', 'siedmimi', 'siedmich'),
    ),
    8: (
        ('osem', 'ôsmich', 'ôsmim', 'osem', 'ôsmimi', 'ôsmich'),
        ('osem', 'ôsmich', 'ôsmim', 'osem', 'ôsmimi', 'ôsmich'),
        ('osem', 'ôsmich', 'ôsmim', 'osem', 'ôsmimi', 'ôsmich'),
    ),
    9: (
        ('deväť', 'deviatich', 'deviatim', 'deväť', 'deviatimi', 'deviatich'),
        ('deväť', 'deviatich', 'deviatim', 'deväť', 'deviatimi', 'deviatich'),
        ('deväť', 'deviatich', 'deviatim', 'deväť', 'deviatimi', 'deviatich'),
    ),
}

# =============================================================================
# TEENS (10-19) - Gender-neutral, 6 cases
# =============================================================================
TEENS = {
    0: ('desať', 'desiatich', 'desiatim', 'desať', 'desiatimi', 'desiatich'),
    1: ('jedenásť', 'jedenástich', 'jedenástim', 'jedenásť', 'jedenástimi', 'jedenástich'),
    2: ('dvanásť', 'dvanástich', 'dvanástim', 'dvanásť', 'dvanástimi', 'dvanástich'),
    3: ('trinásť', 'trinástich', 'trinástim', 'trinásť', 'trinástimi', 'trinástich'),
    4: ('štrnásť', 'štrnástich', 'štrnástim', 'štrnásť', 'štrnástimi', 'štrnástich'),
    5: ('pätnásť', 'pätnástich', 'pätnástim', 'pätnásť', 'pätnástimi', 'pätnástich'),
    6: ('šestnásť', 'šestnástich', 'šestnástim', 'šestnásť', 'šestnástimi', 'šestnástich'),
    7: ('sedemnásť', 'sedemnástich', 'sedemnástim', 'sedemnásť', 'sedemnástimi', 'sedemnástich'),
    8: ('osemnásť', 'osemnástich', 'osemnástim', 'osemnásť', 'osemnástimi', 'osemnástich'),
    9: ('devätnásť', 'devätnástich', 'devätnástim', 'devätnásť', 'devätnástimi', 'devätnástich'),
}

# =============================================================================
# TENS (20-90) - Gender-neutral, 6 cases
# =============================================================================
TENS = {
    2: ('dvadsať', 'dvadsiatich', 'dvadsiatim', 'dvadsať', 'dvadsiatimi', 'dvadsiatich'),
    3: ('tridsať', 'tridsiatich', 'tridsiatim', 'tridsať', 'tridsiatimi', 'tridsiatich'),
    4: ('štyridsať', 'štyridsiatich', 'štyridsiatim', 'štyridsať', 'štyridsiatimi', 'štyridsiatich'),
    5: ('päťdesiat', 'päťdesiatich', 'päťdesiatim', 'päťdesiat', 'päťdesiatimi', 'päťdesiatich'),
    6: ('šesťdesiat', 'šesťdesiatich', 'šesťdesiatim', 'šesťdesiat', 'šesťdesiatimi', 'šesťdesiatich'),
    7: ('sedemdesiat', 'sedemdesiatich', 'sedemdesiatim', 'sedemdesiat', 'sedemdesiatimi', 'sedemdesiatich'),
    8: ('osemdesiat', 'osemdesiatich', 'osemdesiatim', 'osemdesiat', 'osemdesiatimi', 'osemdesiatich'),
    9: ('deväťdesiat', 'deväťdesiatich', 'deväťdesiatim', 'deväťdesiat', 'deväťdesiatimi', 'deväťdesiatich'),
}

# Singular adjectival forms for ALL tens (2-9) when used in compounds with ones
# In oblique cases, tens decline to singular adjectival form
# nom/acc stay cardinal, other cases use singular adjectival forms
TENS_COMPOUND = {
    2: ('dvadsať', 'dvadsiateho', 'dvadsiatemu', 'dvadsať', 'dvadsiatym', 'dvadsiatom'),
    3: ('tridsať', 'tridsiateho', 'tridsiatemu', 'tridsať', 'tridsiatym', 'tridsiatom'),
    4: ('štyridsať', 'štyridsiateho', 'štyridsiatemu', 'štyridsať', 'štyridsiatym', 'štyridsiatom'),
    5: ('päťdesiat', 'päťdesiateho', 'päťdesiatemu', 'päťdesiat', 'päťdesiatym', 'päťdesiatom'),
    6: ('šesťdesiat', 'šesťdesiateho', 'šesťdesiatemu', 'šesťdesiat', 'šesťdesiatym', 'šesťdesiatom'),
    7: ('sedemdesiat', 'sedemdesiateho', 'sedemdesiatemu', 'sedemdesiat', 'sedemdesiatym', 'sedemdesiatom'),
    8: ('osemdesiat', 'osemdesiateho', 'osemdesiatemu', 'osemdesiat', 'osemdesiatym', 'osemdesiatom'),
    9: ('deväťdesiat', 'deväťdesiateho', 'deväťdesiatemu', 'deväťdesiat', 'deväťdesiatym', 'deväťdesiatom'),
}

# Singular adjectival forms for ones 5-9 when used in compounds (to match TENS_COMPOUND)
# Structure: {num: ((masc_cases), (fem_cases), (neut_cases))}
# nom/acc stay cardinal, other cases use singular adjectival forms
ONES_COMPOUND = {
    1: (
        ('jeden', 'jedného', 'jednému', 'jedného', 'jedným', 'jednom'),     # masculine (same as ONES)
        ('jedna', 'jednej', 'jednej', 'jednu', 'jednou', 'jednej'),         # feminine
        ('jedno', 'jedného', 'jednému', 'jedno', 'jedným', 'jednom'),       # neuter
    ),
    2: (
        ('dva', 'druhého', 'druhému', 'dva', 'druhým', 'druhom'),           # masculine singular
        ('dve', 'druhej', 'druhej', 'dve', 'druhou', 'druhej'),             # feminine singular
        ('dve', 'druhého', 'druhému', 'dve', 'druhým', 'druhom'),           # neuter singular
    ),
    3: (
        ('tri', 'tretieho', 'tretiemu', 'tri', 'tretím', 'treťom'),         # masculine singular
        ('tri', 'tretej', 'tretej', 'tri', 'treťou', 'tretej'),             # feminine singular
        ('tri', 'tretieho', 'tretiemu', 'tri', 'tretím', 'treťom'),         # neuter singular
    ),
    4: (
        ('štyri', 'štvrtého', 'štvrtému', 'štyri', 'štvrtým', 'štvrtom'),   # masculine singular
        ('štyri', 'štvrtej', 'štvrtej', 'štyri', 'štvrtou', 'štvrtej'),     # feminine singular
        ('štyri', 'štvrtého', 'štvrtému', 'štyri', 'štvrtým', 'štvrtom'),   # neuter singular
    ),
    5: (
        ('päť', 'piateho', 'piatemu', 'päť', 'piatym', 'piatom'),      # masculine
        ('päť', 'piatej', 'piatej', 'päť', 'piatou', 'piatej'),        # feminine
        ('päť', 'piateho', 'piatemu', 'päť', 'piatym', 'piatom'),      # neuter
    ),
    6: (
        ('šesť', 'šiesteho', 'šiestemu', 'šesť', 'šiestym', 'šiestom'),
        ('šesť', 'šiestej', 'šiestej', 'šesť', 'šiestou', 'šiestej'),
        ('šesť', 'šiesteho', 'šiestemu', 'šesť', 'šiestym', 'šiestom'),
    ),
    7: (
        ('sedem', 'siedmeho', 'siedmemu', 'sedem', 'siedmym', 'siedmom'),
        ('sedem', 'siedmej', 'siedmej', 'sedem', 'siedmou', 'siedmej'),
        ('sedem', 'siedmeho', 'siedmemu', 'sedem', 'siedmym', 'siedmom'),
    ),
    8: (
        ('osem', 'ôsmeho', 'ôsmemu', 'osem', 'ôsmym', 'ôsmom'),
        ('osem', 'ôsmej', 'ôsmej', 'osem', 'ôsmou', 'ôsmej'),
        ('osem', 'ôsmeho', 'ôsmemu', 'osem', 'ôsmym', 'ôsmom'),
    ),
    9: (
        ('deväť', 'deviateho', 'deviatemu', 'deväť', 'deviatym', 'deviatom'),
        ('deväť', 'deviatej', 'deviatej', 'deväť', 'deviatou', 'deviatej'),
        ('deväť', 'deviateho', 'deviatemu', 'deväť', 'deviatym', 'deviatom'),
    ),
}

# =============================================================================
# HUNDREDS (100-900) - Gender AND case sensitive
# Structure: {num: ((masc_cases), (fem_cases), (neut_cases))}
# Each gender has 6 cases: nom, gen, dat, acc, ins, loc
# =============================================================================
HUNDREDS = {
    1: (
        ('sto', 'stého', 'stému', 'sto', 'stým', 'stom'),           # masculine
        ('stá', 'stej', 'stej', 'stú', 'stou', 'stej'),             # feminine
        ('sté', 'stého', 'stému', 'sté', 'stým', 'stom'),           # neuter
    ),
    2: (
        ('dvesto', 'dvestého', 'dvestému', 'dvesto', 'dvestým', 'dvestom'),
        ('dvestá', 'dvestej', 'dvestej', 'dvestú', 'dvestou', 'dvestej'),
        ('dvesté', 'dvestého', 'dvestému', 'dvesté', 'dvestým', 'dvestom'),
    ),
    3: (
        ('tristo', 'tristého', 'tristému', 'tristo', 'tristým', 'tristom'),
        ('tristá', 'tristej', 'tristej', 'tristú', 'tristou', 'tristej'),
        ('tristé', 'tristého', 'tristému', 'tristé', 'tristým', 'tristom'),
    ),
    4: (
        ('štyristo', 'štyristého', 'štyristému', 'štyristo', 'štyristým', 'štyristom'),
        ('štyristá', 'štyristej', 'štyristej', 'štyristú', 'štyristou', 'štyristej'),
        ('štyristé', 'štyristého', 'štyristému', 'štyristé', 'štyristým', 'štyristom'),
    ),
    5: (
        ('päťsto', 'päťstého', 'päťstému', 'päťsto', 'päťstým', 'päťstom'),
        ('päťstá', 'päťstej', 'päťstej', 'päťstú', 'päťstou', 'päťstej'),
        ('päťsté', 'päťstého', 'päťstému', 'päťsté', 'päťstým', 'päťstom'),
    ),
    6: (
        ('šesťsto', 'šesťstého', 'šesťstému', 'šesťsto', 'šesťstým', 'šesťstom'),
        ('šesťstá', 'šesťstej', 'šesťstej', 'šesťstú', 'šesťstou', 'šesťstej'),
        ('šesťsté', 'šesťstého', 'šesťstému', 'šesťsté', 'šesťstým', 'šesťstom'),
    ),
    7: (
        ('sedemsto', 'sedemstého', 'sedemstému', 'sedemsto', 'sedemstým', 'sedemstom'),
        ('sedemstá', 'sedemstej', 'sedemstej', 'sedemstú', 'sedemstou', 'sedemstej'),
        ('sedemsté', 'sedemstého', 'sedemstému', 'sedemsté', 'sedemstým', 'sedemstom'),
    ),
    8: (
        ('osemsto', 'osemstého', 'osemstému', 'osemsto', 'osemstým', 'osemstom'),
        ('osemstá', 'osemstej', 'osemstej', 'osemstú', 'osemstou', 'osemstej'),
        ('osemsté', 'osemstého', 'osemstému', 'osemsté', 'osemstým', 'osemstom'),
    ),
    9: (
        ('deväťsto', 'deväťstého', 'deväťstému', 'deväťsto', 'deväťstým', 'deväťstom'),
        ('deväťstá', 'deväťstej', 'deväťstej', 'deväťstú', 'deväťstou', 'deväťstej'),
        ('deväťsté', 'deväťstého', 'deväťstému', 'deväťsté', 'deväťstým', 'deväťstom'),
    ),
}

# =============================================================================
# THOUSANDS compound prefixes (indeclinable in compound form)
# =============================================================================
THOUSAND_PREFIXES = {
    1: '',        # tisíc (not jedentisíc)
    2: 'dve',     # dvetisíc
    3: 'tri',     # tritisíc
    4: 'štyri',   # štyritisíc
    5: 'päť',     # päťtisíc
    6: 'šesť',    # šesťtisíc
    7: 'sedem',   # sedemtisíc
    8: 'osem',    # osemtisíc
    9: 'deväť',   # deväťtisíc
}

THOUSAND_WORD = 'tisíc'

# Cardinal "tisíc" case declension (NO gender variation for cardinals)
# Used when "1000" appears alone, not in compounds like "dvetisíc"
TISIC_CARDINAL = (
    'tisíc',     # nominative
    'tisíca',    # genitive  
    'tisícu',    # dative
    'tisíc',     # accusative
    'tisícom',   # instrumental
    'tisíci',    # locative
)

# Compound thousands declension with gender (for dvetisíc, desaťtisíc, etc.)
# Structure: ((masc_cases), (fem_cases), (neut_cases))
# Each gender has 6 cases: nom, gen, dat, acc, ins, loc
TISIC_COMPOUND = (
    ('tisíc', 'tisíceho', 'tisícemu', 'tisíc', 'tisícim', 'tisícom'),    # masculine
    ('tisíc', 'tisícej', 'tisícej', 'tisíc', 'tisícou', 'tisícej'),      # feminine
    ('tisíc', 'tisíceho', 'tisícemu', 'tisíc', 'tisícim', 'tisícom'),    # neuter
)

# =============================================================================
# SCALES - Compound declension forms for millions and above
# =============================================================================

# Nominative forms for scales by plural index (0=singular, 1=2-4, 2=5+)
SCALE_NOM_FORMS = {
    2: ('milión', 'milióny', 'miliónov'),
    3: ('miliarda', 'miliardy', 'miliárd'),
    4: ('bilión', 'bilióny', 'biliónov'),
    5: ('biliarda', 'biliardy', 'biliárd'),
    6: ('trilión', 'trilióny', 'triliónov'),
}

# Compound "t" suffix forms for oblique cases (gen, dat, ins, loc)
# Structure: ((masc_cases), (fem_cases), (neut_cases))
# Cases: nom (unused - use SCALE_NOM_FORMS), gen, dat, acc (same as nom), ins, loc
SCALE_T_COMPOUND = {
    2: (  # milión-t-
        ('milión', 'miliónteho', 'milióntemu', 'milión', 'milióntym', 'milióntom'),
        ('milión', 'milióntej', 'milióntej', 'milión', 'milióntou', 'milióntej'),
        ('milión', 'miliónteho', 'milióntemu', 'milión', 'milióntym', 'milióntom'),
    ),
    3: (  # miliard-t-
        ('miliarda', 'miliardtého', 'miliardtému', 'miliarda', 'miliardtým', 'miliardtom'),
        ('miliarda', 'miliardtej', 'miliardtej', 'miliarda', 'miliardtou', 'miliardtej'),
        ('miliarda', 'miliardtého', 'miliardtému', 'miliarda', 'miliardtým', 'miliardtom'),
    ),
    4: (  # bilión-t-
        ('bilión', 'biliónteho', 'bilióntemu', 'bilión', 'bilióntym', 'bilióntom'),
        ('bilión', 'bilióntej', 'bilióntej', 'bilión', 'bilióntou', 'bilióntej'),
        ('bilión', 'biliónteho', 'bilióntemu', 'bilión', 'bilióntym', 'bilióntom'),
    ),
    5: (  # biliard-t-
        ('biliarda', 'biliardtého', 'biliardtému', 'biliarda', 'biliardtým', 'biliardtom'),
        ('biliarda', 'biliardtej', 'biliardtej', 'biliarda', 'biliardtou', 'biliardtej'),
        ('biliarda', 'biliardtého', 'biliardtému', 'biliarda', 'biliardtým', 'biliardtom'),
    ),
    6: (  # trilión-t-
        ('trilión', 'triliónteho', 'trilióntemu', 'trilión', 'trilióntym', 'trilióntom'),
        ('trilión', 'trilióntej', 'trilióntej', 'trilión', 'trilióntou', 'trilióntej'),
        ('trilión', 'triliónteho', 'trilióntemu', 'trilión', 'trilióntym', 'trilióntom'),
    ),
}

# Prefixes for compound scales - masculine forms
SCALE_PREFIXES_MASC = {
    2: 'dva',
    3: 'tri',
    4: 'štyri',
    5: 'päť',
    6: 'šesť',
    7: 'sedem',
    8: 'osem',
    9: 'deväť',
}

# Prefixes for compound scales - feminine forms (for miliarda, biliarda)
SCALE_PREFIXES_FEM = {
    2: 'dve',
    3: 'tri',
    4: 'štyri',
    5: 'päť',
    6: 'šesť',
    7: 'sedem',
    8: 'osem',
    9: 'deväť',
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
    """Get index for case (0-5)"""
    if isinstance(case, int):
        return min(case, 5)
    return CASE_INDICES.get(case, 0)


def get_plural_index(count: int) -> int:
    """
    Determine plural form index based on count.
    0 = singular (1)
    1 = plural 2-4
    2 = plural 5+ (and 0, 11-14)
    """
    if count == 1:
        return 0
    last_two = abs(count) % 100
    if 11 <= last_two <= 14:
        return 2
    last_digit = abs(count) % 10
    if 2 <= last_digit <= 4:
        return 1
    return 2


# Gender of scale words
SCALE_GENDERS = {
    2: 'm',  # milión
    3: 'f',  # miliarda
    4: 'm',  # bilión
    5: 'f',  # biliarda
    6: 'm',  # trilión
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


def convert_hundreds(h: int, gender_idx: int, case_idx: int) -> str:
    """Convert 100-900 with BOTH gender AND case."""
    return HUNDREDS[h][gender_idx][case_idx]


def convert_0_99(n: int, gender_idx: int, case_idx: int) -> str:
    """Convert 0-99 with declension. Always compound form (no spaces)."""
    if n == 0:
        return ''
    if n < 10:
        return convert_ones(n, gender_idx, case_idx)
    if n < 20:
        return convert_teens(n - 10, case_idx)
    
    tens_digit = n // 10
    ones_digit = n % 10
    
    if ones_digit == 0:
        # Tens alone: use full case declension (plural quantitative forms)
        return convert_tens(tens_digit, case_idx)
    else:
        # Compound (e.g., 21, 55, 99): 
        # All tens use singular adjectival forms in oblique cases
        tens_word = TENS_COMPOUND[tens_digit][case_idx]
        
        # All ones use singular compound forms in compounds
        ones_word = ONES_COMPOUND[ones_digit][gender_idx][case_idx]
        
        return tens_word + ones_word


def convert_0_999(n: int, gender_idx: int, case_idx: int) -> str:
    """Convert 0-999 with full gender+case declension."""
    if n == 0:
        return ''
    
    hundreds = n // 100
    remainder = n % 100
    
    result = ''
    if hundreds > 0:
        if remainder > 0:
            # Hundreds stay NOMINATIVE when followed by other digits
            # e.g., 313 genitive: "tristotrinástich" not "tristéhotrinástich"
            result = convert_hundreds(hundreds, gender_idx, 0)  # nominative
        else:
            # Hundreds alone: full declension
            result = convert_hundreds(hundreds, gender_idx, case_idx)
    
    if remainder > 0:
        remainder_word = convert_0_99(remainder, gender_idx, case_idx)
        if result:
            result += remainder_word  # compound: stodvadsať
        else:
            result = remainder_word
    
    return result


def convert_thousands_compound(n: int, gender_idx: int, case_idx: int, has_remainder: bool = False) -> str:
    """
    Convert 1-999 thousands with gender and case declension.
    All thousands (including 1000 alone) use adjectival declension with gender.
    """
    if n == 0:
        return ''
    
    if n == 1:
        # "tisíc" - uses adjectival declension with gender
        if has_remainder:
            return THOUSAND_WORD  # "tisíc" - undeclined when followed by remainder
        else:
            return TISIC_COMPOUND[gender_idx][case_idx]  # tisíc/tisíceho/tisícemu/...
    
    # Compound forms 2-9: prefix + tisíc with gender+case
    if n < 10:
        prefix = THOUSAND_PREFIXES[n]
        suffix = TISIC_COMPOUND[gender_idx][case_idx]
        return prefix + suffix
    
    # For 10-999 thousands: nominative prefix + tisíc with gender+case
    prefix = convert_0_999(n, 0, 0)  # nominative masculine for prefix
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
    
    # Thousands part - pass gender_idx and has_remainder
    result = convert_thousands_compound(thousands, gender_idx, case_idx, has_remainder=(remainder > 0))
    
    if remainder > 0:
        # Remainder uses proper gender+case declension
        result += convert_0_999(remainder, gender_idx, case_idx)
    
    return result


def convert_scale_compound(count: int, scale_idx: int, gender_idx: int, case_idx: int, has_remainder: bool = False) -> str:
    """
    Convert a count of a scale (milión, miliarda, etc.) as compound form.
    E.g., 5 million -> päťmiliónov (nominative), päťmiliónteho (genitive)
    
    Uses:
    - SCALE_NOM_FORMS for nominative/accusative (with plural selection)
    - SCALE_T_COMPOUND for oblique cases (gen, dat, ins, loc)
    """
    scale_gender = get_scale_gender(scale_idx)
    is_feminine_scale = (scale_gender == 'f')
    plural_idx = get_plural_index(count)
    
    if count == 1:
        # Singular form: just the scale word with proper declension
        if has_remainder or case_idx in (0, 3):  # nominative or accusative
            return SCALE_NOM_FORMS[scale_idx][0]  # singular nominative
        else:
            return SCALE_T_COMPOUND[scale_idx][gender_idx][case_idx]
    
    # Build prefix based on count
    if count < 10:
        # Use simple prefix with gender agreement for feminine scales
        if is_feminine_scale:
            prefix = SCALE_PREFIXES_FEM.get(count, '')
        else:
            prefix = SCALE_PREFIXES_MASC.get(count, '')
        if not prefix:
            prefix = convert_ones(count, 0, 0)  # nominative masculine fallback
    elif count < 1000:
        # Use nominative form for prefix
        prefix = convert_0_999(count, 0, 0)
    elif count < 1_000_000:
        # Thousands prefix
        prefix = convert_below_million(count, 0, 0)
    else:
        # Recursive for very large counts
        prefix = int_to_cardinal_compound(count, 'masculine', 'nominative')
    
    # Choose suffix based on case
    if has_remainder or case_idx in (0, 3):  # nominative or accusative
        # Use appropriate plural form
        suffix = SCALE_NOM_FORMS[scale_idx][plural_idx]
    else:
        # Use t-suffix forms for oblique cases
        suffix = SCALE_T_COMPOUND[scale_idx][gender_idx][case_idx]
    
    return prefix + suffix


def int_to_cardinal_compound(n: int, gender: str, case: str) -> str:
    """
    Convert integer to Slovak cardinal number as compound (no spaces).
    Internal recursive function.
    """
    gender_idx = get_gender_index(gender)
    case_idx = get_case_index(case)
    
    if n < 1_000_000:
        return convert_below_million(n, gender_idx, case_idx)
    
    # Build compound from largest scale down
    result = ''
    remaining = n
    
    for scale_idx in sorted(SCALE_VALUES.keys(), reverse=True):
        scale_value = SCALE_VALUES[scale_idx]
        
        if remaining >= scale_value:
            count = remaining // scale_value
            remaining = remaining % scale_value
            
            has_remainder = (remaining > 0)
            
            # Convert scale part as compound
            scale_part = convert_scale_compound(count, scale_idx, gender_idx, case_idx, has_remainder)
            result += scale_part
    
    # Add remaining part (below million)
    if remaining > 0:
        result += convert_below_million(remaining, gender_idx, case_idx)
    
    return result


# =============================================================================
# ORDINAL NUMBERS - Slovak ordinals with full gender+case declension
# Ordinals are adjectives in Slovak, following hard/soft adjective patterns
# =============================================================================

# Ordinal ones (1st-9th) - full adjective declension
# Structure: {num: ((masc_cases), (fem_cases), (neut_cases))}
# Cases: nom, gen, dat, acc, ins, loc
ORDINAL_ONES = {
    1: (  # prvý (hard pattern)
        ('prvý', 'prvého', 'prvému', 'prvého', 'prvým', 'prvom'),
        ('prvá', 'prvej', 'prvej', 'prvú', 'prvou', 'prvej'),
        ('prvé', 'prvého', 'prvému', 'prvé', 'prvým', 'prvom'),
    ),
    2: (  # druhý (hard pattern)
        ('druhý', 'druhého', 'druhému', 'druhého', 'druhým', 'druhom'),
        ('druhá', 'druhej', 'druhej', 'druhú', 'druhou', 'druhej'),
        ('druhé', 'druhého', 'druhému', 'druhé', 'druhým', 'druhom'),
    ),
    3: (  # tretí (soft pattern)
        ('tretí', 'tretieho', 'tretiemu', 'tretieho', 'tretím', 'treťom'),
        ('tretia', 'tretej', 'tretej', 'tretiu', 'treťou', 'tretej'),
        ('tretie', 'tretieho', 'tretiemu', 'tretie', 'tretím', 'treťom'),
    ),
    4: (  # štvrtý (hard pattern)
        ('štvrtý', 'štvrtého', 'štvrtému', 'štvrtého', 'štvrtým', 'štvrtom'),
        ('štvrtá', 'štvrtej', 'štvrtej', 'štvrtú', 'štvrtou', 'štvrtej'),
        ('štvrté', 'štvrtého', 'štvrtému', 'štvrté', 'štvrtým', 'štvrtom'),
    ),
    5: (  # piaty (hard pattern)
        ('piaty', 'piateho', 'piatemu', 'piateho', 'piatym', 'piatom'),
        ('piata', 'piatej', 'piatej', 'piatu', 'piatou', 'piatej'),
        ('piate', 'piateho', 'piatemu', 'piate', 'piatym', 'piatom'),
    ),
    6: (  # šiesty (hard pattern)
        ('šiesty', 'šiesteho', 'šiestemu', 'šiesteho', 'šiestym', 'šiestom'),
        ('šiesta', 'šiestej', 'šiestej', 'šiestu', 'šiestou', 'šiestej'),
        ('šieste', 'šiesteho', 'šiestemu', 'šieste', 'šiestym', 'šiestom'),
    ),
    7: (  # siedmy (hard pattern)
        ('siedmy', 'siedmeho', 'siedmemu', 'siedmeho', 'siedmym', 'siedmom'),
        ('siedma', 'siedmej', 'siedmej', 'siedmu', 'siedmou', 'siedmej'),
        ('siedme', 'siedmeho', 'siedmemu', 'siedme', 'siedmym', 'siedmom'),
    ),
    8: (  # ôsmy (hard pattern)
        ('ôsmy', 'ôsmeho', 'ôsmemu', 'ôsmeho', 'ôsmym', 'ôsmom'),
        ('ôsma', 'ôsmej', 'ôsmej', 'ôsmu', 'ôsmou', 'ôsmej'),
        ('ôsme', 'ôsmeho', 'ôsmemu', 'ôsme', 'ôsmym', 'ôsmom'),
    ),
    9: (  # deviaty (hard pattern)
        ('deviaty', 'deviateho', 'deviatemu', 'deviateho', 'deviatym', 'deviatom'),
        ('deviata', 'deviatej', 'deviatej', 'deviatu', 'deviatou', 'deviatej'),
        ('deviate', 'deviateho', 'deviatemu', 'deviate', 'deviatym', 'deviatom'),
    ),
}

# Ordinal teens (10th-19th) - follow hard adjective pattern
# Structure: {num: ((masc_cases), (fem_cases), (neut_cases))}
ORDINAL_TEENS = {
    0: (  # desiaty (10th)
        ('desiaty', 'desiateho', 'desiatemu', 'desiateho', 'desiatym', 'desiatom'),
        ('desiata', 'desiatej', 'desiatej', 'desiatu', 'desiatou', 'desiatej'),
        ('desiate', 'desiateho', 'desiatemu', 'desiate', 'desiatym', 'desiatom'),
    ),
    1: (  # jedenásty (11th)
        ('jedenásty', 'jedenásteho', 'jedenástemu', 'jedenásteho', 'jedenástym', 'jedenástom'),
        ('jedenásta', 'jedenástej', 'jedenástej', 'jedenástu', 'jedenástou', 'jedenástej'),
        ('jedenáste', 'jedenásteho', 'jedenástemu', 'jedenáste', 'jedenástym', 'jedenástom'),
    ),
    2: (  # dvanásty (12th)
        ('dvanásty', 'dvanásteho', 'dvanástemu', 'dvanásteho', 'dvanástym', 'dvanástom'),
        ('dvanásta', 'dvanástej', 'dvanástej', 'dvanástu', 'dvanástou', 'dvanástej'),
        ('dvanáste', 'dvanásteho', 'dvanástemu', 'dvanáste', 'dvanástym', 'dvanástom'),
    ),
    3: (  # trinásty (13th)
        ('trinásty', 'trinásteho', 'trinástemu', 'trinásteho', 'trinástym', 'trinástom'),
        ('trinásta', 'trinástej', 'trinástej', 'trinástu', 'trinástou', 'trinástej'),
        ('trináste', 'trinásteho', 'trinástemu', 'trináste', 'trinástym', 'trinástom'),
    ),
    4: (  # štrnásty (14th)
        ('štrnásty', 'štrnásteho', 'štrnástemu', 'štrnásteho', 'štrnástym', 'štrnástom'),
        ('štrnásta', 'štrnástej', 'štrnástej', 'štrnástu', 'štrnástou', 'štrnástej'),
        ('štrnáste', 'štrnásteho', 'štrnástemu', 'štrnáste', 'štrnástym', 'štrnástom'),
    ),
    5: (  # pätnásty (15th)
        ('pätnásty', 'pätnásteho', 'pätnástemu', 'pätnásteho', 'pätnástym', 'pätnástom'),
        ('pätnásta', 'pätnástej', 'pätnástej', 'pätnástu', 'pätnástou', 'pätnástej'),
        ('pätnáste', 'pätnásteho', 'pätnástemu', 'pätnáste', 'pätnástym', 'pätnástom'),
    ),
    6: (  # šestnásty (16th)
        ('šestnásty', 'šestnásteho', 'šestnástemu', 'šestnásteho', 'šestnástym', 'šestnástom'),
        ('šestnásta', 'šestnástej', 'šestnástej', 'šestnástu', 'šestnástou', 'šestnástej'),
        ('šestnáste', 'šestnásteho', 'šestnástemu', 'šestnáste', 'šestnástym', 'šestnástom'),
    ),
    7: (  # sedemnásty (17th)
        ('sedemnásty', 'sedemnásteho', 'sedemnástemu', 'sedemnásteho', 'sedemnástym', 'sedemnástom'),
        ('sedemnásta', 'sedemnástej', 'sedemnástej', 'sedemnástu', 'sedemnástou', 'sedemnástej'),
        ('sedemnáste', 'sedemnásteho', 'sedemnástemu', 'sedemnáste', 'sedemnástym', 'sedemnástom'),
    ),
    8: (  # osemnásty (18th)
        ('osemnásty', 'osemnásteho', 'osemnástemu', 'osemnásteho', 'osemnástym', 'osemnástom'),
        ('osemnásta', 'osemnástej', 'osemnástej', 'osemnástu', 'osemnástou', 'osemnástej'),
        ('osemnáste', 'osemnásteho', 'osemnástemu', 'osemnáste', 'osemnástym', 'osemnástom'),
    ),
    9: (  # devätnásty (19th)
        ('devätnásty', 'devätnásteho', 'devätnástemu', 'devätnásteho', 'devätnástym', 'devätnástom'),
        ('devätnásta', 'devätnástej', 'devätnástej', 'devätnástu', 'devätnástou', 'devätnástej'),
        ('devätnáste', 'devätnásteho', 'devätnástemu', 'devätnáste', 'devätnástym', 'devätnástom'),
    ),
}

# Ordinal tens (20th, 30th, ... 90th) - all follow hard adjective pattern
# Structure: {tens_digit: ((masc_cases), (fem_cases), (neut_cases))}
ORDINAL_TENS = {
    2: (  # dvadsiaty (20th)
        ('dvadsiaty', 'dvadsiateho', 'dvadsiatemu', 'dvadsiateho', 'dvadsiatym', 'dvadsiatom'),
        ('dvadsiata', 'dvadsiatej', 'dvadsiatej', 'dvadsiatu', 'dvadsiatou', 'dvadsiatej'),
        ('dvadsiate', 'dvadsiateho', 'dvadsiatemu', 'dvadsiate', 'dvadsiatym', 'dvadsiatom'),
    ),
    3: (  # tridsiaty (30th)
        ('tridsiaty', 'tridsiateho', 'tridsiatemu', 'tridsiateho', 'tridsiatym', 'tridsiatom'),
        ('tridsiata', 'tridsiatej', 'tridsiatej', 'tridsiatu', 'tridsiatou', 'tridsiatej'),
        ('tridsiate', 'tridsiateho', 'tridsiatemu', 'tridsiate', 'tridsiatym', 'tridsiatom'),
    ),
    4: (  # štyridsiaty (40th)
        ('štyridsiaty', 'štyridsiateho', 'štyridsiatemu', 'štyridsiateho', 'štyridsiatym', 'štyridsiatom'),
        ('štyridsiata', 'štyridsiatej', 'štyridsiatej', 'štyridsiatu', 'štyridsiatou', 'štyridsiatej'),
        ('štyridsiate', 'štyridsiateho', 'štyridsiatemu', 'štyridsiate', 'štyridsiatym', 'štyridsiatom'),
    ),
    5: (  # päťdesiaty (50th)
        ('päťdesiaty', 'päťdesiateho', 'päťdesiatemu', 'päťdesiateho', 'päťdesiatym', 'päťdesiatom'),
        ('päťdesiata', 'päťdesiatej', 'päťdesiatej', 'päťdesiatu', 'päťdesiatou', 'päťdesiatej'),
        ('päťdesiate', 'päťdesiateho', 'päťdesiatemu', 'päťdesiate', 'päťdesiatym', 'päťdesiatom'),
    ),
    6: (  # šesťdesiaty (60th)
        ('šesťdesiaty', 'šesťdesiateho', 'šesťdesiatemu', 'šesťdesiateho', 'šesťdesiatym', 'šesťdesiatom'),
        ('šesťdesiata', 'šesťdesiatej', 'šesťdesiatej', 'šesťdesiatu', 'šesťdesiatou', 'šesťdesiatej'),
        ('šesťdesiate', 'šesťdesiateho', 'šesťdesiatemu', 'šesťdesiate', 'šesťdesiatym', 'šesťdesiatom'),
    ),
    7: (  # sedemdesiaty (70th)
        ('sedemdesiaty', 'sedemdesiateho', 'sedemdesiatemu', 'sedemdesiateho', 'sedemdesiatym', 'sedemdesiatom'),
        ('sedemdesiata', 'sedemdesiatej', 'sedemdesiatej', 'sedemdesiatu', 'sedemdesiatou', 'sedemdesiatej'),
        ('sedemdesiate', 'sedemdesiateho', 'sedemdesiatemu', 'sedemdesiate', 'sedemdesiatym', 'sedemdesiatom'),
    ),
    8: (  # osemdesiaty (80th)
        ('osemdesiaty', 'osemdesiateho', 'osemdesiatemu', 'osemdesiateho', 'osemdesiatym', 'osemdesiatom'),
        ('osemdesiata', 'osemdesiatej', 'osemdesiatej', 'osemdesiatu', 'osemdesiatou', 'osemdesiatej'),
        ('osemdesiate', 'osemdesiateho', 'osemdesiatemu', 'osemdesiate', 'osemdesiatym', 'osemdesiatom'),
    ),
    9: (  # deväťdesiaty (90th)
        ('deväťdesiaty', 'deväťdesiateho', 'deväťdesiatemu', 'deväťdesiateho', 'deväťdesiatym', 'deväťdesiatom'),
        ('deväťdesiata', 'deväťdesiatej', 'deväťdesiatej', 'deväťdesiatu', 'deväťdesiatou', 'deväťdesiatej'),
        ('deväťdesiate', 'deväťdesiateho', 'deväťdesiatemu', 'deväťdesiate', 'deväťdesiatym', 'deväťdesiatom'),
    ),
}

# Ordinal hundreds (100th, 200th, ... 900th)
# Structure: {hundreds_digit: ((masc_cases), (fem_cases), (neut_cases))}
ORDINAL_HUNDREDS = {
    1: (  # stý (100th)
        ('stý', 'stého', 'stému', 'stého', 'stým', 'stom'),
        ('stá', 'stej', 'stej', 'stú', 'stou', 'stej'),
        ('sté', 'stého', 'stému', 'sté', 'stým', 'stom'),
    ),
    2: (  # dvojstý (200th)
        ('dvojstý', 'dvojstého', 'dvojstému', 'dvojstého', 'dvojstým', 'dvojstom'),
        ('dvojstá', 'dvojstej', 'dvojstej', 'dvojstú', 'dvojstou', 'dvojstej'),
        ('dvojsté', 'dvojstého', 'dvojstému', 'dvojsté', 'dvojstým', 'dvojstom'),
    ),
    3: (  # trojstý (300th)
        ('trojstý', 'trojstého', 'trojstému', 'trojstého', 'trojstým', 'trojstom'),
        ('trojstá', 'trojstej', 'trojstej', 'trojstú', 'trojstou', 'trojstej'),
        ('trojsté', 'trojstého', 'trojstému', 'trojsté', 'trojstým', 'trojstom'),
    ),
    4: (  # štvorsťý (400th)
        ('štvorstý', 'štvorstého', 'štvorstému', 'štvorstého', 'štvorstým', 'štvorstom'),
        ('štvorstá', 'štvorstej', 'štvorstej', 'štvorstú', 'štvorstou', 'štvorstej'),
        ('štvorsté', 'štvorstého', 'štvorstému', 'štvorsté', 'štvorstým', 'štvorstom'),
    ),
    5: (  # päťstý (500th)
        ('päťstý', 'päťstého', 'päťstému', 'päťstého', 'päťstým', 'päťstom'),
        ('päťstá', 'päťstej', 'päťstej', 'päťstú', 'päťstou', 'päťstej'),
        ('päťsté', 'päťstého', 'päťstému', 'päťsté', 'päťstým', 'päťstom'),
    ),
    6: (  # šesťstý (600th)
        ('šesťstý', 'šesťstého', 'šesťstému', 'šesťstého', 'šesťstým', 'šesťstom'),
        ('šesťstá', 'šesťstej', 'šesťstej', 'šesťstú', 'šesťstou', 'šesťstej'),
        ('šesťsté', 'šesťstého', 'šesťstému', 'šesťsté', 'šesťstým', 'šesťstom'),
    ),
    7: (  # sedemstý (700th)
        ('sedemstý', 'sedemstého', 'sedemstému', 'sedemstého', 'sedemstým', 'sedemstom'),
        ('sedemstá', 'sedemstej', 'sedemstej', 'sedemstú', 'sedemstou', 'sedemstej'),
        ('sedemsté', 'sedemstého', 'sedemstému', 'sedemsté', 'sedemstým', 'sedemstom'),
    ),
    8: (  # osemstý (800th)
        ('osemstý', 'osemstého', 'osemstému', 'osemstého', 'osemstým', 'osemstom'),
        ('osemstá', 'osemstej', 'osemstej', 'osemstú', 'osemstou', 'osemstej'),
        ('osemsté', 'osemstého', 'osemstému', 'osemsté', 'osemstým', 'osemstom'),
    ),
    9: (  # deväťstý (900th)
        ('deväťstý', 'deväťstého', 'deväťstému', 'deväťstého', 'deväťstým', 'deväťstom'),
        ('deväťstá', 'deväťstej', 'deväťstej', 'deväťstú', 'deväťstou', 'deväťstej'),
        ('deväťsté', 'deväťstého', 'deväťstému', 'deväťsté', 'deväťstým', 'deväťstom'),
    ),
}

# Ordinal thousands (1000th = tisíci, soft pattern like tretí)
ORDINAL_TISICI = (
    ('tisíci', 'tisíceho', 'tisícemu', 'tisíceho', 'tisícim', 'tisícom'),
    ('tisícia', 'tisícej', 'tisícej', 'tisíciu', 'tisícou', 'tisícej'),
    ('tisície', 'tisíceho', 'tisícemu', 'tisície', 'tisícim', 'tisícom'),
)

# Ordinal prefixes for compound thousands (2000th = dvojtisíci, etc.)
ORDINAL_THOUSAND_PREFIXES = {
    2: 'dvoj',
    3: 'troj',
    4: 'štvor',
    5: 'päť',
    6: 'šesť',
    7: 'sedem',
    8: 'osem',
    9: 'deväť',
}

# Ordinal scale words (milióny = millionth, miliardtý = billionth)
ORDINAL_SCALE = {
    2: (  # milióny (millionth) - hard pattern
        ('milióny', 'milióntého', 'milióntému', 'milióntého', 'milióntým', 'milióntom'),
        ('miliónta', 'milióntej', 'milióntej', 'milióntu', 'milióntou', 'milióntej'),
        ('miliónte', 'milióntého', 'milióntému', 'miliónte', 'milióntým', 'milióntom'),
    ),
    3: (  # miliardtý (billionth)
        ('miliardtý', 'miliardtého', 'miliardtému', 'miliardtého', 'miliardtým', 'miliardtom'),
        ('miliardta', 'miliardtej', 'miliardtej', 'miliardtu', 'miliardtou', 'miliardtej'),
        ('miliardté', 'miliardtého', 'miliardtému', 'miliardté', 'miliardtým', 'miliardtom'),
    ),
    4: (  # bilióny (trillionth)
        ('bilióny', 'bilióntého', 'bilióntému', 'bilióntého', 'bilióntým', 'bilióntom'),
        ('biliónta', 'bilióntej', 'bilióntej', 'bilióntu', 'bilióntou', 'bilióntej'),
        ('biliónte', 'bilióntého', 'bilióntému', 'biliónte', 'bilióntým', 'bilióntom'),
    ),
    5: (  # biliardtý
        ('biliardtý', 'biliardtého', 'biliardtému', 'biliardtého', 'biliardtým', 'biliardtom'),
        ('biliardta', 'biliardtej', 'biliardtej', 'biliardtu', 'biliardtou', 'biliardtej'),
        ('biliardté', 'biliardtého', 'biliardtému', 'biliardté', 'biliardtým', 'biliardtom'),
    ),
    6: (  # trilióny
        ('trilióny', 'trilióntého', 'trilióntému', 'trilióntého', 'trilióntým', 'trilióntom'),
        ('triliónta', 'trilióntej', 'trilióntej', 'trilióntu', 'trilióntou', 'trilióntej'),
        ('triliónte', 'trilióntého', 'trilióntému', 'triliónte', 'trilióntým', 'trilióntom'),
    ),
}

# Ordinal scale prefixes
ORDINAL_SCALE_PREFIXES = {
    2: 'dvoj',
    3: 'troj',
    4: 'štvor',
    5: 'päť',
    6: 'šesť',
    7: 'sedem',
    8: 'osem',
    9: 'deväť',
}


# =============================================================================
# ORDINAL CONVERSION FUNCTIONS
# =============================================================================

def ordinal_ones(n: int, gender_idx: int, case_idx: int) -> str:
    """Convert 1st-9th to ordinal."""
    return ORDINAL_ONES[n][gender_idx][case_idx]


def ordinal_teens(n: int, gender_idx: int, case_idx: int) -> str:
    """Convert 10th-19th to ordinal."""
    return ORDINAL_TEENS[n][gender_idx][case_idx]


def ordinal_tens(t: int, gender_idx: int, case_idx: int) -> str:
    """Convert 20th, 30th, ... 90th to ordinal."""
    return ORDINAL_TENS[t][gender_idx][case_idx]


def ordinal_hundreds(h: int, gender_idx: int, case_idx: int) -> str:
    """Convert 100th, 200th, ... 900th to ordinal."""
    return ORDINAL_HUNDREDS[h][gender_idx][case_idx]


def ordinal_0_99(n: int, gender_idx: int, case_idx: int) -> str:
    """Convert 1st-99th to ordinal. Compound form (no spaces)."""
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
        # Compound: tens stay nominative masculine, ones take full declension
        # e.g., dvadsiatyprvý, dvadsiatehopiateho
        tens_word = ORDINAL_TENS[tens_digit][gender_idx][case_idx]
        ones_word = ordinal_ones(ones_digit, gender_idx, case_idx)
        return tens_word + ones_word


def ordinal_0_999(n: int, gender_idx: int, case_idx: int) -> str:
    """Convert 1st-999th to ordinal. Compound form."""
    if n == 0:
        return ''
    
    hundreds = n // 100
    remainder = n % 100
    
    if hundreds == 0:
        return ordinal_0_99(remainder, gender_idx, case_idx)
    
    if remainder == 0:
        return ordinal_hundreds(hundreds, gender_idx, case_idx)
    else:
        # Compound: hundreds stay nominative, remainder takes declension
        hundreds_word = ORDINAL_HUNDREDS[hundreds][gender_idx][0]  # nominative
        remainder_word = ordinal_0_99(remainder, gender_idx, case_idx)
        return hundreds_word + remainder_word


def ordinal_thousands(n: int, gender_idx: int, case_idx: int, has_remainder: bool = False) -> str:
    """
    Convert 1000th-999000th to ordinal.
    1000th = tisíci
    2000th = dvojtisíci
    10000th = desaťtisíci
    """
    if n == 0:
        return ''
    
    if n == 1:
        if has_remainder:
            return ORDINAL_TISICI[gender_idx][0]  # nominative when followed by remainder
        return ORDINAL_TISICI[gender_idx][case_idx]
    
    # Build prefix
    if n < 10:
        prefix = ORDINAL_THOUSAND_PREFIXES.get(n, '')
        if not prefix:
            prefix = convert_0_999(n, 0, 0)  # cardinal nominative
    else:
        prefix = convert_0_999(n, 0, 0)  # cardinal nominative
    
    if has_remainder:
        suffix = ORDINAL_TISICI[gender_idx][0]
    else:
        suffix = ORDINAL_TISICI[gender_idx][case_idx]
    
    return prefix + suffix


def ordinal_below_million(n: int, gender_idx: int, case_idx: int) -> str:
    """Convert 1st-999999th to ordinal."""
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
    """
    Convert scale ordinal (millionth, billionth, etc.)
    count: how many of this scale (e.g., 5 for 5 millionth)
    """
    scale_ordinal = ORDINAL_SCALE[scale_idx]
    
    if count == 1:
        if has_remainder:
            return scale_ordinal[gender_idx][0]
        return scale_ordinal[gender_idx][case_idx]
    
    # Build prefix
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
    """
    Convert integer to Slovak ordinal number with full gender+case declension.
    
    Args:
        n: Integer to convert (must be > 0)
        gender: 'masculine', 'feminine', or 'neuter'
        case: One of 'nominative', 'genitive', 'dative', 'accusative', 'instrumental', 'locative'
    
    Returns:
        Slovak ordinal word representation with proper declension
    """
    if n <= 0:
        raise ValueError("Ordinal numbers must be positive integers")
    
    gender_idx = get_gender_index(gender)
    case_idx = get_case_index(case)
    
    if n < 1_000_000:
        return ordinal_below_million(n, gender_idx, case_idx)
    
    # For millions and above, build compound
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
    """
    Convert integer to Slovak cardinal number with full gender+case declension.
    Uses compound forms throughout (no spaces between number parts).
    
    Args:
        n: Integer to convert
        gender: 'masculine', 'feminine', or 'neuter'
        case: One of 'nominative', 'genitive', 'dative', 'accusative', 'instrumental', 'locative'
    
    Returns:
        Slovak word representation with proper declension
    """
    if n < 0:
        return MINUS + ' ' + int_to_cardinal(abs(n), gender, case)
    
    if n == 0:
        return ZERO
    
    return int_to_cardinal_compound(n, gender, case)


def float_to_cardinal(n: float, gender: str = 'masculine', case: str = 'nominative') -> str:
    """Convert float to Slovak words."""
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

class Num2Word_SK:
    """Slovak number to words converter."""
    
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
    """
    Convert number to Slovak words.
    
    Args:
        number: Number to convert
        to: 'cardinal' or 'ordinal'
        gender: 'masculine', 'feminine', or 'neuter' (default: 'masculine')
        case: 'nominative', 'genitive', 'dative', 'accusative', 'instrumental', 'locative'
    
    Returns:
        Slovak word representation
    """
    converter = Num2Word_SK()
    if to == 'ordinal':
        return converter.to_ordinal(number, **kwargs)
    return converter.to_cardinal(number, **kwargs)


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("SLOVAK NUM2WORDS - Full Gender + Case Declension Test")
    print("=" * 80)
    
    # Test key numbers with gender and case
    test_cases = [
        (1, "jeden/jedna/jedno"),
        (2, "dva/dve/dve"),
        (100, "sto/stá/sté"),
        (101, "stojedna (fem gen = stojednej)"),
        (200, "dvesto/dvestá/dvesté"),
    ]
    
    print("\n--- GENDER comparison (nominative) ---")
    for n, desc in test_cases:
        print(f"\n{n} ({desc}):")
        for gender in ['masculine', 'feminine', 'neuter']:
            result = num2words(n, gender=gender, case='nominative')
            print(f"  {gender}: {result}")
    
    print("\n" + "=" * 80)
    print("--- 100 in all cases, all genders ---")
    print("=" * 80)
    for gender in ['masculine', 'feminine', 'neuter']:
        print(f"\n{gender.upper()}:")
        for case in CASES:
            result = num2words(100, gender=gender, case=case)
            print(f"  {case}: {result}")
    
    print("\n" + "=" * 80)
    print("--- 101 in all cases, feminine gender ---")
    print("(Should show: stojedna, stojednej, stojednej, stojednu, stojednou, stojednej)")
    print("=" * 80)
    for case in CASES:
        result = num2words(101, gender='feminine', case=case)
        print(f"  {case}: {result}")
    
    print("\n" + "=" * 80)
    print("--- 121 in all cases, feminine gender ---")
    print("=" * 80)
    for case in CASES:
        result = num2words(121, gender='feminine', case=case)
        print(f"  {case}: {result}")
