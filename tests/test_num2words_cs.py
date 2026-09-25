#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test suite for Czech num2words implementation.
Tests cardinal and ordinal numbers with full gender × case declension.
"""

import sys
if 'num2words_cs' in sys.modules:
    del sys.modules['num2words_cs']

from num2words_cs import num2words

# =============================================================================
# EXPECTED VALUES - CARDINALS
# =============================================================================

EXPECTED_1 = {
    'masculine': {
        'nominative': 'jeden',
        'genitive': 'jednoho',
        'dative': 'jednomu',
        'accusative': 'jednoho',
        'instrumental': 'jedním',
        'locative': 'jednom',
    },
    'feminine': {
        'nominative': 'jedna',
        'genitive': 'jedné',
        'dative': 'jedné',
        'accusative': 'jednu',
        'instrumental': 'jednou',
        'locative': 'jedné',
    },
    'neuter': {
        'nominative': 'jedno',
        'genitive': 'jednoho',
        'dative': 'jednomu',
        'accusative': 'jedno',
        'instrumental': 'jedním',
        'locative': 'jednom',
    },
}

EXPECTED_2 = {
    'masculine': {
        'nominative': 'dva',
        'genitive': 'dvou',
        'dative': 'dvěma',
        'accusative': 'dva',
        'instrumental': 'dvěma',
        'locative': 'dvou',
    },
    'feminine': {
        'nominative': 'dvě',
        'genitive': 'dvou',
        'dative': 'dvěma',
        'accusative': 'dvě',
        'instrumental': 'dvěma',
        'locative': 'dvou',
    },
    'neuter': {
        'nominative': 'dvě',
        'genitive': 'dvou',
        'dative': 'dvěma',
        'accusative': 'dvě',
        'instrumental': 'dvěma',
        'locative': 'dvou',
    },
}

EXPECTED_THOUSANDS = {
    1000: 'tisíc',
    2000: 'dvoutisíc',
    3000: 'třítisíc',
    5000: 'pětitisíc',
}

EXPECTED_HUNDREDS = {
    100: 'sto',
    200: 'dvěstě',
    300: 'třista',
    400: 'čtyřista',
    500: 'pětset',
    600: 'šestset',
    700: 'sedmset',
    800: 'osmset',
    900: 'devětset',
}

# =============================================================================
# EXPECTED VALUES - ORDINALS
# =============================================================================

EXPECTED_ORDINAL_1 = {
    'masculine': {
        'nominative': 'první',
        'genitive': 'prvního',
        'dative': 'prvnímu',
        'accusative': 'prvního',
        'instrumental': 'prvním',
        'locative': 'prvním',
    },
    'feminine': {
        'nominative': 'první',
        'genitive': 'první',
        'dative': 'první',
        'accusative': 'první',
        'instrumental': 'první',
        'locative': 'první',
    },
    'neuter': {
        'nominative': 'první',
        'genitive': 'prvního',
        'dative': 'prvnímu',
        'accusative': 'první',
        'instrumental': 'prvním',
        'locative': 'prvním',
    },
}

EXPECTED_ORDINAL_2 = {
    'masculine': {
        'nominative': 'druhý',
        'genitive': 'druhého',
        'dative': 'druhému',
        'accusative': 'druhého',
        'instrumental': 'druhým',
        'locative': 'druhém',
    },
    'feminine': {
        'nominative': 'druhá',
        'genitive': 'druhé',
        'dative': 'druhé',
        'accusative': 'druhou',
        'instrumental': 'druhou',
        'locative': 'druhé',
    },
    'neuter': {
        'nominative': 'druhé',
        'genitive': 'druhého',
        'dative': 'druhému',
        'accusative': 'druhé',
        'instrumental': 'druhým',
        'locative': 'druhém',
    },
}

EXPECTED_ORDINALS_NOM = {
    1: ('první', 'první', 'první'),
    2: ('druhý', 'druhá', 'druhé'),
    3: ('třetí', 'třetí', 'třetí'),
    4: ('čtvrtý', 'čtvrtá', 'čtvrté'),
    5: ('pátý', 'pátá', 'páté'),
    6: ('šestý', 'šestá', 'šesté'),
    7: ('sedmý', 'sedmá', 'sedmé'),
    8: ('osmý', 'osmá', 'osmé'),
    9: ('devátý', 'devátá', 'deváté'),
    10: ('desátý', 'desátá', 'desáté'),
    11: ('jedenáctý', 'jedenáctá', 'jedenácté'),
    12: ('dvanáctý', 'dvanáctá', 'dvanácté'),
    20: ('dvacátý', 'dvacátá', 'dvacáté'),
    21: ('dvacátýprvní', 'dvacátáprvní', 'dvacátéprvní'),
    100: ('stý', 'stá', 'sté'),
    1000: ('tisící', 'tisící', 'tisící'),
    1000000: ('miliontý', 'miliontá', 'milionté'),
}


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_number_gender_case(n, expected_dict, test_name, to='cardinal'):
    """Test a number across all specified genders and cases."""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")
    
    passed = 0
    failed = 0
    
    for gender, cases_dict in expected_dict.items():
        for case, expected in cases_dict.items():
            result = num2words(n, to=to, gender=gender, case=case)
            if result == expected:
                print(f"  ✓ {gender} {case}: {result}")
                passed += 1
            else:
                print(f"  ✗ {gender} {case}: got '{result}', expected '{expected}'")
                failed += 1
    
    return passed, failed


def test_nominative_values(expected_dict, test_name, to='cardinal'):
    """Test nominative values."""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")
    
    passed = 0
    failed = 0
    
    for n, expected in expected_dict.items():
        result = num2words(n, to=to, gender='masculine', case='nominative')
        if result == expected:
            print(f"  ✓ {n}: {result}")
            passed += 1
        else:
            print(f"  ✗ {n}: got '{result}', expected '{expected}'")
            failed += 1
    
    return passed, failed


def test_ordinal_nominative():
    """Test ordinal nominative forms across genders."""
    print(f"\n{'='*60}")
    print(f"TEST: Ordinal Numbers - Nominative (all genders)")
    print(f"{'='*60}")
    
    passed = 0
    failed = 0
    genders = ['masculine', 'feminine', 'neuter']
    
    for n, expected_tuple in EXPECTED_ORDINALS_NOM.items():
        all_ok = True
        results = []
        for i, gender in enumerate(genders):
            result = num2words(n, to='ordinal', gender=gender, case='nominative')
            results.append(result)
            if result != expected_tuple[i]:
                all_ok = False
        
        if all_ok:
            print(f"  ✓ {n}: {'/'.join(results)}")
            passed += 1
        else:
            print(f"  ✗ {n}: got {'/'.join(results)}, expected {'/'.join(expected_tuple)}")
            failed += 1
    
    return passed, failed


def test_basic_cardinals():
    """Test basic cardinal numbers."""
    print(f"\n{'='*60}")
    print(f"TEST: Basic Cardinal Numbers (neuter nominative)")
    print(f"{'='*60}")
    
    expected = {
        1: 'jedno',
        2: 'dvě',
        3: 'tři',
        4: 'čtyři',
        5: 'pět',
        6: 'šest',
        7: 'sedm',
        8: 'osm',
        9: 'devět',
        10: 'deset',
        11: 'jedenáct',
        12: 'dvanáct',
        20: 'dvacet',
        21: 'dvacetjedno',
        100: 'sté',
        200: 'dvěstě',
        1000: 'tisíc',
        1000000: 'milion',
        1000000000: 'miliarda',
    }
    
    passed = 0
    failed = 0
    
    for n, exp in expected.items():
        result = num2words(n, gender='neuter', case='nominative')
        if result == exp:
            print(f"  ✓ {n}: {result}")
            passed += 1
        else:
            print(f"  ✗ {n}: got '{result}', expected '{exp}'")
            failed += 1
    
    return passed, failed


def main():
    """Run all tests."""
    print("=" * 70)
    print("CZECH NUM2WORDS - COMPREHENSIVE TEST SUITE")
    print("Testing gender and case declension")
    print("=" * 70)
    
    total_passed = 0
    total_failed = 0
    
    # Test 1: Number 1 full declension
    p, f = test_number_gender_case(1, EXPECTED_1, "Cardinal 1 - Full declension")
    total_passed += p
    total_failed += f
    
    # Test 2: Number 2 full declension
    p, f = test_number_gender_case(2, EXPECTED_2, "Cardinal 2 - Full declension")
    total_passed += p
    total_failed += f
    
    # Test 3: Hundreds
    p, f = test_nominative_values(EXPECTED_HUNDREDS, "Hundreds - Nominative")
    total_passed += p
    total_failed += f
    
    # Test 4: Compound thousands
    p, f = test_nominative_values(EXPECTED_THOUSANDS, "Compound thousands - Nominative")
    total_passed += p
    total_failed += f
    
    # Test 5: Basic cardinals
    p, f = test_basic_cardinals()
    total_passed += p
    total_failed += f
    
    # Test 6: Ordinal 1st full declension
    p, f = test_number_gender_case(1, EXPECTED_ORDINAL_1, "Ordinal 1st - Full declension", to='ordinal')
    total_passed += p
    total_failed += f
    
    # Test 7: Ordinal 2nd full declension
    p, f = test_number_gender_case(2, EXPECTED_ORDINAL_2, "Ordinal 2nd - Full declension", to='ordinal')
    total_passed += p
    total_failed += f
    
    # Test 8: Ordinal nominative
    p, f = test_ordinal_nominative()
    total_passed += p
    total_failed += f
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"  Passed: {total_passed}")
    print(f"  Failed: {total_failed}")
    print(f"  Total:  {total_passed + total_failed}")
    
    if total_failed == 0:
        print("\n  ✓ ALL TESTS PASSED!")
    else:
        print(f"\n  ✗ {total_failed} TESTS FAILED")
    
    return total_failed == 0


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
