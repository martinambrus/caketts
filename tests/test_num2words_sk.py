#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Test Suite for Slovak num2words Implementation
Tests gender and case declension across various number types.
"""

from num2words_sk import num2words, CASES

# =============================================================================
# TEST DATA - Expected values from Slovak grammar
# =============================================================================

# Test: 100 in all genders and cases
EXPECTED_100 = {
    'masculine': {
        'nominative': 'sto',
        'genitive': 'stého',
        'dative': 'stému',
        'accusative': 'sto',
        'instrumental': 'stým',
        'locative': 'stom',
    },
    'feminine': {
        'nominative': 'stá',
        'genitive': 'stej',
        'dative': 'stej',
        'accusative': 'stú',
        'instrumental': 'stou',
        'locative': 'stej',
    },
    'neuter': {
        'nominative': 'sté',
        'genitive': 'stého',
        'dative': 'stému',
        'accusative': 'sté',
        'instrumental': 'stým',
        'locative': 'stom',
    },
}

# Test: 1 in all genders and cases
EXPECTED_1 = {
    'masculine': {
        'nominative': 'jeden',
        'genitive': 'jedného',
        'dative': 'jednému',
        'accusative': 'jedného',
        'instrumental': 'jedným',
        'locative': 'jednom',
    },
    'feminine': {
        'nominative': 'jedna',
        'genitive': 'jednej',
        'dative': 'jednej',
        'accusative': 'jednu',
        'instrumental': 'jednou',
        'locative': 'jednej',
    },
    'neuter': {
        'nominative': 'jedno',
        'genitive': 'jedného',
        'dative': 'jednému',
        'accusative': 'jedno',
        'instrumental': 'jedným',
        'locative': 'jednom',
    },
}

# Test: 2 in all genders and cases
EXPECTED_2 = {
    'masculine': {
        'nominative': 'dva',
        'genitive': 'dvoch',
        'dative': 'dvom',
        'accusative': 'dva',
        'instrumental': 'dvoma',
        'locative': 'dvoch',
    },
    'feminine': {
        'nominative': 'dve',
        'genitive': 'dvoch',
        'dative': 'dvom',
        'accusative': 'dve',
        'instrumental': 'dvomi',
        'locative': 'dvoch',
    },
    'neuter': {
        'nominative': 'dve',
        'genitive': 'dvoch',
        'dative': 'dvom',
        'accusative': 'dve',
        'instrumental': 'dvoma',
        'locative': 'dvoch',
    },
}

# Test: 101 feminine (sto + jedna compound)
# Note: hundreds stay NOMINATIVE when followed by other digits
EXPECTED_101_FEM = {
    'nominative': 'stájedna',
    'genitive': 'stájednej',
    'dative': 'stájednej',
    'accusative': 'stájednu',
    'instrumental': 'stájednou',
    'locative': 'stájednej',
}

# Test: 200 in genders
EXPECTED_200 = {
    'masculine': {
        'nominative': 'dvesto',
        'genitive': 'dvestého',
    },
    'feminine': {
        'nominative': 'dvestá',
        'genitive': 'dvestej',
    },
    'neuter': {
        'nominative': 'dvesté',
        'genitive': 'dvestého',
    },
}

# Test: Compound thousands (indeclinable)
EXPECTED_THOUSANDS = {
    1000: 'tisíc',
    2000: 'dvetisíc',
    3000: 'tritisíc',
    5000: 'päťtisíc',
}

# Test: Millions with compound forms and t-suffix
EXPECTED_MILLIONS = {
    1000000: {
        'nominative': 'milión',
        'genitive': 'miliónteho',
    },
    2000000: {
        'nominative': 'dvamilióny',
        'genitive': 'dvamiliónteho',
    },
}

# Test: Billions (compound forms with feminine prefix agreement)
EXPECTED_BILLIONS = {
    1000000000: {
        'nominative': 'miliarda',
    },
    2000000000: {
        'nominative': 'dvemiliardy',
        'genitive': 'dvemiliardtého',
    },
}


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_number_gender_case(n, expected_dict, test_name):
    """Test a number across all specified genders and cases."""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")
    
    passed = 0
    failed = 0
    
    for gender, cases_dict in expected_dict.items():
        for case, expected in cases_dict.items():
            result = num2words(n, gender=gender, case=case)
            status = "✓" if result == expected else "✗"
            if result == expected:
                passed += 1
            else:
                failed += 1
            
            if result != expected:
                print(f"  {status} {gender} {case}: got '{result}', expected '{expected}'")
            else:
                print(f"  {status} {gender} {case}: {result}")
    
    return passed, failed


def test_single_case(n, expected_dict, gender, test_name):
    """Test a number with specific gender across all cases."""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")
    
    passed = 0
    failed = 0
    
    for case, expected in expected_dict.items():
        result = num2words(n, gender=gender, case=case)
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        else:
            failed += 1
        
        if result != expected:
            print(f"  {status} {case}: got '{result}', expected '{expected}'")
        else:
            print(f"  {status} {case}: {result}")
    
    return passed, failed


def test_nominative_values(expected_dict, test_name):
    """Test nominative forms for various numbers."""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")
    
    passed = 0
    failed = 0
    
    for n, expected in expected_dict.items():
        result = num2words(n, gender='masculine', case='nominative')
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        else:
            failed += 1
        
        if result != expected:
            print(f"  {status} {n}: got '{result}', expected '{expected}'")
        else:
            print(f"  {status} {n}: {result}")
    
    return passed, failed


def test_scales_with_cases(expected_dict, test_name):
    """Test scale numbers (millions, billions) with case declension."""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")
    
    passed = 0
    failed = 0
    
    for n, cases_dict in expected_dict.items():
        print(f"  {n}:")
        for case, expected in cases_dict.items():
            result = num2words(n, gender='masculine', case=case)
            status = "✓" if result == expected else "✗"
            if result == expected:
                passed += 1
            else:
                failed += 1
            
            if result != expected:
                print(f"    {status} {case}: got '{result}', expected '{expected}'")
            else:
                print(f"    {status} {case}: {result}")
    
    return passed, failed


# =============================================================================
# ORIGINAL TEST NUMBERS FROM JUPYTER NOTEBOOK
# These numbers MUST be preserved in all test suites
# =============================================================================

ORIGINAL_TEST_NUMBERS = [
    # Existing numbers for thoroughness at lower scales
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15,
    20, 21, 29, 30, 40, 45, 50, 55, 88, 90, 99,
    100, 101, 112, 120, 121, 150, 151, 195,
    200, 201, 202, 213, 224,
    300, 303, 313, 333,
    400, 404, 414, 444,
    1000, 1001, 1010, 1011, 1025, 1050, 1098, 1100, 1101, 1111,
    1590, 1900, 1990, 1991, 2000, 2001, 2004, 2012, 2020, 2021, 2059, 2099,
    2100, 2152, 2200, 2201, 2222,
    3000, 3003, 3013, 3033, 3333,
    4000, 4004, 4014, 4044, 4444,
    10000, 10001, 10010, 10011, 10021, 10033,
    20000, 20001, 25000, 29999,
    100000, 100001, 100010, 100100, 100101,
    200000, 250000, 299999,
    1000000, 1000001, 1000010, 1000100, 1001000, 1001010, 1100000,
    2000000, 2500000, 2999999,
    10000000, 15000000, 19999999,
    100000000, 150000000, 199999999,
    1000000000, 1500000000, 1999999999,
    2000000000, 2500000000, 2999999999,
    
    # Expanding to larger scales: billions (miliardy)
    10000000000, 20000000000, 25000000000, 99999999999,
    
    # Trillions (bilióny)
    100000000000, 150000000000, 200000000000, 500000000000, 999999999999,
    
    # Quadrillions (biliardy)
    1000000000000, 2000000000000, 3000000000000, 5000000000000, 9999999999999,
    
    # Quintillions (trilióny)
    10000000000000, 20000000000000, 30000000000000, 50000000000000, 99999999999999,
    
    # Sextillions (triliardy)
    100000000000000, 200000000000000, 300000000000000, 500000000000000, 999999999999999,
    
    # Septillions (kvadrilióny)
    1000000000000000, 2000000000000000, 3000000000000000, 5000000000000000, 9999999999999999,
    
    # Octillions (kvadriliardy)
    10000000000000000, 20000000000000000, 30000000000000000, 50000000000000000, 99999999999999999,
    
    # Nonillions (kvintilióny)
    100000000000000000, 200000000000000000, 300000000000000000, 500000000000000000, 999999999999999999,
    
    # Decillions (decilióny)
    1000000000000000000, 2000000000000000000, 3000000000000000000, 5000000000000000000, 9999999999999999999
]


def test_original_numbers():
    """Test all original numbers from the Jupyter notebook in nominative neuter case."""
    print(f"\n{'='*60}")
    print("TEST: Original Jupyter Notebook Numbers (nominative, neuter)")
    print(f"{'='*60}")
    
    passed = 0
    failed = 0
    errors = []
    
    for number in ORIGINAL_TEST_NUMBERS:
        try:
            result = num2words(number, gender='neuter', case='nominative')
            passed += 1
            # Only print first few and any notable ones
            if number <= 10 or number in [100, 1000, 1000000, 1000000000]:
                print(f"  ✓ {number}: {result}")
        except Exception as e:
            failed += 1
            errors.append((number, str(e)))
            print(f"  ✗ {number}: ERROR - {e}")
    
    if passed > 10:
        print(f"  ... ({passed - 10} more numbers tested successfully)")
    
    if errors:
        print(f"\n  Errors encountered:")
        for num, err in errors[:5]:  # Show first 5 errors
            print(f"    {num}: {err}")
    
    return passed, failed


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def main():
    print("=" * 70)
    print("SLOVAK NUM2WORDS - COMPREHENSIVE TEST SUITE")
    print("Testing gender and case declension")
    print("=" * 70)
    
    total_passed = 0
    total_failed = 0
    
    # Test 1: Number 1 in all genders and cases
    p, f = test_number_gender_case(1, EXPECTED_1, "Number 1 - Full declension")
    total_passed += p
    total_failed += f
    
    # Test 2: Number 2 in all genders and cases
    p, f = test_number_gender_case(2, EXPECTED_2, "Number 2 - Full declension")
    total_passed += p
    total_failed += f
    
    # Test 3: Number 100 in all genders and cases
    p, f = test_number_gender_case(100, EXPECTED_100, "Number 100 - Full declension")
    total_passed += p
    total_failed += f
    
    # Test 4: Number 101 in feminine gender
    p, f = test_single_case(101, EXPECTED_101_FEM, 'feminine', "Number 101 feminine - Full declension")
    total_passed += p
    total_failed += f
    
    # Test 5: Number 200 partial test
    p, f = test_number_gender_case(200, EXPECTED_200, "Number 200 - Partial declension")
    total_passed += p
    total_failed += f
    
    # Test 6: Compound thousands (nominative)
    p, f = test_nominative_values(EXPECTED_THOUSANDS, "Compound thousands - Nominative")
    total_passed += p
    total_failed += f
    
    # Test 7: Millions with cases
    p, f = test_scales_with_cases(EXPECTED_MILLIONS, "Millions - Case declension")
    total_passed += p
    total_failed += f
    
    # Test 8: Billions with cases
    p, f = test_scales_with_cases(EXPECTED_BILLIONS, "Billions - Case declension")
    total_passed += p
    total_failed += f
    
    # Test 9: Original Jupyter Notebook numbers
    p, f = test_original_numbers()
    total_passed += p
    total_failed += f
    
    # Test 10: Ordinal numbers
    p, f = test_ordinals()
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


# =============================================================================
# ORDINAL TEST DATA AND FUNCTIONS
# =============================================================================

# Expected ordinal values: {number: {gender: {case: expected_word}}}
EXPECTED_ORDINALS_1 = {
    'masculine': {
        'nominative': 'prvý',
        'genitive': 'prvého',
        'dative': 'prvému',
        'accusative': 'prvého',
        'instrumental': 'prvým',
        'locative': 'prvom',
    },
    'feminine': {
        'nominative': 'prvá',
        'genitive': 'prvej',
        'dative': 'prvej',
        'accusative': 'prvú',
        'instrumental': 'prvou',
        'locative': 'prvej',
    },
    'neuter': {
        'nominative': 'prvé',
        'genitive': 'prvého',
        'dative': 'prvému',
        'accusative': 'prvé',
        'instrumental': 'prvým',
        'locative': 'prvom',
    },
}

EXPECTED_ORDINALS_3 = {
    'masculine': {
        'nominative': 'tretí',
        'genitive': 'tretieho',
        'dative': 'tretiemu',
        'accusative': 'tretieho',
        'instrumental': 'tretím',
        'locative': 'treťom',
    },
    'feminine': {
        'nominative': 'tretia',
        'genitive': 'tretej',
        'dative': 'tretej',
        'accusative': 'tretiu',
        'instrumental': 'treťou',
        'locative': 'tretej',
    },
    'neuter': {
        'nominative': 'tretie',
        'genitive': 'tretieho',
        'dative': 'tretiemu',
        'accusative': 'tretie',
        'instrumental': 'tretím',
        'locative': 'treťom',
    },
}

# Simple ordinal nominative tests
EXPECTED_ORDINALS_NOM = {
    1: ('prvý', 'prvá', 'prvé'),
    2: ('druhý', 'druhá', 'druhé'),
    3: ('tretí', 'tretia', 'tretie'),
    4: ('štvrtý', 'štvrtá', 'štvrté'),
    5: ('piaty', 'piata', 'piate'),
    6: ('šiesty', 'šiesta', 'šieste'),
    7: ('siedmy', 'siedma', 'siedme'),
    8: ('ôsmy', 'ôsma', 'ôsme'),
    9: ('deviaty', 'deviata', 'deviate'),
    10: ('desiaty', 'desiata', 'desiate'),
    11: ('jedenásty', 'jedenásta', 'jedenáste'),
    12: ('dvanásty', 'dvanásta', 'dvanáste'),
    20: ('dvadsiaty', 'dvadsiata', 'dvadsiate'),
    21: ('dvadsiatyprvý', 'dvadsiataprvá', 'dvadsiateprvé'),
    100: ('stý', 'stá', 'sté'),
    1000: ('tisíci', 'tisícia', 'tisície'),
    1000000: ('milióny', 'miliónta', 'miliónte'),
}


def test_ordinal_full_declension(n, expected_dict, test_name):
    """Test ordinal number across all genders and cases."""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")
    
    passed = 0
    failed = 0
    
    for gender, cases_dict in expected_dict.items():
        for case, expected in cases_dict.items():
            result = num2words(n, to='ordinal', gender=gender, case=case)
            if result == expected:
                print(f"  ✓ {gender} {case}: {result}")
                passed += 1
            else:
                print(f"  ✗ {gender} {case}: got '{result}', expected '{expected}'")
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


def test_ordinals():
    """Run all ordinal tests."""
    print(f"\n{'='*60}")
    print("ORDINAL NUMBER TESTS")
    print(f"{'='*60}")
    
    total_passed = 0
    total_failed = 0
    
    # Test 1: First (1st) full declension
    p, f = test_ordinal_full_declension(1, EXPECTED_ORDINALS_1, "Ordinal 1st - Full declension")
    total_passed += p
    total_failed += f
    
    # Test 2: Third (3rd) full declension (soft pattern)
    p, f = test_ordinal_full_declension(3, EXPECTED_ORDINALS_3, "Ordinal 3rd - Full declension (soft)")
    total_passed += p
    total_failed += f
    
    # Test 3: Various ordinals nominative
    p, f = test_ordinal_nominative()
    total_passed += p
    total_failed += f
    
    return total_passed, total_failed


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
