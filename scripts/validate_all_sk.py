#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive validation: All ORIGINAL_TEST_NUMBERS × all genders × all cases
For both CARDINAL and ORDINAL numbers.
This file prints ALL results for human review.
"""

import sys
if 'num2words_sk' in sys.modules:
    del sys.modules['num2words_sk']

from num2words_sk import num2words

# Complete list from Jupyter notebook - 170 numbers
ORIGINAL_TEST_NUMBERS = [
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
    # Extended to decillions
    10000000000, 50000000000, 99999999999,
    100000000000, 500000000000, 999999999999,
    1000000000000, 5000000000000, 9999999999999,
    10000000000000, 50000000000000, 99999999999999,
    100000000000000, 500000000000000, 999999999999999,
    1000000000000000, 5000000000000000, 9999999999999999,
    10000000000000000, 50000000000000000, 99999999999999999,
    100000000000000000, 500000000000000000, 999999999999999999,
    1000000000000000000, 5000000000000000000, 9999999999999999999,
]

# Ordinal test numbers (subset for ordinals)
ORDINAL_TEST_NUMBERS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 30, 33, 40, 44, 50, 55, 60, 70, 80, 90, 99,
    100, 101, 111, 121, 200, 222, 300, 333, 400, 444, 500, 600, 700, 800, 900,
    1000, 1001, 1111, 2000, 2222, 3000, 5000, 10000, 100000,
    1000000, 1000001, 2000000, 1000000000, 2000000000,
]

GENDERS = ['masculine', 'feminine', 'neuter']
CASES = ['nominative', 'genitive', 'dative', 'accusative', 'instrumental', 'locative']
CASE_ABBR = {'nominative': 'nom', 'genitive': 'gen', 'dative': 'dat', 
             'accusative': 'acc', 'instrumental': 'ins', 'locative': 'loc'}

def print_header(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

def validate_cardinal(n):
    """Print all gender × case combinations for a cardinal number."""
    print(f"\n{n:,}:")
    print("-" * 90)
    
    errors = []
    
    for gender in GENDERS:
        results = []
        for case in CASES:
            try:
                result = num2words(n, to='cardinal', gender=gender, case=case)
                results.append(f"{CASE_ABBR[case]}:{result}")
            except Exception as e:
                results.append(f"{CASE_ABBR[case]}:ERROR({e})")
                errors.append((gender, case, str(e)))
        
        print(f"  {gender[:3]}: {' | '.join(results)}")
    
    return errors

def validate_ordinal(n):
    """Print all gender × case combinations for an ordinal number."""
    print(f"\n{n:,}. (ordinal):")
    print("-" * 90)
    
    errors = []
    
    for gender in GENDERS:
        results = []
        for case in CASES:
            try:
                result = num2words(n, to='ordinal', gender=gender, case=case)
                results.append(f"{CASE_ABBR[case]}:{result}")
            except Exception as e:
                results.append(f"{CASE_ABBR[case]}:ERROR({e})")
                errors.append((gender, case, str(e)))
        
        print(f"  {gender[:3]}: {' | '.join(results)}")
    
    return errors

def main():
    print_header("SLOVAK NUM2WORDS - COMPREHENSIVE VALIDATION")
    print(f"Testing CARDINALS: {len(ORIGINAL_TEST_NUMBERS)} numbers")
    print(f"Testing ORDINALS: {len(ORDINAL_TEST_NUMBERS)} numbers")
    print("Each number tested across 3 genders × 6 cases = 18 combinations")
    
    all_errors = []
    
    # Test cardinal numbers
    print_header("CARDINAL NUMBERS")
    for n in ORIGINAL_TEST_NUMBERS:
        errors = validate_cardinal(n)
        all_errors.extend([(n, 'cardinal', *e) for e in errors])
    
    # Test ordinal numbers
    print_header("ORDINAL NUMBERS")
    for n in ORDINAL_TEST_NUMBERS:
        errors = validate_ordinal(n)
        all_errors.extend([(n, 'ordinal', *e) for e in errors])
    
    # Summary
    print_header("SUMMARY")
    
    cardinal_tests = len(ORIGINAL_TEST_NUMBERS) * len(GENDERS) * len(CASES)
    ordinal_tests = len(ORDINAL_TEST_NUMBERS) * len(GENDERS) * len(CASES)
    total_tests = cardinal_tests + ordinal_tests
    
    print(f"Cardinal numbers tested: {len(ORIGINAL_TEST_NUMBERS)}")
    print(f"Cardinal test combinations: {cardinal_tests}")
    print(f"Ordinal numbers tested: {len(ORDINAL_TEST_NUMBERS)}")
    print(f"Ordinal test combinations: {ordinal_tests}")
    print(f"Total test combinations: {total_tests}")
    print(f"Errors encountered: {len(all_errors)}")
    
    if all_errors:
        print("\nErrors:")
        for n, num_type, gender, case, err in all_errors[:20]:
            print(f"  {n} ({num_type}) {gender} {case}: {err}")
    else:
        print("\n✓ All tests completed without exceptions!")

if __name__ == '__main__':
    main()
