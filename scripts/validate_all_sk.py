#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive validation for num2words_sk v2: all ORIGINAL_TEST_NUMBERS x all genders x all cases,
for CARDINAL and ORDINAL numbers, plus decimals. Prints everything for human review.

Besides the three gender rows, extra rows are printed only where they differ from the default:
  m.anim   masculine animate (jedného; prvého)
  m.pers   masculine personal, congruent type (dvaja, traja; dvadsiati piati; A = G)
  undecl.  the undeclined variant for compounds 22-99 (dvadsaťdva in every case)
  pl ...   ordinal plurals (prví, piati; prvé)
  codified the codified dvojstý / dvojtisíci instead of the default dvestý / dvetisíci
"""

try:  # standalone: the module sits next to this script
    from num2words_sk import num2words
except ImportError:  # inside the TTS repository: scripts/ -> src/text/
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.text.num2words_sk import num2words

# Complete list from the Jupyter notebook
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
    # Extended to quintillions
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

ORDINAL_TEST_NUMBERS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 30, 33, 40, 44, 50, 55, 60, 70, 80, 90, 99,
    100, 101, 111, 121, 200, 222, 300, 333, 400, 444, 500, 600, 700, 800, 900,
    1000, 1001, 1111, 2000, 2222, 3000, 5000, 10000, 100000,
    1000000, 1000001, 2000000, 1000000000, 2000000000,
]

DECIMAL_TEST_NUMBERS = ["0,25", "0,5", "1,5", "1,9", "2,5", "2,84", "3,14", "5,25", "21,3", "22,5",
                        "41,2", "68,50", "396,4", "2,531", "1000,05", "-2,5"]

GENDERS = ['masculine', 'feminine', 'neuter']
CASES = ['nominative', 'genitive', 'dative', 'accusative', 'instrumental', 'locative']
CASE_ABBR = {'nominative': 'nom', 'genitive': 'gen', 'dative': 'dat',
             'accusative': 'acc', 'instrumental': 'ins', 'locative': 'loc'}


def print_header(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def row(n, errors, label, **kw):
    results = []
    for case in CASES:
        try:
            results.append(f"{CASE_ABBR[case]}:{num2words(n, case=case, **kw)}")
        except Exception as e:  # noqa: BLE001 - report everything
            results.append(f"{CASE_ABBR[case]}:ERROR({e})")
            errors.append((label, case, str(e)))
    return results


def show(n, variants, errors):
    base = None
    for label, kw in variants:
        results = row(n, errors, label, **kw)
        if label in ('mas', 'fem', 'neu') or results != base:
            print(f"  {label:9s}: {' | '.join(results)}")
        if label == 'mas':
            base = results


def validate_cardinal(n):
    print(f"\n{n:,}:")
    print("-" * 90)
    errors = []
    show(n, [('mas', dict(gender='masculine')), ('fem', dict(gender='feminine')),
             ('neu', dict(gender='neuter')),
             ('m.anim', dict(gender='masculine', animacy='animate')),
             ('m.pers', dict(gender='masculine', animacy='personal', construction='agreement')),
             ('undecl.', dict(gender='masculine', declined=False))], errors)
    return errors


def validate_ordinal(n):
    print(f"\n{n:,}. (ordinal):")
    print("-" * 90)
    errors = []
    show(n, [('mas', dict(to='ordinal', gender='masculine')), ('fem', dict(to='ordinal', gender='feminine')),
             ('neu', dict(to='ordinal', gender='neuter')),
             ('m.anim', dict(to='ordinal', gender='masculine', animacy='animate')),
             ('pl m.pers', dict(to='ordinal', gender='masculine', animacy='personal', plural=True)),
             ('pl f.', dict(to='ordinal', gender='feminine', plural=True)),
             ('codified', dict(to='ordinal', gender='masculine', codified=True))], errors)
    return errors


def main():
    print_header("SLOVAK NUM2WORDS v2 - COMPREHENSIVE VALIDATION")
    print(f"Testing CARDINALS: {len(ORIGINAL_TEST_NUMBERS)} numbers")
    print(f"Testing ORDINALS: {len(ORDINAL_TEST_NUMBERS)} numbers")
    print("Each number: 3 genders x 6 cases, plus variant rows where they differ")

    all_errors = []
    print_header("CARDINAL NUMBERS")
    for n in ORIGINAL_TEST_NUMBERS:
        all_errors.extend([(n, 'cardinal', *e) for e in validate_cardinal(n)])

    print_header("ORDINAL NUMBERS")
    for n in ORDINAL_TEST_NUMBERS:
        all_errors.extend([(n, 'ordinal', *e) for e in validate_ordinal(n)])

    print_header("DECIMALS (read in the nominative)")
    for x in DECIMAL_TEST_NUMBERS:
        try:
            print(f"  {x:>9}: {num2words(x)}")
        except Exception as e:  # noqa: BLE001
            print(f"  {x:>9}: ERROR({e})")
            all_errors.append((x, 'decimal', '-', '-', str(e)))

    print_header("SUMMARY")
    print(f"Cardinal numbers tested: {len(ORIGINAL_TEST_NUMBERS)}")
    print(f"Ordinal numbers tested: {len(ORDINAL_TEST_NUMBERS)}")
    print(f"Decimals tested: {len(DECIMAL_TEST_NUMBERS)}")
    print(f"Errors encountered: {len(all_errors)}")
    if all_errors:
        print("\nErrors:")
        for n, num_type, label, case, err in all_errors[:20]:
            print(f"  {n} ({num_type}) {label} {case}: {err}")
    else:
        print("\n✓ All tests completed without exceptions!")


if __name__ == '__main__':
    main()
