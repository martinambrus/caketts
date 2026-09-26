#!/usr/bin/env python3
"""
Cross-check num2words_cs / num2words_sk against Unicode CLDR spell-out rules (Step 2.4).

CLDR is NOT ground truth (its Slovak data writes 2000 as "dve tisíce", and older ICU builds
cannot render every rule), so this prints DISAGREEMENTS FOR NATIVE REVIEW, grouped by
gender x case, ignoring spacing and soft hyphens. Needs PyICU, which builds against ICU
(apt install libicu-dev), and network access to GitHub for the current CLDR rule files.

usage: uv run --with PyICU scripts/cldr_crosscheck.py --module-dir src/text --out cldr_diff.tsv
"""
import argparse
import collections
import html
import importlib
import re
import sys
import urllib.request

import icu

CLDR = "https://raw.githubusercontent.com/unicode-org/cldr/main/common/rbnf/{}.xml"
CASES = ["nominative", "genitive", "dative", "accusative", "instrumental", "locative"]
GENDERS = ["masculine", "feminine", "neuter"]
NUMBERS = list(range(0, 130)) + [199, 200, 201, 222, 300, 345, 400, 500, 999, 1000, 1001, 1100,
                                 1999, 2000, 2001, 2024, 3000, 5000, 10000, 21000, 100000,
                                 1_000_000, 2_000_000, 5_000_000, 1_000_000_000, 2_000_000_000]


def cldr_formatter(lang: str) -> icu.RuleBasedNumberFormat:
    xml = urllib.request.urlopen(CLDR.format(lang)).read().decode("utf-8")
    rules = html.unescape(re.findall(r"<rbnfRules>(.*?)</rbnfRules>", xml, re.S)[0])
    return icu.RuleBasedNumberFormat(rules, icu.Locale(lang))


def norm(s: str) -> str:
    return s.replace("­", "").replace(" ", "").lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module-dir", default="src/text")
    ap.add_argument("--out", default="cldr_diff.tsv")
    args = ap.parse_args()
    sys.path.insert(0, args.module_dir)
    rows = []
    for lang in ("cs", "sk"):
        mod = importlib.import_module(f"num2words_{lang}")
        fmt = cldr_formatter(lang)
        per_cell, total = collections.Counter(), 0
        for kind in ("cardinal", "ordinal"):
            for g in GENDERS:
                for c in CASES:
                    rs = f"%spellout-{kind}-{g}" + ("" if c == "nominative" else f"-{c}")
                    try:
                        fmt.setDefaultRuleSet(rs)
                    except icu.ICUError:
                        continue  # rule set absent in this CLDR/ICU combination
                    for n in NUMBERS:
                        if kind == "ordinal" and n == 0:
                            continue
                        ref = fmt.format(n)
                        if not ref or "|" in ref or "%" in ref:  # rule syntax this ICU cannot render
                            continue
                        ours = mod.num2words(n, to=kind, gender=g, case=c)
                        total += 1
                        if norm(ours) != norm(ref):
                            per_cell[(kind, g, c)] += 1
                            rows.append((lang, kind, g, c, n, ours, ref))
        print(f"{lang}: {sum(per_cell.values())} of {total} forms differ from CLDR (spacing ignored)")
        for (kind, g, c), k in per_cell.most_common(6):
            print(f"   {kind:8s} {g:9s} {c:12s} {k}")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("lang\tkind\tgender\tcase\tnumber\tours\tcldr\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")
    print(f"wrote {len(rows)} disagreements to {args.out} — review them, do not auto-apply")


if __name__ == "__main__":
    main()
