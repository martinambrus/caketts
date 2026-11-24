#!/usr/bin/env python3
"""Export a private pronunciation review TSV; never apply review choices.

Frequent selection groups NFC/casefold spellings per language; each exact source
spelling gets its own row and occurrence count. Capitalization flags are review
candidates, not name recognition. Pattern/name/explicit-span flags bypass top-n.
English runs use phrase-level review, matching the tokenizer's ASCII-space run
boundaries and lexical weak forms. Inline cs/sk/en tags are never counted.

Phones labeled 'isolated' use lexical input at a pause, with current eSpeak NG
postrules. For cs/sk final obstruents, 'context' shows the full actual utterance
token stream (including quote continuation), since voicing can change across
word boundaries. English phrase phones likewise use the actual phrase tokenizer.
No spelling, grammar, slang or lexicon correction is inferred. Nested or escaped
markup is accepted/rejected exactly as the existing tokenizer handles it.
The TSV contains private source text; stdout/stderr contain no transcript words.
"""
import argparse
from collections import Counter
import csv
import io
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.text.corpus_tools import (
    key, output_guard, phonemizers, read_lexicon, read_manifests, row_tokens, source_words, tokenize, write_outputs,
)
from src.text.phonemizer import OBSTRUENTS


def review(rows, g2ps, top_n):
    groups, entries = Counter(), {}

    def add(word, language, row, label, reasons, context, kind="word", initial=False):
        identity = (language, key(word), kind)
        if kind == "word":
            groups[identity] += 1
        exact = (*identity, word, initial)
        entry = entries.setdefault(exact, {"language": language, "word": word, "count": 0, "reasons": set(),
            "example_id": row.get("id", ""), "example_text": row["text"], "label": label,
            "context": context, "initial": initial})
        entry["count"] += 1
        entry["reasons"].update(reasons)
        candidate = (row.get("id", ""), row["text"], row.get("quote_before", 0), row.get("quote_after", 0))
        if "example_key" not in entry or candidate < entry["example_key"]:
            entry.update(example_key=candidate, example_id=row.get("id", ""), example_text=row["text"],
                         label=label, context=context)

    for row, label in rows:
        context = row_tokens(row, label, g2ps)
        for words in source_words(row["text"], row["lang"]):
            phrase = []

            def flush():
                if phrase:
                    a, b = phrase[0][1][0], phrase[-1][1][1]
                    add(row["text"][a:b], "en", row, label, {"english_phrase"}, context, "phrase")
                    phrase.clear()

            for word, (start, end), language, explicit in words:
                reasons = set()
                normalized = key(word)
                if explicit:
                    reasons.add("explicit_span")
                if any(ch.isupper() for ch in word):
                    reasons.add("capitalized_name_candidate")
                if language in ("cs", "sk") and any(p in normalized for p in ("zh", "sh")):
                    reasons.add("postrule_pattern")
                if language == "sk" and any(p in normalized for p in ("ia", "ie", "iu", "ô", "ä")):
                    reasons.add("slovak_pattern")
                if language in ("cs", "sk") and (any(ch in normalized for ch in "qwxöü")
                        or language == "sk" and "y" in normalized
                        or "é" in normalized and any(ch.isupper() for ch in word)):
                    reasons.add("foreign_letter_candidate")
                initial = len(normalized) == 1 and word.isupper() and row["text"][end:end + 1] == "."
                if initial:
                    reasons.add("initial")
                add(word, language, row, label, reasons, context, initial=initial)
                if language == "en":
                    if phrase and row["text"][phrase[-1][1][1]:start].strip(" "):
                        flush()
                    phrase.append((word, (start, end)))
            flush()
    frequent = set(sorted(groups, key=lambda identity: (-groups[identity], identity))[:top_n])
    variants = {}
    for language, normalized, kind, word, initial in entries:
        variants.setdefault((language, normalized, kind), set()).add(word)
    result = []
    for exact, entry in sorted(entries.items()):
        identity = exact[:3]
        reasons = entry["reasons"]
        if identity in frequent:
            reasons.add("frequent")
        if len(variants[identity]) > 1:
            reasons.add("casing_variant")
        if not reasons:
            continue
        language = entry["language"]
        lexical_text = entry["word"] + ("." if entry["initial"] else "")
        tokens = tokenize(g2ps[language], lexical_text, entry["label"])
        phones = "isolated: " + " ".join(t.symbol for t in tokens)
        last = next((t for t in reversed(tokens) if t.kind == "phone"), None)
        if language in ("cs", "sk") and last and last.symbol in OBSTRUENTS:
            reasons.add("context_voicing")
            phones += " ; context: " + " ".join(t.symbol for t in entry["context"])
        result.append({name: entry[name] for name in ("language", "word", "count", "example_id", "example_text")}
                      | {"phones": phones, "reason": ",".join(sorted(reasons))})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lexicon", type=Path)
    parser.add_argument("--top-n", type=int, default=3000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        if args.top_n < 0:
            raise ValueError("top-n must be nonnegative")
        output_guard([args.out], [*args.manifest, *([args.lexicon] if args.lexicon else [])], args.force)
        rows, _ = read_manifests(args.manifest)
        lexicon, _ = read_lexicon(args.lexicon)
        result = review(rows, phonemizers(lexicon), args.top_n)
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=["language", "word", "count", "phones", "reason", "example_id", "example_text"],
                                delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(result)
        write_outputs({args.out: stream.getvalue()}, args.force)
        print(json.dumps({"rows": len(rows), "review_entries": len(result)}))
    except (ValueError, OSError, RuntimeError) as error:
        message = str(error) if type(error) is ValueError else "input/output or eSpeak NG backend unavailable"
        parser.exit(2, f"G2P review failed: {message}\n")


if __name__ == "__main__":
    main()
