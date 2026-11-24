#!/usr/bin/env python3
"""Freeze deterministic tokenizer IDs from manifests, phonetic lexicon and coverage.

Coverage files contain one normalized sentence per nonempty UTF-8 line. Supported
languages are cs/sk/en. Inline spans and quote continuation use the existing
eSpeak NG tokenizer and postrules. Every manifest row is checked with strict ID
lookup, including stored n_tokens parity. No transcript or lexicon is rewritten.

The plain symbol->ID JSON stays compatible with existing readers. A sidecar at
OUT.meta.json binds input hashes, G2P settings and the vocabulary hash. Lexicon
keys also pass through the tokenizer with voicing disabled to retain declared
lexical phone symbols that would otherwise devoice at an isolated-word pause.
Rebuild only before pretraining; later symbol changes need pronunciation review.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.text.corpus_tools import (
    g2p_identity, output_guard, phonemizers, read_input, read_lexicon, read_manifests,
    row_tokens, tokenize, write_outputs,
)
from src.text.phonemizer import CzechSlovakPhonemizer, LANG_CODES, SPECIAL_TOKENS, build_vocabulary, tokens_to_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifests", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lexicon", type=Path)
    parser.add_argument("--coverage", action="append", default=[], metavar="LANG=PATH")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        coverage = []
        for item in args.coverage:
            language, separator, path = item.partition("=")
            if not separator or language not in LANG_CODES or not path:
                raise ValueError("coverage must be LANG=PATH with supported cs/sk/en language")
            coverage.append((language, Path(path)))
        meta_path = Path(str(args.out) + ".meta.json")
        inputs = [*args.manifests, *(path for _, path in coverage)]
        if args.lexicon:
            inputs.append(args.lexicon)
        output_guard([args.out, meta_path], inputs, args.force)
        rows, sources = read_manifests(args.manifests)
        lexicon, lexicon_source = read_lexicon(args.lexicon)
        g2ps = phonemizers(lexicon)
        texts, active = {}, set()
        for row, label in rows:
            tokens = row_tokens(row, label, g2ps)
            active.add(row["lang"])
            active.update(t.lang for t in tokens)
            texts.setdefault(row["lang"], []).append(row["text"])
        coverage_sources = []
        for index, (language, path) in enumerate(coverage, 1):
            text, source = read_input(path, f"coverage {index}")
            lines = [line for line in text.splitlines() if line.strip()]
            for line_number, line in enumerate(lines, 1):
                tokens = tokenize(g2ps[language], line, f"coverage {index}, sentence {line_number}")
                active.update(t.lang for t in tokens)
            active.add(language)
            texts.setdefault(language, []).extend(lines)
            source.update(language=language, sentences=len(lines))
            coverage_sources.append(source)
        for language, entries in lexicon.items():
            if entries:
                active.add(language)
                texts.setdefault(language, []).extend(sorted(entries))
        # Validate every added lexicon key through the same tokenizer before the
        # vocabulary helper, so errors cannot echo private words from its backend.
        for language, entries in lexicon.items():
            for word in entries:
                tokenize(g2ps[language], word, "lexicon entry")
        vocab = build_vocabulary(texts, phonemizers=g2ps)
        inventory_g2ps = {lang: CzechSlovakPhonemizer(lang, lexicon=lexicon,
                           config=replace(g2ps[lang].config, apply_voicing_rules=False)) for lang in lexicon}
        lexical_vocab = build_vocabulary({lang: sorted(entries) for lang, entries in lexicon.items()},
                                         phonemizers=inventory_g2ps)
        symbols = set(vocab) | set(lexical_vocab)
        vocab = {symbol: index for index, symbol in enumerate(SPECIAL_TOKENS)}
        for symbol in sorted(symbols - set(vocab)):
            vocab[symbol] = len(vocab)
        for row, label in rows:
            try:
                tokens_to_ids(row_tokens(row, label, g2ps), vocab, strict=True)
            except KeyError:
                raise ValueError(f"{label}: unknown symbol; rebuild vocabulary with matching G2P settings") from None
        content = json.dumps(vocab, ensure_ascii=False, indent=1) + "\n"
        meta = {"version": 1, "rows": len(rows), "symbols": len(vocab), "languages": sorted(active),
                "manifests": sources, "coverage": coverage_sources, "lexicon": lexicon_source,
                "g2p": g2p_identity(), "lexicon_inventory_apply_voicing_rules": False,
                "vocab_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()}
        write_outputs({args.out: content, meta_path: json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n"},
                      args.force)
        print(json.dumps({"rows": len(rows), "symbols": len(vocab), "languages": len(active)}))
    except (ValueError, OSError, RuntimeError) as error:
        # Known validation errors are already scrubbed; OS/backend failures may
        # contain source content, so never print their raw messages.
        message = str(error) if type(error) is ValueError else "input/output or eSpeak NG backend unavailable"
        parser.exit(2, f"vocabulary build failed: {message}\n")


if __name__ == "__main__":
    main()
