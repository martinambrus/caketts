"""Read-only corpus tooling shared by the review and vocabulary CLIs.

Errors identify input/row numbers, never transcript text or utterance IDs.
The phonemizer remains the authority on markup, punctuation and phone symbols.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unicodedata

from src.text.phonemizer import (
    CzechSlovakPhonemizer, G2PConfig, LANG_CODES, SPECIAL_TOKENS, _WORD_RE,
)


def key(text):
    return unicodedata.normalize("NFC", text).casefold()


def _object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("duplicate JSON key")
        result[name] = value
    return result


def parse_json(text, label):
    try:
        return json.loads(text, object_pairs_hook=_object)
    except (ValueError, TypeError):
        raise ValueError(f"{label}: invalid JSON object or duplicate key") from None


def read_input(path, label):
    try:
        data = Path(path).read_bytes()
        text = data.decode("utf-8")
    except (OSError, UnicodeError):
        raise ValueError(f"{label}: cannot read UTF-8 input") from None
    return text, {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(data).hexdigest()}


def read_manifests(paths):
    rows, sources = [], []
    for index, path in enumerate(paths, 1):
        text, source = read_input(path, f"manifest {index}")
        count = 0
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            label = f"manifest {index}, line {line_number}"
            row = parse_json(line, label)
            if (not isinstance(row, dict) or not isinstance(row.get("lang"), str)
                    or row["lang"] not in LANG_CODES):
                raise ValueError(f"{label}: row requires supported lang (cs/sk/en)")
            if not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError(f"{label}: row requires nonempty text")
            if "id" in row and not isinstance(row["id"], str):
                raise ValueError(f"{label}: id must be a string")
            for field in ("quote_before", "quote_after", "n_tokens"):
                if field in row and (type(row[field]) is not int or row[field] < 0):
                    raise ValueError(f"{label}: {field} must be a nonnegative integer")
            rows.append((row, label))
            count += 1
        source["rows"] = count
        sources.append(source)
    if not rows:
        raise ValueError("manifests contain no rows")
    return rows, sources


def read_lexicon(path):
    if path is None:
        return {}, None
    text, source = read_input(path, "lexicon")
    data = parse_json(text, "lexicon")
    if not isinstance(data, dict) or any(lang not in LANG_CODES for lang in data):
        raise ValueError("lexicon requires an object with supported cs/sk/en language keys")
    clean = {}
    count = 0
    for lang, entries in data.items():
        if not isinstance(entries, dict):
            raise ValueError("lexicon language entries must be word-to-phones objects")
        normalized = {}
        for word, phones in entries.items():
            if not isinstance(word, str) or not _WORD_RE.fullmatch(unicodedata.normalize("NFC", word)):
                raise ValueError("lexicon keys must be single tokenizer words")
            if (not isinstance(phones, str) or not phones.split()
                    or any(p in SPECIAL_TOKENS for p in phones.split())
                    or any(unicodedata.category(ch).startswith("C") for ch in phones)):
                raise ValueError("lexicon entries require nonempty phone strings without special/control tokens")
            identity, phones = key(word), unicodedata.normalize("NFC", phones)
            if identity in normalized and normalized[identity] != phones:
                raise ValueError("lexicon has conflicting NFC/casefold word entries")
            normalized[identity] = phones
            count += 1
        clean[lang] = normalized
    source["entries"] = count
    return clean, source


def phonemizers(lexicon):
    return {lang: CzechSlovakPhonemizer(lang, lexicon=lexicon) for lang in LANG_CODES}


def tokenize(g2p, text, label, **kwargs):
    try:
        return g2p.tokenize(text, **kwargs)
    except (ValueError, KeyError, RuntimeError, IndexError):
        raise ValueError(f"{label}: tokenizer rejected input; check normalization, language spans and lexicon privately") from None


def row_tokens(row, label, g2ps):
    tokens = tokenize(g2ps[row["lang"]], row["text"], label,
                      quote_before=row.get("quote_before", 0), quote_after=row.get("quote_after", 0))
    if "n_tokens" in row and row["n_tokens"] != len(tokens):
        raise ValueError(f"{label}: n_tokens mismatch (stored {row['n_tokens']}, actual {len(tokens)}); "
                         "regenerate manifest with the same G2P settings and lexicon")
    return tokens


def output_guard(outputs, inputs, force=False):
    resolved = [Path(path).resolve() for path in outputs]
    if len(set(resolved)) != len(resolved) or set(resolved) & {Path(path).resolve() for path in inputs}:
        raise ValueError("output paths must differ from each other and every input")
    if not force and any(Path(path).exists() for path in outputs):
        raise ValueError("output exists; use --force to replace it explicitly")


def write_outputs(outputs, force=False):
    """Publish complete, owner-readable files; never truncate an input hardlink.

    Each file is atomic. The sidecar hash detects a mixed pair if a forced
    multi-file replacement is interrupted between its two publications.
    """
    created = []
    temporary = None
    try:
        for path, content in outputs.items():
            path = Path(path)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent,
                                             prefix=".corpus-tool-", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(content)
            if force:
                os.replace(temporary, path)
            else:
                os.link(temporary, path)  # atomic publication, fails if destination exists
                temporary.unlink()
            temporary = None
            created.append(path)
    except OSError:
        if not force:
            for path in created:
                path.unlink(missing_ok=True)
        raise ValueError("cannot write outputs (check destination or existing files)") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def g2p_identity():
    from importlib.metadata import version
    from phonemizer.backend import EspeakBackend
    from src.text import phonemizer
    phonemizer._ensure_espeak_library()
    return {"backend": "espeak-ng", "espeak_version": list(EspeakBackend.version()),
            "phonemizer_version": version("phonemizer"), "language_codes": LANG_CODES,
            "config": asdict(G2PConfig()),
            "implementation_sha256": hashlib.sha256(Path(phonemizer.__file__).read_bytes()).hexdigest()}


def source_words(text, default_language):
    """Tokenizer word/span boundaries with exact original Unicode slices.

    Canonical clusters map NFC indices back to source indices. Language markup
    is split by the same regex as the tokenizer. The caller first tokenizes the
    entire row, so nested/escaped/unsupported markup gets the same rejection.
    """
    from src.text.phonemizer import _SPAN_RE
    clusters = []
    start = 0
    for index, ch in enumerate(text):
        if index and not unicodedata.combining(ch):
            clusters.append((start, index))
            start = index
    if text:
        clusters.append((start, len(text)))
    normalized, offsets = [], []
    for start, end in clusters:
        part = unicodedata.normalize("NFC", text[start:end])
        normalized.append(part)
        offsets.extend([(start, end)] * len(part))
    normalized = "".join(normalized)
    chunks, position = [], 0
    for span in _SPAN_RE.finditer(normalized):
        chunks.append((position, span.start(), default_language, False))
        chunks.append((span.start(2), span.end(2), span.group(1), True))
        position = span.end()
    chunks.append((position, len(normalized), default_language, False))
    for start, end, language, explicit in chunks:
        words = []
        for match in _WORD_RE.finditer(normalized, start, end):
            a, b = offsets[match.start()][0], offsets[match.end() - 1][1]
            words.append((text[a:b], (a, b), language, explicit))
        yield words
