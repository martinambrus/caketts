"""Small, explicit book terminology policy, independent of pronunciation guessing.

TSV columns: canonical, alias, language, kind. Each row is a reviewed exact word
form; alias may be empty, kind is name (default), protected (e.g. spoken slang),
or word (trusted ordinary vocabulary). Enumerate inflections
separately. Language describes the source context, not a G2P override: use the
existing inline spans / book_config.english for pronunciation. The orthographic
lexicon remains the mechanism for deliberate book-wide source rewrites.
"""
from __future__ import annotations

import csv
import difflib
import hashlib
import json
import unicodedata
from pathlib import Path

from src.text.phonemizer import _SPAN_RE, _WORD_RE


def key(word):
    return unicodedata.normalize("NFC", word).casefold()


def contextual_words(text, language):
    """Words, offsets and declared languages, with no isolated-word inference."""
    spans = list(_SPAN_RE.finditer(text))
    for match in _WORD_RE.finditer(text):
        if any(s.start() <= match.start() < s.start(2) or s.end(2) <= match.start() < s.end() for s in spans):
            continue
        declared = next((s.group(1) for s in spans if s.start(2) <= match.start() < s.end(2)), language)
        explicit = any(s.start(2) <= match.start() < s.end(2) for s in spans)
        yield match.group(), match.span(), declared, explicit


class BookTerms:
    def __init__(self, language, path=None):
        self.language, self.entries, self.names, self.vocabulary, self.aliases = language, [], set(), set(), {}
        self.source_hashes = {}
        self.policy_sha256 = None
        if path is None:
            return
        data = Path(path).read_bytes()
        self.policy_sha256 = hashlib.sha256(data).hexdigest()
        with Path(path).open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            if (not reader.fieldnames or not {"canonical", "alias", "language"} <= set(reader.fieldnames)
                    or set(reader.fieldnames) - {"canonical", "alias", "language", "kind"}):
                raise ValueError("terminology TSV requires canonical, alias, language and optional kind columns")
            for row in reader:
                canonical, alias, lang = (row.get(k, "") or "" for k in ("canonical", "alias", "language"))
                kind = row.get("kind") or "name"
                if (lang not in ("cs", "sk", "en") or kind not in ("name", "protected", "word")
                        or not _WORD_RE.fullmatch(canonical) or (alias and not _WORD_RE.fullmatch(alias))):
                    raise ValueError("terminology entries must declare exact word forms, language and name/protected/word kind")
                identity = (lang, key(canonical))
                self.vocabulary.add(identity)
                if kind in ("name", "protected"):
                    self.names.add(identity)
                if alias:
                    alias_key = (lang, key(alias))
                    if alias_key in self.aliases and self.aliases[alias_key] != key(canonical):
                        raise ValueError("a terminology alias has competing canonical choices")
                    self.aliases[alias_key] = key(canonical)
                self.entries.append({"canonical": canonical, "alias": alias, "language": lang, "kind": kind})

    def add_source(self, text):
        for word, _, language, explicit in contextual_words(text, self.language):
            identity = (language, key(word))
            self.vocabulary.add(identity)
            # Capitalization is conservative protection, not an NER claim.
            if explicit or word[:1].isupper():
                self.names.add(identity)

    def read_book(self, work):
        for path in sorted((Path(work) / "text").glob("*.units.jsonl")):
            self.source_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.add_source(json.loads(line)["text"])
        return self

    def identity(self):
        return {"version": 1, "language": self.language, "policy_sha256": self.policy_sha256,
                "source_sha256": self.source_hashes}

    def reasons(self, original, replacement, language):
        reasons = []
        if (language, key(original)) in self.names:
            reasons.append("protected_source")
        if (language, key(replacement)) not in self.vocabulary:
            reasons.append("new_vocabulary")
        return reasons

    def compare(self, reference, hypothesis):
        """Diagnostic substitutions only where aligned source expects the canonical.

        No insertion, deletion, repeat, case, suffix or fuzzy name rewriting.
        Return mapped text and the explicit forms used; raw text stays with caller.
        """
        ref = list(contextual_words(reference, self.language))
        hyp = list(contextual_words(hypothesis, self.language))
        changes, applied = [], []
        for op, a, b, c, d in difflib.SequenceMatcher(
                a=[key(w[0]) for w in ref], b=[key(w[0]) for w in hyp], autojunk=False).get_opcodes():
            if op != "replace" or b - a != 1 or d - c != 1:
                continue
            canonical, _, lang, _ = ref[a]
            alias, span, hyp_lang, explicit = hyp[c]
            if explicit and hyp_lang != lang:
                continue
            if self.aliases.get((lang, key(alias))) == key(canonical):
                changes.append((span, canonical))
                applied.append({"canonical": canonical, "alias": alias, "language": lang})
        for (start, end), word in reversed(changes):
            hypothesis = hypothesis[:start] + word + hypothesis[end:]
        return hypothesis, applied
