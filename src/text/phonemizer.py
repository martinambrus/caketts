"""
Czech / Slovak / English G2P on top of espeak-ng (via `phonemizer`).

What this module guarantees (and the v1 plan did not):
  * valid espeak-ng language codes ('en' -> 'en-us'; 'en' alone crashes espeak-ng)
  * TOKEN output, not a string: phones, word boundaries ('|') and punctuation are
    separate tokens, so nothing like 'ˈo', 'iː|' or 'e?' ever reaches the vocabulary
  * one language ID per token (for the text encoder's language embedding)
  * inline language spans:  "Čítal <en>Harry Potter</en> po anglicky."
  * a word-level exception lexicon that is really applied (looked up per word,
    before espeak-ng)
  * post-rules for known espeak-ng 1.52 errors in Czech/Slovak:
      - 'zh' devoiced to [s h] ("rozhlas" -> [roshlas])            -> fixed per word
      - Slovak 'ô' split into two vowels [u o]                       -> merged to [u̯o]
      - voicing at word boundaries (prepositions k/s/v/z, final
        obstruents) left inconsistent                                -> fixed per language
  * cs/sk stress marks dropped (stress is fixed on the first syllable; espeak-ng
    puts marks on the second half of diphthongs, e.g. "viem" -> v i ˈe m)
  * a vocabulary built from real espeak-ng output and STRICT id lookup:
    an unknown symbol raises instead of silently becoming <unk>

Rules are defaults for native-speaker review: every one of them can be switched off
or extended, and the test-suite pins their behaviour.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------------------
# espeak-ng backend loading
# --------------------------------------------------------------------------------------

LANG_CODES: Dict[str, str] = {"cs": "cs", "sk": "sk", "en": "en-us"}
LANG_IDS: Dict[str, int] = {"cs": 0, "sk": 1, "en": 2}

_BACKENDS: Dict[Tuple[str, bool], object] = {}


def _ensure_espeak_library() -> None:
    """Use the system libespeak-ng if phonemizer finds it, else the pip `espeakng-loader` copy."""
    from phonemizer.backend import EspeakBackend

    try:
        EspeakBackend.version()
        return
    except RuntimeError:
        pass
    import espeakng_loader  # pip install espeakng-loader (bundles espeak-ng 1.52)
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())


def _backend(lang: str, with_stress: bool):
    key = (lang, with_stress)
    if key not in _BACKENDS:
        _ensure_espeak_library()
        from phonemizer.backend import EspeakBackend

        _BACKENDS[key] = EspeakBackend(
            LANG_CODES[lang],
            preserve_punctuation=False,  # punctuation is tokenized by us, not by espeak
            with_stress=with_stress,
            language_switch="remove-flags",
        )
    return _BACKENDS[key]


# --------------------------------------------------------------------------------------
# Token inventory helpers
# --------------------------------------------------------------------------------------

STRESS_MARKS = ("ˈ", "ˌ")
WORD_BOUNDARY = "|"
# Canonical punctuation tokens kept for prosody; everything else is mapped or dropped.
PUNCT_TOKENS = (",", ".", "!", "?", ":", ";", "…", "—")
_PUNCT_MAP = {
    ",": ",", ".": ".", "!": "!", "?": "?", ":": ":", ";": ";",
    "…": "…", "—": "—", "–": "—", "-": None,  # a lone hyphen between spaces behaves like a dash
    "(": ",", ")": ",", "[": ",", "]": ",",
}
_QUOTES = set("\"'„“”‚‘’«»‹›`´")

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", WORD_BOUNDARY, *PUNCT_TOKENS]

_WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)
_SPAN_RE = re.compile(r"<(cs|sk|en)>(.*?)</\1>", re.DOTALL)

# Paired obstruents in espeak-ng 1.52 cs/sk output (voiced -> voiceless)
VOICED_TO_VOICELESS = {
    "b": "p", "d": "t", "ɟ": "c", "ɡ": "k", "v": "f", "z": "s", "ʒ": "ʃ",
    "h": "x", "dz": "ts", "dʒ": "tʃ", "dʲ": "tʲ", "dʑ": "tɕ", "r̝": "r̝̊",
}
VOICELESS_TO_VOICED = {v: k for k, v in VOICED_TO_VOICELESS.items()}
OBSTRUENTS = set(VOICED_TO_VOICELESS) | set(VOICELESS_TO_VOICED)
# 'v' undergoes devoicing but does not trigger voicing; ř is left out as a trigger.
NON_TRIGGERS = {"v", "f", "r̝", "r̝̊"}


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


# --------------------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------------------


@dataclass
class Token:
    symbol: str
    lang: str
    kind: str  # 'phone' | 'stress' | 'boundary' | 'punct'

    @property
    def lang_id(self) -> int:
        return LANG_IDS[self.lang]


@dataclass
class G2PConfig:
    keep_stress: Dict[str, bool] = field(default_factory=lambda: {"cs": False, "sk": False, "en": True})
    apply_voicing_rules: bool = True
    apply_word_rules: bool = True


# --------------------------------------------------------------------------------------
# Word-level post-rules (orthography available)
# --------------------------------------------------------------------------------------


def _fix_zh(word: str, phones: List[str]) -> List[str]:
    """'zh' inside a cs/sk word is [z ɦ] (espeak-ng writes 'h' for ɦ), never [s h]."""
    n_zh = word.lower().count("zh")
    out = list(phones)
    i = 0
    while n_zh and i < len(out) - 1:
        if out[i] == "s" and out[i + 1] == "h":
            out[i] = "z"
            n_zh -= 1
        i += 1
    return out


def _merge_sk_o_circumflex(word: str, phones: List[str]) -> List[str]:
    """Slovak 'ô' is the diphthong [u̯o]; espeak-ng splits it into 'u o'."""
    n = word.lower().count("ô")
    out: List[str] = []
    i = 0
    while i < len(phones):
        if n and phones[i] == "u" and i + 1 < len(phones) and phones[i + 1] == "o":
            out.append("u̯o")
            n -= 1
            i += 2
        else:
            out.append(phones[i])
            i += 1
    return out


WORD_RULES = {
    "cs": [_fix_zh],
    "sk": [_fix_zh, _merge_sk_o_circumflex],
    "en": [],
}

# Built-in entries for one-letter words (applied unless the letter is an initial like "K.")
FUNCTION_WORDS = {
    "cs": {"a": "a", "i": "i", "o": "o", "u": "u", "k": "k", "s": "s", "v": "v", "z": "z"},
    "sk": {"a": "a", "i": "i", "o": "o", "u": "u", "k": "k", "s": "s", "v": "v", "z": "z"},
}


# --------------------------------------------------------------------------------------
# Cross-word voicing (cs/sk)
# --------------------------------------------------------------------------------------


def _first_phone(tokens: Sequence[Token], start: int) -> Optional[Token]:
    for t in tokens[start:]:
        if t.kind == "phone":
            return t
        if t.kind in ("punct", "boundary"):
            return None
    return None


def apply_voicing_rules(tokens: List[Token]) -> List[Token]:
    """
    Regressive voicing assimilation at word boundaries for Czech and Slovak.

    For the obstruent cluster at the end of each cs/sk word:
      next word starts with a voicing trigger (obstruent except v/f/ř) -> copy its voicing
      next is punctuation or end of input (pause)                    -> voiceless
      next word starts with a vowel or sonorant: Czech -> voiceless, Slovak -> voiced
    Language switches (e.g. into an English span) count as a pause.
    """
    out = list(tokens)
    n = len(out)
    i = 0
    while i < n:
        # find the end of the current word: last phone before boundary/punct/end
        if out[i].kind != "phone" or out[i].lang not in ("cs", "sk"):
            i += 1
            continue
        j = i
        while j + 1 < n and out[j + 1].kind in ("phone", "stress") and out[j + 1].lang == out[i].lang:
            j += 1
        word_lang = out[i].lang
        # collect the trailing obstruent cluster [k0..j]
        k0 = j + 1
        while k0 - 1 >= i and out[k0 - 1].kind == "phone" and out[k0 - 1].symbol in OBSTRUENTS:
            k0 -= 1
        if k0 <= j:
            nxt_idx = j + 1
            voiced: Optional[bool]
            if nxt_idx >= n or out[nxt_idx].kind == "punct":
                voiced = False  # pause
            else:
                # skip the boundary token
                look = nxt_idx + 1 if out[nxt_idx].kind == "boundary" else nxt_idx
                nxt = _first_phone(out, look)
                if nxt is None or nxt.lang != word_lang:
                    voiced = False
                elif nxt.symbol in OBSTRUENTS and nxt.symbol not in NON_TRIGGERS:
                    voiced = nxt.symbol in VOICED_TO_VOICELESS
                else:  # vowel or sonorant (incl. v)
                    voiced = word_lang == "sk"
            table = VOICELESS_TO_VOICED if voiced else VOICED_TO_VOICELESS
            for k in range(k0, j + 1):
                sym = out[k].symbol
                if sym in table:
                    out[k] = Token(table[sym], out[k].lang, "phone")
        i = j + 1
    return out


# --------------------------------------------------------------------------------------
# Phonemizer
# --------------------------------------------------------------------------------------


def _split_espeak_word(raw: str, lang: str, keep_stress: bool) -> List[List[Token]]:
    """Turn espeak output for ONE input word into one or more words of tokens."""
    words: List[List[Token]] = []
    for chunk in raw.split(WORD_BOUNDARY):
        toks: List[Token] = []
        for item in chunk.split():
            item = _nfc(item)
            while item and item[0] in STRESS_MARKS:
                if keep_stress:
                    toks.append(Token(item[0], lang, "stress"))
                item = item[1:]
            if not item:
                continue
            item = item.replace("g", "ɡ")  # ASCII g -> IPA ɡ (U+0261)
            toks.append(Token(item, lang, "phone"))
        if toks:
            words.append(toks)
    return words


class CzechSlovakPhonemizer:
    """
    Usage:
        g2p = CzechSlovakPhonemizer(language="sk", lexicon_path="lexicon.json")
        tokens = g2p.tokenize("Čítal <en>Harry Potter</en> po anglicky.")
        symbols = [t.symbol for t in tokens]; lang_ids = [t.lang_id for t in tokens]
    """

    def __init__(
        self,
        language: str = "cs",
        lexicon: Optional[Dict[str, Dict[str, str]]] = None,
        lexicon_path: Optional[str] = None,
        config: Optional[G2PConfig] = None,
    ):
        if language not in LANG_CODES:
            raise ValueError(f"language must be one of {sorted(LANG_CODES)}, got {language!r}")
        self.language = language
        self.config = config or G2PConfig()
        # lexicon: {lang: {word_casefolded: "space separated phones"}}
        self.lexicon: Dict[str, Dict[str, str]] = {k: {} for k in LANG_CODES}
        if lexicon_path:
            data = json.loads(Path(lexicon_path).read_text(encoding="utf-8"))
            for lang, entries in data.items():
                self.add_lexicon_entries(lang, entries)
        if lexicon:
            for lang, entries in lexicon.items():
                self.add_lexicon_entries(lang, entries)

    # ---- lexicon -------------------------------------------------------------------
    def add_lexicon_entries(self, lang: str, entries: Dict[str, str]) -> None:
        for word, phones in entries.items():
            self.lexicon[lang][_nfc(word).casefold()] = _nfc(phones)
        self._espeak.cache_clear()

    # ---- espeak (cached per word / per English phrase) -------------------------------
    @lru_cache(maxsize=200_000)
    def _espeak(self, lang: str, text: str) -> Tuple[Tuple[Tuple[str, str], ...], ...]:
        keep = self.config.keep_stress[lang]
        raw = _backend(lang, keep).phonemize([text], strip=True, separator=_separator())[0]
        words = _split_espeak_word(raw, lang, keep)
        return tuple(tuple((t.symbol, t.kind) for t in w) for w in words)

    def _lexicon_tokens(self, word: str, lang: str, is_initial: bool) -> Optional[List[Token]]:
        key = _nfc(word).casefold()
        entry = self.lexicon[lang].get(key)
        if entry is None and not is_initial:
            # single-letter prepositions/conjunctions: espeak-ng spells an isolated
            # "s" as the letter name [e s] and lengthens "a" to [aː]
            entry = FUNCTION_WORDS.get(lang, {}).get(key)
        if entry is None:
            return None
        return [Token(p, lang, "stress" if p in STRESS_MARKS else "phone") for p in entry.split()]

    def _word_tokens(self, word: str, lang: str, is_initial: bool = False) -> List[List[Token]]:
        lex = self._lexicon_tokens(word, lang, is_initial)
        if lex is not None:
            return [lex]
        words = [[Token(s, lang, k) for s, k in w] for w in self._espeak(lang, word)]
        if not words:
            raise ValueError(f"espeak-ng produced no phones for {word!r} ({lang}); add a lexicon entry")
        if self.config.apply_word_rules and WORD_RULES[lang] and len(words) == 1:
            symbols = [t.symbol for t in words[0]]  # rules only match adjacent phones
            for rule in WORD_RULES[lang]:
                symbols = rule(word, symbols)
            words = [[Token(s, lang, "stress" if s in STRESS_MARKS else "phone") for s in symbols]]
        return words

    def _phrase_tokens(self, phrase: List[str], lang: str) -> List[List[Token]]:
        """English: phonemize runs of words together so weak forms ('the', 'a', 'to') are right."""
        out: List[List[Token]] = []
        run: List[str] = []

        def flush():
            if run:
                ws = [[Token(s, lang, k) for s, k in w] for w in self._espeak(lang, " ".join(run))]
                if not ws:
                    raise ValueError(f"espeak-ng produced no phones for {' '.join(run)!r} ({lang})")
                out.extend(ws)
                run.clear()

        for w in phrase:
            lex = self._lexicon_tokens(w, lang, is_initial=False) if _nfc(w).casefold() in self.lexicon[lang] else None
            if lex is not None:
                flush()
                out.append(lex)
            else:
                run.append(w)
        flush()
        return out

    # ---- public API ----------------------------------------------------------------------
    def tokenize(self, text: str) -> List[Token]:
        text = _nfc(text).replace("...", "…")
        tokens: List[Token] = []
        pos = 0
        spans: List[Tuple[str, str]] = []
        for m in _SPAN_RE.finditer(text):
            if m.start() > pos:
                spans.append((self.language, text[pos:m.start()]))
            spans.append((m.group(1), m.group(2)))
            pos = m.end()
        if pos < len(text):
            spans.append((self.language, text[pos:]))

        def add_word(w_tokens: List[Token]) -> None:
            if tokens and tokens[-1].kind in ("phone", "stress"):
                tokens.append(Token(WORD_BOUNDARY, w_tokens[0].lang, "boundary"))
            tokens.extend(w_tokens)

        for lang, chunk in spans:
            idx = 0
            phrase: List[str] = []  # English words waiting to be phonemized together
            for m in _WORD_RE.finditer(chunk):
                between = chunk[idx:m.start()]
                if lang == "en" and phrase and between.strip(" "):
                    for w in self._phrase_tokens(phrase, lang):
                        add_word(w)
                    phrase = []
                self._emit_punct(between, lang, tokens, text)
                word = m.group(0)
                if lang == "en":
                    phrase.append(word)
                else:
                    is_initial = len(word) == 1 and word.isupper() and chunk[m.end():m.end() + 1] == "."
                    for w in self._word_tokens(word, lang, is_initial):
                        add_word(w)
                idx = m.end()
            if phrase:
                for w in self._phrase_tokens(phrase, lang):
                    add_word(w)
            self._emit_punct(chunk[idx:], lang, tokens, text)

        if self.config.apply_voicing_rules:
            tokens = apply_voicing_rules(tokens)
        return tokens

    # characters that may be dropped silently (they carry no sound)
    IGNORABLE = _QUOTES | set("*_~")

    def _emit_punct(self, between: str, lang: str, tokens: List[Token], text: str) -> None:
        for ch in between:
            if ch.isspace() or ch in self.IGNORABLE:
                continue
            if ch == "-":
                # " - " is a dash; an intra-word hyphen ("Rimsky-Korsakov") is just a boundary
                mapped = "—" if between != "-" and between.strip() == "-" else None
            elif ch in _PUNCT_MAP:
                mapped = _PUNCT_MAP[ch]
            else:
                # digits, %, /, &, @ ... must be verbalised by the text normalizer first;
                # dropping them here would silently skip words
                raise ValueError(f"unnormalized character {ch!r} in {text!r}; run the normalizer first")
            if mapped is None:
                continue
            if tokens and tokens[-1].kind == "punct" and tokens[-1].symbol == mapped:
                continue  # collapse repeats like ",," or ".."
            tokens.append(Token(mapped, lang, "punct"))

    def phonemize(self, text: str) -> str:
        """Human-readable form for logs and lexicon work, e.g. 'r o z h l a s .'"""
        return " ".join(t.symbol for t in self.tokenize(text))


def _separator():
    from phonemizer.separator import Separator

    return Separator(phone=" ", word=f" {WORD_BOUNDARY} ", syllable="")


# --------------------------------------------------------------------------------------
# Vocabulary: built from real espeak-ng output, strict lookup
# --------------------------------------------------------------------------------------


class UnknownSymbolError(KeyError):
    pass


def build_vocabulary(texts_by_lang: Dict[str, Iterable[str]],
                     phonemizers: Optional[Dict[str, CzechSlovakPhonemizer]] = None) -> Dict[str, int]:
    """
    Build the symbol vocabulary from the ACTUAL tokenizer output over the corpus
    (training transcripts + a coverage text per language). Deterministic order.
    """
    symbols = set()
    for lang, texts in texts_by_lang.items():
        g2p = (phonemizers or {}).get(lang) or CzechSlovakPhonemizer(language=lang)
        for text in texts:
            symbols.update(t.symbol for t in g2p.tokenize(text))
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for s in sorted(symbols - set(vocab)):
        vocab[s] = len(vocab)
    return vocab


def tokens_to_ids(tokens: Sequence[Token], vocab: Dict[str, int], strict: bool = True,
                  context: str = "") -> Tuple[List[int], List[int]]:
    """Return (symbol_ids, lang_ids). With strict=True an unknown symbol raises."""
    ids, langs = [], []
    for t in tokens:
        if t.symbol not in vocab:
            if strict:
                raise UnknownSymbolError(f"symbol {t.symbol!r} ({t.lang}) not in vocabulary. {context}")
            ids.append(vocab["<unk>"])
        else:
            ids.append(vocab[t.symbol])
        langs.append(t.lang_id)
    return ids, langs


def save_vocabulary(vocab: Dict[str, int], path: str) -> None:
    Path(path).write_text(json.dumps(vocab, ensure_ascii=False, indent=1), encoding="utf-8")


def load_vocabulary(path: str) -> Dict[str, int]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
