"""Tests for the Czech/Slovak/English G2P. Runs against real espeak-ng 1.52."""
import pytest

from src.text.phonemizer import (
    CzechSlovakPhonemizer, PUNCT_TOKENS, STRESS_MARKS, WORD_BOUNDARY,
    UnknownSymbolError, build_vocabulary, tokens_to_ids,
)

CS = CzechSlovakPhonemizer("cs")
SK = CzechSlovakPhonemizer("sk")
EN = CzechSlovakPhonemizer("en")

COVERAGE = {
    "cs": ["Dobrý den, jak se máte? Rozhlas vysílá zprávy.",
           "Příliš žluťoučký kůň úpěl ďábelské ódy.",
           "Kde je ten muž? Led je tenký, ale hrad byl pevný.",
           "V tom domě bydlí sbor zpěváků; všechno je v pořádku!",
           "Kvůli dešti jsme šli k domu s bratrem a gólman chytil míč.",
           "Tvůj pes štěká: „Haf!“ — a pak se schoval…"],
    "sk": ["Dobrý deň, päť ľudí. Rozhlas vysiela.",
           "Kôň a vôňa, chlieb a mliečny piatok.",
           "Brat ide domov, vták letí nad hradom.",
           "Otec a mama sa s otcom Ďakujem ťave, ďaleko ňufák.",
           "Dievča hovorí: „Nie, ja to viem.“ — a odišla…",
           "Kde je krv? Gól padol v Bratislave; džús a dzurovať!"],
    "en": ["Harry Potter thought the weather was nice.",
           "Sherlock Holmes measured the church through the judge's window.",
           "She sings; they bought a boat, a toy and the cure."],
}


def symbols(tokens):
    return [t.symbol for t in tokens]


def phones(tokens):
    return [t.symbol for t in tokens if t.kind == "phone"]


def words(tokens):
    """Phones grouped per word (split at '|' and punctuation)."""
    out, cur = [], []
    for t in tokens:
        if t.kind == "phone":
            cur.append(t.symbol)
        elif t.kind in ("boundary", "punct") and cur:
            out.append(cur); cur = []
    if cur:
        out.append(cur)
    return out


class TestTokenization:
    @pytest.mark.parametrize("g2p,text", [(CS, s) for s in COVERAGE["cs"]] +
                             [(SK, s) for s in COVERAGE["sk"]] + [(EN, s) for s in COVERAGE["en"]])
    def test_tokens_are_atomic(self, g2p, text):
        """No stress mark, boundary or punctuation glued onto a phone (v1 bug: 'ˈo', 'iː|', 'e?')."""
        for t in g2p.tokenize(text):
            if t.kind == "phone":
                assert not any(m in t.symbol for m in STRESS_MARKS), t
                assert WORD_BOUNDARY not in t.symbol, t
                assert not any(p in t.symbol for p in PUNCT_TOKENS), t
                assert " " not in t.symbol and t.symbol, t

    def test_english_code_is_valid(self):
        # v1 called espeak with 'en' and crashed; we map to 'en-us'
        assert "ɹ" in symbols(EN.tokenize("Harry Potter"))

    def test_empty_input(self):
        assert CS.tokenize("") == [] and CS.tokenize("   ") == []

    def test_unnormalized_digits_raise(self):
        # dropping '5' silently would be a skipped word; the normalizer must run first
        with pytest.raises(ValueError):
            CS.tokenize("Mám 5 jablek.")

    def test_punctuation_tokens(self):
        toks = CS.tokenize("Řekl: „Ano…“ – a odešel.")
        assert [t.symbol for t in toks if t.kind == "punct"] == [":", "…", "—", "."]

    def test_intra_word_hyphen_is_not_a_dash(self):
        assert "—" not in symbols(CS.tokenize("Rimsky-Korsakov"))

    def test_deterministic(self):
        s = "Brat ide domov, vták letí."
        assert symbols(SK.tokenize(s)) == symbols(SK.tokenize(s))


class TestStressPolicy:
    def test_no_stress_for_cs_sk(self):
        assert not any(t.kind == "stress" for t in SK.tokenize("viem, že nie"))
        assert not any(t.kind == "stress" for t in CS.tokenize("automobil"))

    def test_stress_kept_for_english(self):
        assert any(t.kind == "stress" for t in EN.tokenize("hello world"))


class TestLexiconAndSpans:
    def test_lexicon_override_is_applied(self):
        g2p = CzechSlovakPhonemizer("sk", lexicon={"sk": {"covid": "k o v i t"}})
        assert phones(g2p.tokenize("covid")) == ["k", "o", "v", "i", "t"]

    def test_lexicon_is_case_insensitive(self):
        g2p = CzechSlovakPhonemizer("cs", lexicon={"cs": {"Tolkien": "t o l k iː n"}})
        assert phones(g2p.tokenize("TOLKIEN")) == ["t", "o", "l", "k", "iː", "n"]

    def test_language_span(self):
        toks = SK.tokenize("Čítal <en>Harry Potter</en> po slovensky.")
        en = [t for t in toks if t.lang == "en" and t.kind == "phone"]
        sk = [t for t in toks if t.lang == "sk" and t.kind == "phone"]
        assert en and sk
        assert "ɹ" in [t.symbol for t in en]
        assert all(t.lang_id == 2 for t in en) and all(t.lang_id == 1 for t in sk)


class TestEspeakCorrections:
    @pytest.mark.parametrize("g2p", [CS, SK])
    def test_zh_is_voiced(self, g2p):
        p = phones(g2p.tokenize("rozhlas"))
        assert p[:4] == ["r", "o", "z", "h"], p   # espeak-ng 1.52 says [r o s h ...]

    def test_slovak_o_circumflex_is_one_diphthong(self):
        assert "u̯o" in phones(SK.tokenize("kôň"))

    @pytest.mark.parametrize("text,expected_first", [
        ("k domu", "ɡ"), ("s bratrem", "z"), ("v tom", "f"), ("k hradu", "ɡ"),
    ])
    def test_czech_preposition_voicing(self, text, expected_first):
        assert phones(CS.tokenize(text))[0] == expected_first

    @pytest.mark.parametrize("text,expected_last", [
        ("muž je", "ʃ"),       # Czech: voiceless before a sonorant
        ("led je", "t"),
        ("list byl", "d"),     # the cluster assimilates to the voiced b: [lɪzd bɪl]
        ("hrad padl", "t"),
        ("krev.", "f"),        # final devoicing before a pause
    ])
    def test_czech_word_final_voicing(self, text, expected_last):
        assert words(CS.tokenize(text))[0][-1] == expected_last

    @pytest.mark.parametrize("text,expected_last", [
        ("brat ide", "d"),     # Slovak: VOICED before a vowel
        ("vták letí", "ɡ"),    # and before a sonorant
        ("otec a mama", "dz"),
        ("od brata", "d"),
    ])
    def test_slovak_word_final_voicing(self, text, expected_last):
        assert words(SK.tokenize(text))[0][-1] == expected_last

    def test_slovak_final_devoicing_before_pause(self):
        assert phones(SK.tokenize("hrad."))[-1] == "t"

    def test_slovak_preposition_before_vowel(self):
        assert phones(SK.tokenize("s otcom"))[0] == "z"

    @pytest.mark.parametrize("g2p", [CS, SK])
    def test_one_letter_words_are_not_spelled(self, g2p):
        # espeak-ng reads an isolated "s" as the letter name [e s] and "a" as [aː]
        assert words(g2p.tokenize("a s ním"))[:2] == [["a"], ["s"]] or \
            words(g2p.tokenize("a s ním"))[:2] == [["a"], ["z"]]

    def test_initial_is_spelled(self):
        # "K." as an initial keeps its letter name
        assert words(CS.tokenize("K. Čapek"))[0] == ["k", "aː"]

    def test_english_weak_forms_in_phrases(self):
        # phrase-level phonemization keeps the reduced vowel in "a" and "the"
        w = words(EN.tokenize("a cup of the tea"))
        assert w[0] != ["eɪ"]


@pytest.fixture(scope="module")
def vocab():
    return build_vocabulary(COVERAGE)


class TestVocabulary:
    def test_special_tokens_first(self, vocab):
        assert [vocab[t] for t in ("<pad>", "<unk>", "<bos>", "<eos>", "|")] == [0, 1, 2, 3, 4]

    def test_every_training_token_is_known(self, vocab):
        # v1 mapped 41-62% of real cs/sk tokens (88% of English) to <unk>
        for lang, texts in COVERAGE.items():
            g2p = CzechSlovakPhonemizer(lang)
            for text in texts:
                ids, langs = tokens_to_ids(g2p.tokenize(text), vocab)
                assert vocab["<unk>"] not in ids and len(ids) == len(langs)

    def test_unknown_symbol_raises(self, vocab):
        toks = CzechSlovakPhonemizer("en").tokenize("the")
        toks[0].symbol = "ʘ"  # a click consonant nobody trained on
        with pytest.raises(UnknownSymbolError):
            tokens_to_ids(toks, vocab)

    def test_vocabulary_is_deterministic(self, vocab):
        assert build_vocabulary(COVERAGE) == vocab
