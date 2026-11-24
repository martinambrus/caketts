"""Exact, reviewed terminology preserves source spelling and declared language."""
import pytest

from src.data.prepare.terms import BookTerms, contextual_words


def policy(tmp_path, rows):
    path = tmp_path / "terms.tsv"
    path.write_text("canonical\talias\tlanguage\tkind\n" + rows, encoding="utf-8")
    return BookTerms("cs", path)


def test_exact_contextual_aliases_preserve_inflections_case_and_real_errors(tmp_path):
    terms = policy(tmp_path, "Zorim\tSorim\tcs\tname\nZorima\tSorima\tcs\tname\n")
    assert terms.compare("Přišel Zorim.", "Přišel Sorim.")[0] == "Přišel Zorim."
    assert terms.compare("U Zorima čekám.", "U Sorima čekám.")[0] == "U Zorima čekám."
    for ref, hyp in [("Přišel Zorim.", "Přišel Sorima."),  # different inflection
                     ("Přišel Zorim.", "Přišel Sorim Sorim."),  # repetition
                     ("Přišel Zorim.", "Přišel."),  # omission
                     ("Přišel Zorim.", "Přišel Sorimův."),  # not a suffix guess
                     ("Přišel jiný.", "Přišel Sorim.")]:  # wrong context
        assert terms.compare(ref, hyp) == (hyp, [])
    mapped, applied = terms.compare("Přišel ZORIM pozdě.", "Přišel sorim pozdě!")
    assert mapped == "Přišel ZORIM pozdě!" and len(applied) == 1


def test_explicit_language_is_inherited_without_word_guessing(tmp_path):
    terms = policy(tmp_path, "Zorim\tSorim\tcs\tname\n")
    words = list(contextual_words("Zorim a <en>Zorim</en> a <sk>Zorim</sk>", "cs"))
    assert [w[2] for w in words] == ["cs", "cs", "en", "cs", "sk"]
    assert terms.compare("<en>Zorim</en>", "Sorim") == ("Sorim", [])
    assert terms.compare("Zorim", "<sk>Sorim</sk>") == ("<sk>Sorim</sk>", [])
    assert terms.compare("<cs>Zorim</cs>", "Sorim")[0] == "Zorim"


def test_protected_spoken_colloquial_form_is_never_grammatically_corrected(tmp_path):
    terms = policy(tmp_path, "mladej\tmladý\tcs\tprotected\nnovinka\t\tcs\tword\n")
    terms.add_source("Ten mladej člověk je mladý. <en>custom</en>")
    assert terms.reasons("mladej", "mladý", "cs") == ["protected_source"]
    assert terms.reasons("custom", "novinka", "en") == ["protected_source", "new_vocabulary"]
    assert terms.reasons("něco", "novinka", "cs") == []
    assert terms.reasons("něco", "unseen", "cs") == ["new_vocabulary"]


@pytest.mark.parametrize("rows", ["Zorim\tSorim\tcs\tname\nJorim\tSorim\tcs\tname\n",
                                  "Zorim\tSorim\tguess\tname\n",
                                  "Zorim Jorim\tSorim\tcs\tname\n"])
def test_policy_rejects_competing_choices_and_guessed_language(tmp_path, rows):
    with pytest.raises(ValueError):
        policy(tmp_path, rows)
