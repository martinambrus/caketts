import pytest

from src.data.prepare.lexicon import apply_lexicon, derive, derive_to_file, read_lexicon, write_lexicon
from src.text.normalizer import TextNormalizer


@pytest.fixture(scope="module")
def normalizer():
    n = TextNormalizer("cs")
    n.normalize("Přišel domů.")
    return n


def pairs_of(normalizer, book, spoken):
    return derive({"p1": book}, {"p1": spoken}, normalizer)


def find(pairs, form, spoken=None):
    found = [p for p in pairs if p["book_form"] == form and (spoken is None or p["spoken_form"] == spoken)]
    assert found, (form, spoken, pairs)
    return found


def test_respelling_twice_is_consistent_and_applied(normalizer):
    pairs = pairs_of(normalizer, "Jack přišel domů. Potom šel Jack pryč.", "Džek přišel domů. Potom šel Džek pryč.")
    assert pairs == [{"book_form": "Jack", "spoken_form": "Džek", "category": "respelling", "count": 2,
                      "distinct": 1, "consistent": True, "apply": True}]


def test_inconsistent_respelling_is_listed_and_not_applied(normalizer):
    pairs = pairs_of(normalizer, "Mike šel domů. Potom Mike spal.", "Majk šel domů. Potom Majck spal.")
    assert sorted((p["spoken_form"], p["count"], p["distinct"], p["consistent"], p["apply"]) for p in pairs) == [
        ("Majck", 1, 2, False, False), ("Majk", 1, 2, False, False)]


def test_respelling_of_a_word_that_is_changed_only_once_is_not_consistent(normalizer):
    pairs = pairs_of(normalizer, "Šel domů a potom šel pryč.", "Šel domů a potom jel pryč.")
    got = find(pairs, "šel", "jel")[0]
    assert got["count"] == 1 and got["consistent"] is False and got["apply"] is False


def test_number_stays_with_the_normalizer(normalizer):
    got = find(pairs_of(normalizer, "Měl 15 psů doma.", "Měl patnáct psů doma."), "15")[0]
    assert (got["category"], got["spoken_form"], got["apply"]) == ("number", "patnáct", False)


def test_abbreviation_stays_with_the_normalizer(normalizer):
    got = find(pairs_of(normalizer, "Přišel např. včera domů.", "Přišel například včera domů."), "např.")[0]
    assert (got["category"], got["spoken_form"], got["apply"]) == ("abbreviation", "například", False)


def test_inserted_and_deleted_words(normalizer):
    ins = find(pairs_of(normalizer, "Přišel domů včera.", "Přišel potom domů včera."), "", "potom")[0]
    assert (ins["category"], ins["apply"]) == ("insertion", False)
    dele = find(pairs_of(normalizer, "Přišel domů úplně včera.", "Přišel domů včera."), "úplně", "")[0]
    assert (dele["category"], dele["apply"]) == ("deletion", False)


def test_slash_symbol_pair(normalizer):
    got = find(pairs_of(normalizer, "Volba A / B doma.", "Volba A lomeno B doma."), "/")[0]
    assert (got["category"], got["spoken_form"], got["apply"]) == ("symbol", "lomeno", True)


def test_symbol_inside_a_long_span_is_one_phrase_pair(normalizer):
    book = "Volba A / B C D E F G H doma."
    spoken = "Volba A lomeno bé cé dé é ef gé há doma."
    got = find(pairs_of(normalizer, book, spoken), "/ B C D E F G H")[0]
    assert (got["category"], got["spoken_form"], got["apply"]) == ("symbol", "lomeno bé cé dé é ef gé há", True)
    assert apply_lexicon(book, {got["book_form"]: got["spoken_form"]}) == (spoken, 1)


def test_long_span_without_an_unreadable_symbol_is_not_a_pair(normalizer):
    book = "Dva psi / tři kočky doma a B C D E F G H tu."
    spoken = "Dva psi nebo tři kočky doma a j k l m n o p tu."
    pairs = pairs_of(normalizer, book, spoken)
    assert not [p for p in pairs if "/" in p["book_form"] or len(p["book_form"].split()) > 6]


def test_long_span_does_not_cross_a_line_break(normalizer):
    pairs = pairs_of(normalizer, "Volba A / B C\nD E F G H doma.", "Volba A lomeno bé cé dé é ef gé há doma.")
    assert not [p for p in pairs if len(p["book_form"].split()) > 6]


def test_symbol_without_counterpart_is_applied_as_removed(normalizer):
    got = find(pairs_of(normalizer, "Dva psi | tři kočky doma.", "Dva psi tři kočky doma."), "|")[0]
    assert (got["category"], got["spoken_form"], got["apply"]) == ("symbol", "", True)


def test_slash_the_normalizer_reads_is_no_symbol(normalizer):
    pairs = pairs_of(normalizer, "Dva psi / tři kočky doma.", "Dva psi nebo tři kočky doma.")
    assert [(p["book_form"], p["spoken_form"], p["category"]) for p in pairs] == [("", "nebo", "insertion")]


def test_readable_symbol_is_no_symbol(normalizer):
    pairs = pairs_of(normalizer, "Zaplatil 5 % z ceny.", "Zaplatil pět procent z ceny.")
    assert not [p for p in pairs if p["category"] == "symbol"]


def test_inconsistent_symbol_applies_its_most_frequent_spoken_form(normalizer):
    book = "A / b doma. C / d doma. E / f doma."
    spoken = "A lomeno b doma. C lomeno d doma. E na f doma."
    pairs = pairs_of(normalizer, book, spoken)
    assert sorted((p["spoken_form"], p["count"], p["consistent"], p["apply"]) for p in find(pairs, "/")) == [
        ("lomeno", 2, False, True), ("na", 1, False, False)]


def test_name_that_is_a_prefix_of_another_word_is_whole_word_only(normalizer):
    pairs = pairs_of(normalizer, "Pat přišel domů. Patrik přišel domů.", "Pet přišel domů. Patrik přišel domů.")
    assert find(pairs, "Pat", "Pet")[0]["consistent"] is True
    assert apply_lexicon("Pat přišel. Patrik přišel. Pata přišla.", {"Pat": "Pet"}) == ("Pet přišel. Patrik přišel. Pata přišla.", 1)


def test_respelling_at_a_sentence_start(normalizer):
    pairs = pairs_of(normalizer, "Okay šel domů. Řekl okay.", "Okej šel domů. Řekl okej.")
    assert {(p["book_form"], p["spoken_form"], p["consistent"]) for p in pairs} == {
        ("Okay", "Okej", True), ("okay", "okej", True)}


def test_apply_keeps_the_capital_of_a_sentence_start(normalizer):
    text = "Okay šel domů. Řekl okay. „Okay,“ řekl. Řekl Okay."
    assert apply_lexicon(text, {"okay": "okej"}) == ("Okej šel domů. Řekl okej. „Okej,“ řekl. Řekl Okay.", 3)


def test_apply_phrase_longest_first_and_not_inside_a_phrase(normalizer):
    entries = {"Harry Potter": "Heri Poter", "Potter": "Poter"}
    assert apply_lexicon("Harry Potter a Potter. Harry Pottera.", entries) == ("Heri Poter a Poter. Harry Pottera.", 2)


def test_apply_symbols_without_touching_fractions():
    entries = {"/": "lomeno", "|": ""}
    got, count = apply_lexicon("a / b a 1/2 a|b | c\n/ d", entries)
    assert got == "a lomeno b a 1/2 a b c\nlomeno d" and count == 4


def test_apply_empty_lexicon_changes_nothing():
    assert apply_lexicon("Vše zůstane.", {}) == ("Vše zůstane.", 0)


def test_lexicon_file_roundtrip_and_defaults(tmp_path, normalizer):
    pairs = pairs_of(normalizer, "Jack šel A / B domů. Měl 15 psů. Mike přišel. Mike odešel.",
                     "Džek šel A lomeno B domů. Měl patnáct psů. Majk přišel. Majck odešel.")
    path = tmp_path / "lexicon.tsv"
    write_lexicon(pairs, str(path))
    lines = path.read_text(encoding="utf-8").split("\n")
    assert lines[0] == "book_form\tspoken_form\tcategory\tcount\tconsistent\tapply"
    assert read_lexicon(str(path)) == {"Jack": "Džek", "/": "lomeno"}


def test_hand_written_lexicon_needs_only_three_columns(tmp_path):
    path = tmp_path / "hand.tsv"
    path.write_text("book_form\tspoken_form\tapply\nPat\tPet\ty\nBob\tBop\tn\n", encoding="utf-8")
    assert read_lexicon(str(path)) == {"Pat": "Pet"}


def test_empty_lexicon_files(tmp_path):
    path = tmp_path / "empty.tsv"
    path.write_text("", encoding="utf-8")
    assert read_lexicon(str(path)) == {}
    path.write_text("book_form\tspoken_form\tapply\n", encoding="utf-8")
    assert read_lexicon(str(path)) == {}


def test_two_applied_rows_for_one_book_form_are_refused(tmp_path):
    path = tmp_path / "two.tsv"
    path.write_text("book_form\tspoken_form\tapply\nPat\tPet\ty\nPat\tPit\ty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Pat"):
        read_lexicon(str(path))


def test_missing_required_column_is_refused(tmp_path):
    path = tmp_path / "bad.tsv"
    path.write_text("book_form\tspoken_form\nPat\tPet\n", encoding="utf-8")
    with pytest.raises(ValueError, match="apply"):
        read_lexicon(str(path))


def test_derive_to_file_reads_the_two_directories(tmp_path, normalizer):
    book, rewrite = tmp_path / "book", tmp_path / "rewrite"
    book.mkdir()
    rewrite.mkdir()
    (book / "a2.txt").write_text("Jack přišel domů.\n", encoding="utf-8")
    (book / "a10.txt").write_text("Potom šel Jack pryč.\n", encoding="utf-8")
    (rewrite / "a2.txt.orig").write_bytes("Džek přišel domů.\r\n".encode("utf-8"))
    (rewrite / "a10.txt.orig").write_bytes("Potom šel Džek pryč.\r\n".encode("utf-8"))
    out = tmp_path / "lexicon.tsv"
    pairs = derive_to_file(str(book), str(rewrite), str(out), normalizer=normalizer)
    assert [(p["book_form"], p["count"], p["consistent"]) for p in pairs] == [("Jack", 2, True)]
    assert read_lexicon(str(out)) == {"Jack": "Džek"}
