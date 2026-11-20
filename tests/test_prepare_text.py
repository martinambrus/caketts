import csv
import json
import logging
import math
import os
import re
import subprocess
import sys
import unicodedata

import pytest

from src.data.prepare import text as stage
from src.data.prepare.text import TextConfig, TextStageError, run_text_stage
from src.text.normalizer import TextNormalizer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KEYS = ["id", "part", "index", "text", "sentences", "boundary", "chapter", "paragraph", "heading", "est_s",
        "words", "word_spans", "tags", "source"]


@pytest.fixture(scope="module")
def normalizer():
    n = TextNormalizer("cs")
    n.normalize("Přišel domů.")
    return n


def book(*paragraphs):
    return "\n\n".join(paragraphs) + "\n"


def make(tmp_path, parts, work="work", **kw):
    text_dir = tmp_path / "in"
    text_dir.mkdir(exist_ok=True)
    for name, data in parts.items():
        (text_dir / f"{name}.txt.orig").write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
    return TextConfig(text_dir=str(text_dir), work_dir=str(tmp_path / work), **kw)


def run(tmp_path, parts, work="work", **kw):
    report = run_text_stage(make(tmp_path, parts, work, **kw))
    return report, tmp_path / work / "text"


def units_of(out, part="p1"):
    lines = (out / f"{part}.units.jsonl").read_text(encoding="utf-8").split("\n")
    assert lines.pop() == ""
    return [json.loads(line) for line in lines]


def at_line(units, line):
    return [u for u in units if u["source"]["line_start"] == line]


def write_placement(tmp_path, parts):
    path = tmp_path / "placement.json"
    path.write_text(json.dumps({"source": "orig", "parts": parts}), encoding="utf-8")
    return str(path)


def end(next_="paragraph", inside=False, mid=False):
    return {"inside_paragraph": inside, "mid_sentence": mid, "next": next_}


def place(starts=(), start=(False, False), finish=None):
    return {"chapter_starts": list(starts), "start": {"inside_paragraph": start[0], "mid_sentence": start[1]},
            "end": finish or end()}


FORMAT_LINES = [
    "Návrat domů",
    "",
    "Kočka seděla na střeše domu. Slunce zapadalo za kopce.",
    "Druhý řádek odstavce bez tečky",
    "",
    "KAPITOLA PRVNÍ",
    "",
    "Tab\tuvnitř věty zde.",
    "Nebeská báseň zazněla.",
    "mezera na konci řádku. ",
    "",
    "   ",
    "Dlouhý rozklad é slova zde.",
    "# Nadpis",
    "",
    "Poslední řádek bez konce.",
]


def format_bytes():
    seps = ["\r\n"] * 15
    seps[1] = "\n"
    seps[4] = "\r"
    return b"\xef\xbb\xbf" + "".join(line + sep for line, sep in zip(FORMAT_LINES, seps + [""])).encode("utf-8")


# ---- format check ----------------------------------------------------------------------------------------------

def test_format_check_reports_and_does_not_apply(tmp_path):
    report, out = run(tmp_path, {"p1": format_bytes()})
    fmt = report["parts"]["p1"]["format"]
    assert fmt["bom"] is True
    assert fmt["line_endings"]["counts"] == {"CRLF": 13, "LF": 1, "CR": 1}
    assert fmt["line_endings"]["minority_lines"] == {"LF": [2], "CR": [5]}
    assert fmt["missing_final_newline"] is True
    assert fmt["non_nfc_lines"] == [13]
    assert fmt["tab_lines"] == [8]
    assert fmt["nbsp_lines"] == [9]
    assert fmt["edge_space_lines"] == [10]
    assert fmt["whitespace_only_lines"] == [12]
    assert fmt["blank_runs"] == [{"line": 11, "length": 2}]
    assert fmt["heading_lines"] == [14]
    assert [p["line"] for p in fmt["consecutive_pairs"]] == [3, 8, 9]
    assert fmt["consecutive_pairs"][0] == {"line": 3, "ends_sentence": True, "next_starts_upper_or_quote": True}
    assert fmt["consecutive_pairs"][2] == {"line": 9, "ends_sentence": True, "next_starts_upper_or_quote": False}
    candidates = {c["line"]: c for c in fmt["heading_candidates"]}
    assert 3 not in candidates and {1, 4, 6} <= set(candidates)
    assert "first_line" in candidates[1]["reasons"] and candidates[1]["shape"] == "Aaaaaa aaaa"
    assert "all_caps" in candidates[6]["reasons"] and candidates[6]["shape"] == "AAAAAAAA AAAAA"
    norm = (out / "p1.norm.txt").read_text(encoding="utf-8").split("\n")
    assert [line for line in norm if line.startswith("# ")] == ["# Nadpis"]
    assert report["parts"]["p1"]["headings"] == 1


def test_other_line_separator_is_converted_and_named(tmp_path):
    data = "První věta zde. Druhá věta zde.\nTřetí věta zde.\n".encode("utf-8")
    report, out = run(tmp_path, {"p1": data})
    fmt = report["parts"]["p1"]["format"]
    assert fmt["line_endings"]["counts"] == {"LF": 2, "U+2028": 1}
    assert fmt["line_endings"]["minority_lines"] == {"U+2028": [1]}
    assert (out / "p1.norm.txt").read_text(encoding="utf-8").count("\n") == 3


def test_invalid_utf8_names_part_and_offset(tmp_path):
    with pytest.raises(TextStageError) as info:
        run(tmp_path, {"p1": b"Dobr\xff den.\n"})
    assert info.value.failures == [{"part": "p1", "error": "utf-8", "byte_offset": 4}]


def test_config_headings_applied(tmp_path):
    parts = {"p1": book("Úvodní řádek zde", "Prostě věta jedna. Prostě věta dvě.")}
    report, out = run(tmp_path, parts, headings={"p1": [1, 3, 99]})
    norm = (out / "p1.norm.txt").read_text(encoding="utf-8").split("\n")
    assert norm[0] == "# Úvodní řádek zde" and norm[1] == "" and norm[2].startswith("# Prostě")
    first = units_of(out)[0]
    assert first["heading"] is True and "heading" in first["tags"]
    ch = report["parts"]["p1"]["config_headings"]
    assert ch["applied"] == [1, 3]
    assert ch["not_applied"] == [{"line": 99, "reason": "line_out_of_range"}]


# ---- logged readings ---------------------------------------------------------------------------------------

def test_logged_readings_reach_csv_with_paragraph_and_line(tmp_path):
    text = book("# 5. Návrat domů", "Přišel domů.", "# 9. Cesta zpět", "Odešel pryč.")
    report, out = run(tmp_path, {"p1": text})
    with open(out / "review_readings.csv", encoding="utf-8", newline="") as f:
        raw = f.read()
        rows = list(csv.reader(raw.splitlines()))
    assert "\r" not in raw
    assert rows[0] == ["part", "paragraph", "line", "reason", "source", "reading", "context", "matches"]
    body = rows[1:]
    assert len(body) == 2
    assert [(r[0], r[1], r[2], r[4], r[6], r[7]) for r in body] == [
        ("p1", "1", "1", "5.", "5. Návrat domů", "1"), ("p1", "3", "5", "9.", "9. Cesta zpět", "1")]
    assert all(r[3] and r[5] for r in body)
    assert sum(report["parts"]["p1"]["readings_by_reason"].values()) == 2
    assert report["total"]["readings_by_reason"] == report["parts"]["p1"]["readings_by_reason"]


def test_record_args_shape_is_pinned(normalizer):
    out, records = stage._normalise_logged(normalizer, "# 5. Návrat domů\n")
    assert out.startswith("# ")
    assert len(records) == 1 and len(records[0]) == 4 and all(isinstance(x, str) for x in records[0])

    class Odd:
        def normalize(self, text):
            import logging
            logging.getLogger("src.text.normalizer").warning("odd %s", "one")
            return text

    with pytest.raises(TextStageError, match="log record"):
        stage._normalise_logged(Odd(), "x\n")


def test_logger_level_restored(normalizer):
    import logging
    logger = logging.getLogger("src.text.normalizer")
    logger.setLevel(logging.ERROR)
    stage._normalise_logged(normalizer, "Přišel domů.\n")
    assert logger.level == logging.ERROR and logger.handlers == []
    logger.setLevel(logging.NOTSET)


def test_unmatched_context_gives_empty_paragraph(tmp_path, monkeypatch):
    class Fake:
        abbreviations = {}

        def __init__(self, *a):
            pass

        def normalize(self, text):
            import logging
            logging.getLogger("src.text.normalizer").warning("%s: read %r as %r in %r", "why", "a", "b", "nowhere")
            return text

    monkeypatch.setattr(stage, "TextNormalizer", Fake)
    _, out = run(tmp_path, {"p1": book("Jedna věta zde.")})
    rows = list(csv.reader((out / "review_readings.csv").read_text(encoding="utf-8").splitlines()))
    assert rows[1] == ["p1", "", "", "why", "a", "b", "nowhere", "0"]


# ---- the ValueError path --------------------------------------------------------------------------------------

def test_value_error_lists_every_failing_block_and_code_point(tmp_path):
    bad = book("Dva ≈ tři.", "Dobrý den.", "Čtyři ☃ a ∑ pět.")
    with pytest.raises(TextStageError, match="2") as info:
        run(tmp_path, {"p1": bad, "p2": book("Přišel domů.")})
    assert info.value.failures == [
        {"part": "p1", "line": 1, "block": 1, "codepoints": ["U+2248"]},
        {"part": "p1", "line": 5, "block": 3, "codepoints": ["U+2603", "U+2211"]}]
    out = tmp_path / "work" / "text"
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed" and report["failures"] == info.value.failures
    assert report["parts"]["p1"]["format"]["line_count"] == 5
    assert not (out / "p1.units.jsonl").exists()


def test_skip_unreadable_drops_only_the_failing_blocks_and_reports_them(tmp_path):
    report, out = run(tmp_path, {"p1": book("Dva ≈ tři.", "Dobrý den.", "Čtyři ☃ a ∑ pět.", "Přišel domů.")},
                      skip_unreadable=True)
    assert report["status"] == "ok"
    assert report["parts"]["p1"]["skipped_blocks"] == [
        {"part": "p1", "line": 1, "block": 1, "codepoints": ["U+2248"]},
        {"part": "p1", "line": 5, "block": 3, "codepoints": ["U+2603", "U+2211"]}]
    rows = units_of(out)
    assert [r["text"] for r in rows] == ["Dobrý den.", "Přišel domů."]
    assert [r["source"]["line_start"] for r in rows] == [3, 7]


def test_failure_path_leaves_no_log_record_to_the_last_resort_handler(tmp_path, monkeypatch):
    unhandled = []

    class Count(logging.Handler):
        def emit(self, record):
            unhandled.append(record)

    monkeypatch.setattr(logging.getLogger(), "handlers", [])
    monkeypatch.setattr(logging, "lastResort", Count(logging.WARNING))
    with pytest.raises(TextStageError):
        run(tmp_path, {"p1": book("# 5. Návrat domů", "Čtyři ☃ pět.")})
    assert unhandled == []


def test_failures_of_all_parts_are_collected(tmp_path):
    with pytest.raises(TextStageError) as info:
        run(tmp_path, {"p1": book("Jedna ≈ věta."), "p2": book("Druhá ☃ věta.")})
    assert [f["part"] for f in info.value.failures] == ["p1", "p2"]


def test_unparsed_error_message_is_reported(tmp_path, monkeypatch):
    class Fake:
        abbreviations = {}

        def __init__(self, *a):
            pass

        def normalize(self, text):
            raise ValueError("an unexpected wording")

    monkeypatch.setattr(stage, "TextNormalizer", Fake)
    with pytest.raises(TextStageError) as info:
        run(tmp_path, {"p1": book("Jedna věta.")})
    assert info.value.failures == [{"part": "p1", "line": 1, "block": 1, "codepoints": ["unparsed"]}]


def test_failing_character_is_parsed_from_both_messages(normalizer):
    for text in ("Dva ≈ tři.", "Dva <en>≈</en> tři."):
        with pytest.raises(ValueError) as info:
            normalizer.normalize(text)
        assert stage._failing_character(str(info.value)) == "≈"
    assert stage._failing_character("no quotes") is None


# ---- units -----------------------------------------------------------------------------------------------------

BASIC = book("Kočka seděla na střeše domu. Slunce zapadalo za kopce.", "„Kam jdeš?“ zeptal se tiše. „Domů,“ řekl chlapec.",
             "Ráno pršelo.")


def test_units_schema_ids_and_norm_file(tmp_path, normalizer):
    report, out = run(tmp_path, {"p1": BASIC})
    raw = (out / "p1.units.jsonl").read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n") and "ě".encode() in raw
    units = units_of(out)
    assert [list(u) for u in units] == [KEYS] * len(units)
    assert [u["id"] for u in units] == [f"p1-{i:05d}" for i in range(len(units))]
    assert [u["index"] for u in units] == list(range(len(units)))
    assert all(u["part"] == "p1" for u in units)
    assert all(u["tags"] == sorted(u["tags"]) for u in units)
    assert all(list(u["source"]) == ["line_start", "line_end"] for u in units)
    assert (out / "p1.norm.txt").read_text(encoding="utf-8") == normalizer.normalize(BASIC)
    first = units[0]
    assert first["text"] == "Kočka seděla na střeše domu."
    assert first["sentences"] == [{"text": first["text"], "boundary": "mid"}]
    assert first["est_s"] == round(first["est_s"], 3)
    assert first["source"] == {"line_start": 1, "line_end": 1}
    assert at_line(units, 3)[0]["source"] == {"line_start": 3, "line_end": 3}
    assert [u["paragraph"] for u in units] == [0, 0, 1, 1, 2]
    assert report["parts"]["p1"]["units"] == len(units) == 5


def test_second_run_is_byte_identical(tmp_path):
    cfg = {"p1": BASIC, "p2": book("Druhá část začíná zde.")}
    r1, o1 = run(tmp_path, cfg, work="w1")
    r2, o2 = run(tmp_path, cfg, work="w2")
    for name in ("p1.norm.txt", "p1.units.jsonl", "p2.units.jsonl", "review_readings.csv"):
        assert (o1 / name).read_bytes() == (o2 / name).read_bytes()

    def strip(r):
        r = json.loads(json.dumps(r))
        r["config"].pop("work_dir")
        r["total"]["timing"] = None
        for p in r["parts"].values():
            p["timing"] = None
        return r

    assert strip(r1) == strip(r2)
    assert list(r1["parts"]) == ["p1", "p2"]


def test_parts_default_to_natural_order(tmp_path):
    report, _ = run(tmp_path, {"a10": book("Desátá část."), "a2": book("Druhá část."), "a1": book("První část.")})
    assert list(report["parts"]) == ["a1", "a2", "a10"]


def test_words_and_word_spans(tmp_path):
    text = book("Řekl <en>Okay now</en> a potom Tom's kolo zmizelo.")
    _, out = run(tmp_path, {"p1": text})
    u = units_of(out)[0]
    assert u["words"] == ["Řekl", "Okay", "now", "a", "potom", "Tom's", "kolo", "zmizelo"]
    assert [u["text"][s:e] for s, e in u["word_spans"]] == u["words"]
    assert "en_span" in u["tags"]
    assert "en" not in u["words"]


# ---- tags ------------------------------------------------------------------------------------------------------

TAG_PARAGRAPHS = [
    "Kočka seděla na střeše domu.",
    "Kam jdeš dnes večer?",
    "Pojď sem hned teď!",
    "A potom přišlo dlouhé ticho…",
    "A potom přišlo dlouhé ticho...",
    "Konec věty bez tečky",
    "„Ne!“",
    "„Pojď sem,“ řekl otec tiše.",
    "Otec řekl nahlas: „Pojď sem hned.“",
    "Otec řekl: „Pojď sem hned,“ a usmál se.",
    "„Pojď sem,“ řekl otec, „je již pozdě.“",
    "Otec řekl „ano“ a potom řekl „ne“ a odešel.",
    '"Pojď sem," řekl otec.',
    "„Ne. Nikdy. Rozumíš?“ zašeptala.",
    "— Pojď sem hned, řekl otec.",
    "Přišel domů (a byl velmi unaven) dost pozdě.",
    "(Nikdo tomu vůbec nevěřil.)",
    "Za první, nikdo nepřišel domů. Po druhé, nikdo nezavolal.",
    "Otec řekl takto: za první, nikdo nepřišel; po druhé, nikdo nezavolal.",
    "Slovo <en>hello</en> a slovo <sk>ahoj</sk> a slovo <cs>ahoj</cs> znamenají totéž.",
    "# Nadpis kapitoly",
    "Text po nadpisu.",
]

TAG_EXPECT = {
    0: ({"statement", "para_end"}, {"question", "quote_only", "joined", "heading"}),
    1: ({"question", "has_question"}, {"statement", "has_exclamation"}),
    2: ({"exclamation", "has_exclamation"}, {"has_question"}),
    3: ({"ellipsis"}, {"statement"}),
    4: ({"ellipsis"}, {"statement"}),
    5: ({"unterminated"}, {"statement"}),
    6: ({"quote_only", "exclamation", "very_short"}, {"quote_then_narration"}),
    7: ({"quote_then_narration"}, {"quote_only", "multi_quote"}),
    8: ({"narration_then_quote"}, {"quote_only"}),
    9: ({"narration_quote_narration"}, {"multi_quote"}),
    10: ({"multi_quote", "quote_narration_quote"}, {"quote_only"}),
    11: ({"multi_quote"}, {"quote_narration_quote"}),
    12: ({"quote_then_narration"}, {"quote_only"}),
    13: ({"joined", "multi_sentence_quote", "quote_then_narration", "has_question", "statement"}, {"clause_cut"}),
    14: ({"dash_dialogue"}, set()),
    15: ({"parenthesis"}, {"parenthesis_whole"}),
    16: ({"parenthesis", "parenthesis_whole"}, set()),
    18: ({"list_item"}, set()),
    19: ({"en_span", "sk_span", "cs_span", "chapter_end"}, set()),
    20: ({"heading", "para_end"}, set()),
    21: ({"para_end"}, {"heading", "chapter_end"}),
}


@pytest.fixture(scope="module")
def tag_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tags")
    report, out = run(tmp, {"p1": book(*TAG_PARAGRAPHS)})
    return report, units_of(out)


@pytest.mark.parametrize("k", sorted(TAG_EXPECT))
def test_tags_of_paragraph(tag_run, k):
    wanted, forbidden = TAG_EXPECT[k]
    found = at_line(tag_run[1], 2 * k + 1)
    assert len(found) == 1, found
    tags = set(found[0]["tags"])
    assert wanted <= tags, (found[0]["text"], tags)
    assert not (forbidden & tags), (found[0]["text"], tags)


def test_list_item_sentences_and_mid(tag_run):
    first, second = at_line(tag_run[1], 2 * 17 + 1)
    assert {"list_item", "mid"} <= set(first["tags"])
    assert {"list_item", "para_end"} <= set(second["tags"])


def test_every_unit_has_exactly_one_end_mark(tag_run):
    marks = {"statement", "question", "exclamation", "ellipsis", "unterminated"}
    heading_free = [u for u in tag_run[1]]
    assert all(len(marks & set(u["tags"])) == 1 for u in heading_free)


def test_long_is_above_the_90th_percentile(tag_run):
    units = tag_run[1]
    est = sorted(u["est_s"] for u in units)
    p90 = est[math.ceil(0.9 * len(est)) - 1]
    assert {u["id"] for u in units if "long" in u["tags"]} == {u["id"] for u in units if u["est_s"] > p90}
    assert any("long" in u["tags"] for u in units)


CAP_PARAGRAPHS = [
    "„Přišel jsem včera domů. Potom jsem šel spát. Ráno jsem odešel.“ řekl.",
    "„Přišel jsem domů.",
    "Přišel jsem včera domů, potom jsem šel spát, a ráno jsem odešel.",
    "Dlouhá věta bez jakékoliv čárky pokračuje stále dál.",
    "Ano.",
]


@pytest.fixture(scope="module")
def cap_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cap")
    report, out = run(tmp, {"p1": book(*CAP_PARAGRAPHS)}, max_est_s=3.0)
    return report, units_of(out)


def test_cap_tags(cap_run):
    units = cap_run[1]
    quote = at_line(units, 1)
    assert [u["boundary"] for u in quote] == ["mid", "mid", "para_end"]
    assert {"open_cut", "mid"} <= set(quote[0]["tags"]) and "open_para_end" not in quote[0]["tags"]
    assert "open_cut" in quote[1]["tags"] and "open_cut" not in quote[2]["tags"]
    unclosed = at_line(units, 3)
    assert {"open_para_end", "para_end"} <= set(unclosed[0]["tags"]) and "open_cut" not in unclosed[0]["tags"]
    cuts = at_line(units, 5)
    assert len(cuts) == 2 and all("clause_cut" in u["tags"] for u in cuts)
    assert cuts[0]["boundary"] == "mid" and "mid" in cuts[0]["tags"]
    assert len(cuts[0]["sentences"]) == 1 and cuts[0]["sentences"][0]["text"] == cuts[0]["text"]
    over = at_line(units, 7)
    assert len(over) == 1 and {"over_cap", "statement"} <= set(over[0]["tags"])
    assert "very_short" in at_line(units, 9)[0]["tags"]
    assert cap_run[0]["parts"]["p1"]["units_over_cap"] == 1
    assert cap_run[0]["parts"]["p1"]["clause_cuts"] == 1


DIFF_PARAGRAPHS = [
    "Dnes přišel domů velmi pozdě.",
    "Přišel v 8:30 domů.",
    "Přišel 3.5.2020 domů.",
    "Měl jich 15 doma.",
    "Karel IV. zemřel včera.",
    "Přišel např. včera domů.",
]


def test_diff_tags(tmp_path):
    report, out = run(tmp_path, {"p1": book(*DIFF_PARAGRAPHS)})
    units = units_of(out)
    diff = {"number", "date", "time", "roman", "abbreviation"}
    got = [diff & set(at_line(units, 2 * k + 1)[0]["tags"]) for k in range(len(DIFF_PARAGRAPHS))]
    assert got == [set(), {"time"}, {"date"}, {"number"}, {"roman"}, {"abbreviation"}]
    assert report["parts"]["p1"]["diff_unmapped_blocks"] == 0
    assert report["parts"]["p1"]["tags"]["number"] == 1


def test_classify_tags_only_real_abbreviations(normalizer):
    known = normalizer.abbreviations
    assert stage._classify("např.", known) == "abbreviation"
    assert stage._classify("(Např.", known) == "abbreviation"
    assert stage._classify("doma.", known) is None
    assert stage._classify("pes", known) is None
    assert stage._classify("15", known) == "number"


def test_diff_block_with_unmatched_word_total_is_skipped_and_counted(tmp_path):
    report, _ = run(tmp_path, {"p1": book("Přišel domů.", "* * *", "Odešel pryč.")})
    assert report["parts"]["p1"]["diff_unmapped_blocks"] == 1
    assert report["total"]["diff_unmapped_blocks"] == 1


# ---- placement ---------------------------------------------------------------------------------------------------

def test_part_end_labels(tmp_path):
    text = book("První odstavec zde.", "Druhý odstavec zde.")
    parts = {n: text for n in ("p1", "p2", "p3", "p4", "p5")}
    placement = write_placement(tmp_path, {
        "p1": place(finish=end(inside=True)),
        "p2": place(finish=end("paragraph")),
        "p3": place(finish=end("heading")),
        "p4": place(finish=end("none")),
    })
    _, out = run(tmp_path, parts, placement=placement)
    last = {p: units_of(out, p)[-1] for p in parts}
    assert [last[p]["boundary"] for p in parts] == ["mid", "para_end", "chapter_end", "chapter_end", "para_end"]
    assert [last[p]["sentences"][-1]["boundary"] for p in parts] == ["mid", "para_end", "chapter_end", "chapter_end",
                                                                    "para_end"]
    assert "mid" in last["p1"]["tags"] and "chapter_end" in last["p3"]["tags"] and "para_end" in last["p5"]["tags"]
    assert units_of(out, "p1")[0]["boundary"] == "para_end"


def test_no_placement_ends_with_para_end(tmp_path):
    _, out = run(tmp_path, {"p1": book("Jediný odstavec zde.")})
    assert units_of(out)[-1]["boundary"] == "para_end"


def test_part_fragment_tags(tmp_path):
    text = book("První odstavec zde.", "Druhý odstavec zde.", "Třetí odstavec zde.")
    placement = write_placement(tmp_path, {
        "p1": place(start=(True, True), finish=end(inside=True, mid=True)),
        "p2": place(start=(True, False), finish=end(inside=True)),
        "p3": place(start=(False, True)),
    })
    report, out = run(tmp_path, {"p1": text, "p2": text, "p3": text}, placement=placement, use_part_fragments=True)
    assert ["part_fragment" in u["tags"] for u in units_of(out, "p1")] == [True, False, True]
    assert not any("part_fragment" in u["tags"] for u in units_of(out, "p2"))
    assert ["part_fragment" in u["tags"] for u in units_of(out, "p3")] == [True, False, False]
    assert report["parts"]["p1"]["fragments"] == 2 and report["total"]["fragments"] == 3
    assert report["config"]["use_part_fragments"] is True


def test_chapter_starts_applied_and_not_applied(tmp_path):
    text = book("Úvod zde.", "Druhý odstavec zde.", "Třetí odstavec zde.").replace("Třetí", "Řádek před\nTřetí")
    lines = text.split("\n")
    assert lines[:6] == ["Úvod zde.", "", "Druhý odstavec zde.", "", "Řádek před", "Třetí odstavec zde."]
    placement = write_placement(tmp_path, {"p1": place(starts=[1, 3, 6, 99])})
    report, out = run(tmp_path, {"p1": text}, placement=placement)
    norm = (out / "p1.norm.txt").read_text(encoding="utf-8").split("\n")
    assert norm[1] == "# " and norm[3] == ""
    cs = report["parts"]["p1"]["chapter_starts"]
    assert cs["applied"] == 1
    assert cs["not_applied"] == [{"line": 1, "reason": "first_line"}, {"line": 6, "reason": "previous_line_not_blank"},
                                 {"line": 99, "reason": "line_out_of_range"}]
    units = units_of(out)
    assert [u["chapter"] for u in units] == [0, 1, 1]
    assert units[0]["boundary"] == "chapter_end"
    assert report["total"]["chapter_starts"]["applied"] == 1


def test_part_missing_from_placement_has_no_starts(tmp_path):
    placement = write_placement(tmp_path, {"p1": place(starts=[3])})
    report, out = run(tmp_path, {"p2": book("Úvod zde.", "Druhý odstavec zde.")}, placement=placement)
    assert report["parts"]["p2"]["chapter_starts"] == {"applied": 0, "not_applied": []}


# ---- report ----------------------------------------------------------------------------------------------------

def test_report_keys_and_percentiles(tmp_path):
    report, out = run(tmp_path, {"p1": BASIC, "p2": book("Ještě jedna věta zde.")})
    assert report["status"] == "ok"
    p = report["parts"]["p1"]
    for key in ("paragraphs", "headings", "sentences", "units", "joined_units", "clause_cuts", "est_s",
                "units_over_cap", "words", "fragments", "tags", "readings_by_reason", "abbreviation_tokens",
                "diff_unmapped_blocks", "units_not_nfc", "format", "chapter_starts", "timing"):
        assert key in p and key in report["total"], key
    assert p["paragraphs"] == 3 and p["sentences"] == 5 and p["headings"] == 0 and p["units_not_nfc"] == 0
    est = sorted(u["est_s"] for u in units_of(out))
    assert p["est_s"] == {"p50": est[math.ceil(0.5 * len(est)) - 1], "p95": est[math.ceil(0.95 * len(est)) - 1],
                          "max": est[-1]}
    assert report["total"]["units"] == report["parts"]["p1"]["units"] + report["parts"]["p2"]["units"]
    assert set(p["timing"]) == {"normalise_s", "peak_rss_mib"} == set(report["total"]["timing"])
    assert json.loads((out / "report.json").read_text(encoding="utf-8")) == report
    assert (out / "report.md").read_text(encoding="utf-8").startswith("#")


def test_abbreviation_tokens_counted(tmp_path):
    text = book("Přišel nz. včera a potom šel ab. domů. Hrál turnaj ATP. pak šel spát. Zase ATP. pak spal.")
    report, _ = run(tmp_path, {"p1": text})
    assert report["parts"]["p1"]["abbreviation_tokens"] == {"atp": 2}
    assert report["total"]["abbreviation_tokens"] == {"atp": 2}


def test_report_holds_no_body_words(tmp_path):
    body = book("Návrat domů", "Kočka seděla na střeše domu. Slunce zapadalo za kopce.", "KAPITOLA PRVNÍ",
                "Poslední řádek bez konce.")
    report, out = run(tmp_path, {"p1": body})
    blob = (out / "report.json").read_text(encoding="utf-8") + (out / "report.md").read_text(encoding="utf-8")
    blob = unicodedata.normalize("NFC", blob).lower()
    words = {w.lower() for w in re.findall(r"[^\W\d_]{5,}", body)}
    assert words
    assert not [w for w in words if w in blob]
    assert "aaaaaa aaaa" in blob


def test_text_dir_is_not_written(tmp_path):
    cfg = make(tmp_path, {"p1": BASIC})
    before = sorted(os.listdir(cfg.text_dir))
    run_text_stage(cfg)
    assert sorted(os.listdir(cfg.text_dir)) == before


def test_import_leaves_torch_and_stanza_out():
    code = ("import sys; sys.path.insert(0, %r); import src.data.prepare.text; "
            "bad = [m for m in ('torch', 'stanza') if m in sys.modules]; assert not bad, bad" % REPO)
    done = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


# ---- lexicon and comparison with the rewrite ------------------------------------------------------------------------

def write_lexicon_file(tmp_path, rows):
    path = tmp_path / "lexicon.tsv"
    path.write_text("book_form\tspoken_form\tapply\n" + "".join(f"{a}\t{b}\t{c}\n" for a, b, c in rows), encoding="utf-8")
    return str(path)


def write_rewrite(tmp_path, parts):
    rewrite = tmp_path / "rewrite"
    rewrite.mkdir(exist_ok=True)
    for name, data in parts.items():
        (rewrite / f"{name}.txt.orig").write_bytes(data.encode("utf-8"))
    return str(rewrite)


def test_lexicon_replaces_whole_words_and_counts_them_per_part(tmp_path):
    lexicon = write_lexicon_file(tmp_path, [("Jack", "Džek", "y"), ("Pat", "Pet", "n")])
    parts = {"p1": book("Jack přišel domů. Patrik přišel také. Pat přišel.", "Potom šel Jack pryč."),
             "p2": book("Nikdo nepřišel.")}
    report, out = run(tmp_path, parts, lexicon=lexicon)
    assert (report["parts"]["p1"]["lexicon_replacements"], report["parts"]["p2"]["lexicon_replacements"]) == (2, 0)
    assert report["total"]["lexicon_replacements"] == 2
    norm = (out / "p1.norm.txt").read_text(encoding="utf-8")
    assert norm == "Džek přišel domů. Patrik přišel také. Pat přišel.\n\nPotom šel Džek pryč.\n"
    assert (tmp_path / "in" / "p1.txt.orig").read_text(encoding="utf-8").startswith("Jack")


def test_no_lexicon_means_no_lexicon_keys(tmp_path):
    report, _ = run(tmp_path, {"p1": BASIC})
    assert "lexicon_replacements" not in report["parts"]["p1"] and "comparison" not in report["parts"]["p1"]


def test_slash_fails_without_a_symbol_pair_and_passes_with_one(tmp_path):
    parts = {"p1": book("Volba A / B doma.")}
    with pytest.raises(TextStageError) as info:
        run(tmp_path, parts)
    assert info.value.failures[0]["codepoints"] == ["U+002F"]
    lexicon = write_lexicon_file(tmp_path, [("/", "lomeno", "y")])
    report, out = run(tmp_path, parts, work="work2", lexicon=lexicon)
    assert report["parts"]["p1"]["lexicon_replacements"] == 1
    assert (out / "p1.norm.txt").read_text(encoding="utf-8") == "Volba A lomeno B doma.\n"


def test_comparison_counts_the_exact_share_before_and_after_the_lexicon(tmp_path):
    book_text = book("Jack přišel včera pozdě domů. Potom odešel zase pryč.")
    rewrite = write_rewrite(tmp_path, {"p1": "Džek přišel včera pozdě domů. Potom úplně odešel zase pryč.\r\n"})
    lexicon = write_lexicon_file(tmp_path, [("Jack", "Džek", "y")])
    report, out = run(tmp_path, {"p1": book_text}, lexicon=lexicon, compare_dir=rewrite)
    cmp = report["parts"]["p1"]["comparison"]
    assert cmp["rewrite_words"] == 10
    assert (cmp["matched_before"], cmp["matched_after"]) == (8, 9)
    assert cmp["share_before"] == pytest.approx(0.8) and cmp["share_after"] == pytest.approx(0.9)
    assert cmp["disagreements"] == {"insertion": 1}
    assert report["total"]["comparison"] == cmp
    with open(out / "review_disagreements.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    assert rows == [["part", "paragraph", "category", "book_form", "normalizer_reading", "rewrite_form"],
                    ["p1", "1", "insertion", "", "", "úplně"]]


def test_comparison_of_a_number_reading_and_a_respelling_the_lexicon_lacks(tmp_path):
    rewrite = write_rewrite(tmp_path, {"p1": "Měl patnácti psů doma. Mike přišel včera pozdě domů.\r\n"})
    _, out = run(tmp_path, {"p1": book("Měl 15 psů doma. Majk přišel včera pozdě domů.")}, compare_dir=rewrite)
    with open(out / "review_disagreements.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))[1:]
    assert rows == [["p1", "1", "number", "15", "patnáct", "patnácti"],
                    ["p1", "1", "respelling", "Majk", "Majk", "Mike"]]


def test_comparison_agrees_when_the_normalizer_reads_as_the_rewrite(tmp_path):
    rewrite = write_rewrite(tmp_path, {"p1": "Měl patnáct psů doma.\r\n"})
    report, out = run(tmp_path, {"p1": book("Měl 15 psů doma.")}, compare_dir=rewrite)
    assert report["parts"]["p1"]["comparison"]["disagreements"] == {}
    assert report["parts"]["p1"]["comparison"]["share_after"] == 1.0
    rows = list(csv.reader((out / "review_disagreements.csv").read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 1
