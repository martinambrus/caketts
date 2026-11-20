import io
import json
import os
import re
import subprocess
import sys
import zipfile

import pytest

from src.data.prepare.source import (
    EpubConfig,
    PartSpec,
    SourceConfig,
    read_epub,
    run_source_stage,
    to_book_text,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SYL = ["ba", "ke", "di", "fo", "gu", "hi", "jo", "lu"]


def V(i):
    return SYL[i % 8] + SYL[(i // 8) % 8] + SYL[(i // 64) % 8]


def toks(a, b, sub=None, sent=15):
    sub = sub or {}
    out = []
    for n, i in enumerate(range(a, b)):
        t = sub.get(i, V(i))
        if n % sent == 0:
            t = t.capitalize()
        if n % sent == sent - 1 or i == b - 1:
            t += "."
        out.append(t)
    return out


def xhtml(*blocks):
    body = "".join("<%s>%s</%s>\n" % (t, x, t) for t, x in blocks)
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>t</title></head>'
        "<body>\n" + body + "</body></html>"
    ).encode("utf-8")


def make_epub(files, spine=None, with_container=True, opf_dir="OEBPS"):
    buf = io.BytesIO()
    names = list(files)
    spine = spine if spine is not None else names
    manifest = "".join(
        '<item id="i%d" href="text/%s" media-type="application/xhtml+xml"/>' % (n, name)
        for n, name in enumerate(names)
    )
    manifest += '<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>'
    refs = "".join('<itemref idref="i%d"/>' % names.index(name) for name in spine)
    opf = (
        '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="2.0">'
        "<manifest>%s</manifest><spine toc=\"ncx\">%s</spine></package>" % (manifest, refs)
    )
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        if with_container:
            z.writestr(
                "META-INF/container.xml",
                '<?xml version="1.0"?><container version="1.0" '
                'xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
                '<rootfile full-path="%s/content.opf" media-type="application/oebps-package+xml"/>'
                "</rootfiles></container>" % opf_dir,
            )
        z.writestr("%s/content.opf" % opf_dir, opf)
        z.writestr("%s/toc.ncx" % opf_dir, "<ncx/>")
        for name, data in files.items():
            z.writestr("%s/text/%s" % (opf_dir, name), data)
    return buf.getvalue()


def write_epub(tmp_path, files, **kw):
    path = tmp_path / "book.epub"
    path.write_bytes(make_epub(files, **kw))
    return str(path)


def read(tmp_path, files, **kw):
    cfg_kw = {k: kw.pop(k) for k in list(kw) if k in EpubConfig.__dataclass_fields__ and k != "path"}
    return read_epub(EpubConfig(write_epub(tmp_path, files, **kw), **cfg_kw))


def test_spine_order_differs_from_file_order(tmp_path):
    files = {
        "a.xhtml": xhtml(("p", "first in file")),
        "b.xhtml": xhtml(("p", "second in file")),
        "c.xhtml": xhtml(("p", "third in file")),
    }
    book = read(tmp_path, files, spine=["c.xhtml", "a.xhtml", "b.xhtml"])
    assert [i.text for i in book.items] == ["third in file", "first in file", "second in file"]
    assert [i.index for i in book.items] == [0, 1, 2]
    assert [i.file for i in book.items][0].endswith("c.xhtml")


def test_container_fallback_to_only_opf(tmp_path):
    files = {"a.xhtml": xhtml(("p", "alone"))}
    book = read(tmp_path, files, with_container=False)
    assert [i.text for i in book.items] == ["alone"]


def test_headings_and_chapter_regex(tmp_path):
    files = {
        "a.xhtml": xhtml(
            ("h2", "Part one"),
            ("h3", "Sub"),
            ("p", "7 / Some title"),
            ("p", "7/Other"),
            ("p", "plain 7 / not a head"),
            ("p", "“Quoted” text"),
        )
    }
    book = read(tmp_path, files)
    kinds = [(i.kind, i.tag, i.text) for i in book.items]
    assert kinds == [
        ("heading", "h2", "Part one"),
        ("heading", "h3", "Sub"),
        ("chapter", "p", "7. Some title"),
        ("chapter", "p", "7. Other"),
        ("body", "p", "plain 7 / not a head"),
        ("body", "p", "“Quoted” text"),
    ]
    assert to_book_text(book.items) == (
        "# Part one\n\n# Sub\n\n# 7. Some title\n\n# 7. Other\n\nplain 7 / not a head\n\n"
        "“Quoted” text\n"
    )


def test_changed_chapter_format_and_heading_tags(tmp_path):
    files = {"a.xhtml": xhtml(("h1", "Top"), ("h2", "Two"), ("p", "12 / Name"))}
    book = read(tmp_path, files, chapter_format="Chapter {num}: {title}", heading_tags=("h1",))
    assert [(i.kind, i.text) for i in book.items] == [
        ("heading", "Top"),
        ("body", "Two"),
        ("chapter", "Chapter 12: Name"),
    ]


def test_tab_paragraph_excluded_and_listed(tmp_path):
    files = {"a.xhtml": xhtml(("p", "before"), ("p", "col1\tcol2"), ("p", "after"))}
    book = read(tmp_path, files)
    assert [i.table for i in book.items] == [False, True, False]
    assert [i.excluded for i in book.items] == [False, True, False]
    assert book.report["table_items"] == [1]
    assert to_book_text(book.items) == "before\n\nafter\n"
    assert "col1" not in to_book_text(book.items)


def test_tab_paragraph_kept_with_flag(tmp_path):
    files = {"a.xhtml": xhtml(("p", "before"), ("p", "col1\tcol2"), ("p", "after"))}
    book = read(tmp_path, files, keep_tables=True)
    assert [i.table for i in book.items] == [False, True, False]
    assert [i.excluded for i in book.items] == [False, False, False]
    assert to_book_text(book.items) == "before\n\ncol1\tcol2\n\nafter\n"


def test_empty_and_nbsp_blocks_skipped(tmp_path):
    files = {
        "a.xhtml": xhtml(("p", "one"), ("p", ""), ("p", " "), ("p", " \n "), ("p", "two x"))
    }
    book = read(tmp_path, files)
    assert [i.text for i in book.items] == ["one", "two x"]
    assert [i.index for i in book.items] == [0, 1]
    assert book.report["empty_skipped"] == 3


def test_entities_inline_tags_br_script_and_outside_text(tmp_path):
    raw = (
        '<html><head><style>p { color: red }</style><script>var x = "p";</script></head><body>'
        "stray words<p>Tom &amp; Jerry <i>ran</i>\n  away<br/>fast&#33; &eacute;</p>"
        "<h2>A <b>bold</b>\n title</h2>"
        "<div>more stray</div></body></html>"
    ).encode("utf-8")
    book = read(tmp_path, {"a.xhtml": raw})
    assert [(i.kind, i.text) for i in book.items] == [
        ("body", "Tom & Jerry ran away fast! é"),
        ("heading", "A bold title"),
    ]
    outside = book.report["outside_text_chars"]
    assert sum(outside.values()) == len("straywords") + len("morestray")
    assert len(outside) == 1


def test_nfc_normalisation(tmp_path):
    files = {"a.xhtml": xhtml(("p", "éx"))}
    assert read(tmp_path, files).items[0].text == "éx"


def test_replacements_applied_and_counted(tmp_path):
    files = {"a.xhtml": xhtml(("p", "a • b • c"), ("p", "d | e"), ("p", "x\ty •"), ("p", "clean"))}
    book = read(tmp_path, files, replacements={"•": "-", "|": ","})
    assert [i.text for i in book.items if not i.excluded] == ["a - b - c", "d , e", "clean"]
    rep = book.report["replacements"]
    assert rep["•"] == {"count": 2, "items": [0]}
    assert rep["|"] == {"count": 1, "items": [1]}


def test_to_book_text_empty(tmp_path):
    assert to_book_text([]) == ""


def vocab_in(text):
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) >= 5}


P = {}
P[1] = toks(0, 30, sub={3: "7"})
P[2] = toks(30, 60)
P[3] = toks(60, 90)
P[5] = toks(90, 120)
P[7] = toks(120, 150)
P[8] = toks(150, 180)
P[9] = toks(180, 210)
P["d1"] = toks(210, 240)
P["d2"] = toks(240, 270)
P[12] = toks(270, 300)
P[13] = toks(300, 330)
P1_ORIG = toks(0, 30, sub={3: "sedmicka"})
P5_ORIG = toks(90, 120, sub={119: "xnine"})
P12_ORIG = toks(270, 300, sub={285: "xsub"})


def sentence(tokens):
    return " ".join(tokens)


MAIN_FILES = {
    "c1.xhtml": xhtml(
        ("h2", "Kapxo"), ("p", sentence(P[1])), ("p", sentence(P[2])),
        ("p", sentence(P[3])), ("p", "2 / Titlexx"), ("p", sentence(P[5])), ("p", "colx\tcoly"),
        ("p", " "),
    ),
    "c2.xhtml": xhtml(
        ("p", sentence(P[7])), ("p", sentence(P[8])), ("h3", "Sechead"), ("p", sentence(P[9])),
        ("p", sentence(P["d1"])), ("h3", "Nohead"),
    ),
    "c3.xhtml": xhtml(
        ("p", sentence(P["d2"])), ("p", sentence(P[12])), ("h3", "Tailhead"), ("p", sentence(P[13])),
    ),
}

MAIN_ORIG = {
    "a1": "\r\n".join(["Kapxo", "", sentence(P1_ORIG), "", sentence(P[2][:8])]) + "\r\n",
    "a2": "\r\n".join(
        [sentence(P[2][8:]), "", sentence(P[3]), "", "Titlexx", "", sentence(P5_ORIG)]
    ) + "\r\n",
    "a3": "\n".join(["%s Sechead" % sentence(P[8]), "", sentence(P[9]), "", sentence(P12_ORIG)]) + "\n",
}


def write_orig(tmp_path, parts, suffix=".txt.orig"):
    d = tmp_path / "raw"
    d.mkdir()
    for name, text in parts.items():
        (d / (name + suffix)).write_bytes(text.encode("utf-8"))
    return str(d)


@pytest.fixture
def main_run(tmp_path):
    epub = write_epub(tmp_path, MAIN_FILES)
    orig = write_orig(tmp_path, MAIN_ORIG)
    work = str(tmp_path / "work")
    cfg = SourceConfig(epub=EpubConfig(epub), work_dir=work, orig_dir=orig)
    return cfg, run_source_stage(cfg), work


def rd(work, rel):
    with open(os.path.join(work, "source", rel), encoding="utf-8", newline="") as f:
        return f.read()


def test_main_item_indices(main_run):
    cfg, report, work = main_run
    book = read_epub(cfg.epub)
    kinds = [(i.index, i.kind, i.excluded) for i in book.items]
    assert kinds[:8] == [
        (0, "heading", False), (1, "body", False), (2, "body", False), (3, "body", False),
        (4, "chapter", False), (5, "body", False), (6, "body", True), (7, "body", False),
    ]
    assert len(book.items) == 17


def test_main_part_texts(main_run):
    cfg, report, work = main_run
    assert rd(work, "epub/a1.txt") == "# Kapxo\n\n%s\n\n%s\n" % (sentence(P[1]), sentence(P[2][:8]))
    assert rd(work, "epub/a2.txt") == "%s\n\n%s\n\n# 2. Titlexx\n\n%s\n" % (
        sentence(P[2][8:]), sentence(P[3]), sentence(P[5]))
    a3 = rd(work, "epub/a3.txt")
    assert a3 == "\n\n".join(
        [sentence(P[8]), "# Sechead", sentence(P[9]), sentence(P["d1"]), "# Nohead",
         sentence(P["d2"]), sentence(P[12])]
    ) + "\n"
    assert "colx" not in a3 and "Tailhead" not in a3 and "\r" not in a3


def test_main_edges_and_flags(main_run):
    cfg, report, work = main_run
    p = report["parts"]
    assert list(p) == ["a1", "a2", "a3"]
    assert p["a1"]["start"]["item"] == 0 and p["a1"]["start"]["word_offset"] == 0
    assert p["a1"]["start"]["score"] == 1.0 and p["a1"]["start"]["exact"] is True
    assert p["a1"]["end"]["item"] == 2 and p["a1"]["end"]["word_offset"] == 7
    assert p["a1"]["end"]["score"] == 1.0
    assert p["a2"]["start"]["item"] == 2 and p["a2"]["start"]["word_offset"] == 8
    assert p["a2"]["end"]["item"] == 5 and p["a2"]["end"]["word_offset"] == 29
    assert p["a2"]["end"]["exact"] is False
    assert p["a2"]["end"]["score"] == pytest.approx(29 / 30, abs=1e-4)
    assert p["a3"]["start"]["item"] == 8 and p["a3"]["start"]["exact"] is True
    assert p["a3"]["end"]["item"] == 14 and p["a3"]["end"]["score"] == pytest.approx(29 / 30, abs=1e-4)
    assert p["a3"]["end"]["exact"] is True
    pl = json.loads(rd(work, "epub/placement.json"))
    assert pl["source"] == "epub"
    assert set(pl) == {"source", "parts"}
    assert pl["parts"] == {
        "a1": {"chapter_starts": [],
               "start": {"inside_paragraph": False, "mid_sentence": False},
               "end": {"inside_paragraph": True, "mid_sentence": True, "next": "paragraph"}},
        "a2": {"chapter_starts": [],
               "start": {"inside_paragraph": True, "mid_sentence": True},
               "end": {"inside_paragraph": False, "mid_sentence": False, "next": "paragraph"}},
        "a3": {"chapter_starts": [],
               "start": {"inside_paragraph": False, "mid_sentence": False},
               "end": {"inside_paragraph": False, "mid_sentence": False, "next": "heading"}},
    }


def test_main_gaps(main_run):
    cfg, report, work = main_run
    assert report["gaps"] == [
        {"after": "a2", "before": "a3", "words_between": 30,
         "items": [{"index": 6, "kind": "body", "excluded": True, "words": 2},
                   {"index": 7, "kind": "body", "excluded": False, "words": 30}]},
        {"after": "a3", "before": None, "words_between": 31,
         "items": [{"index": 15, "kind": "heading", "excluded": False, "words": 1},
                   {"index": 16, "kind": "body", "excluded": False, "words": 30}]},
    ]


def test_main_chapter_placement(main_run):
    cfg, report, work = main_run
    pl = json.loads(rd(work, "orig/placement.json"))
    assert pl["source"] == "orig"
    assert {k: v["chapter_starts"] for k, v in pl["parts"].items()} == {"a1": [1], "a2": [5], "a3": []}
    epl = json.loads(rd(work, "epub/placement.json"))
    for k in pl["parts"]:
        assert pl["parts"][k]["start"] == epl["parts"][k]["start"]
        assert pl["parts"][k]["end"] == epl["parts"][k]["end"]
    rows = report["chapters"]["rows"]
    got = [(r["index"], r["kind"], r["number"], r["epub_part"], r["placed"], r["reason"]) for r in rows]
    assert got == [
        (0, "h2", None, "a1", {"part": "a1", "line": 1}, None),
        (4, "chapter", "2", "a2", {"part": "a2", "line": 5}, None),
        (9, "h3", None, "a3", None, "inside_line"),
        (12, "h3", None, "a3", None, "no_match"),
        (15, "h3", None, None, None, "outside_parts"),
    ]
    assert report["chapters"]["counts"] == {
        "placed": 2, "outside_parts": 1, "no_match": 1, "inside_line": 1}


def test_main_outputs_exist_and_are_deterministic(tmp_path, main_run):
    cfg, report, work = main_run
    names = ["epub/a1.txt", "epub/a2.txt", "epub/a3.txt", "epub/placement.json",
             "orig/placement.json", "report.json", "report.md"]
    first = {n: rd(work, n) for n in names}
    cfg2 = SourceConfig(epub=cfg.epub, work_dir=str(tmp_path / "work2"), orig_dir=cfg.orig_dir)
    report2 = run_source_stage(cfg2)
    assert report2 == report
    assert {n: rd(cfg2.work_dir, n) for n in names} == first
    assert json.loads(first["report.json"]) == report


def test_report_has_no_body_words(main_run):
    cfg, report, work = main_run
    body = set()
    for toklist in list(P.values()) + [P1_ORIG, P5_ORIG, P12_ORIG]:
        body |= vocab_in(sentence(toklist))
    for name in ("report.json", "report.md"):
        text = rd(work, name).lower()
        assert not (body & vocab_in(text)), name
        for title in ("kapxo", "titlexx", "sechead", "nohead", "tailhead"):
            assert title not in text


def test_heading_with_a_tab_is_left_out_and_listed(tmp_path):
    epub = write_epub(tmp_path, {"c.xhtml": xhtml(("h2", "Colx\tColy"), ("p", sentence(P[1])), ("p", sentence(P[2])))})
    orig = write_orig(tmp_path, {"a1": "\n".join([sentence(P[1]), "", sentence(P[2])]) + "\n"})
    work = str(tmp_path / "work")
    report = run_source_stage(SourceConfig(epub=EpubConfig(epub), work_dir=work, orig_dir=orig))
    assert report["chapters"]["excluded_tab_headings"] == [0]
    assert report["chapters"]["rows"] == []
    assert "Colx" not in rd(work, "epub/a1.txt")


JUNCTION_EPUB = {
    "c.xhtml": xhtml(
        ("p", sentence(toks(0, 20, sent=10))),
        ("p", sentence(toks(20, 40))),
        ("p", sentence(toks(40, 60))),
    )
}


def junction_run(tmp_path, parts):
    epub = write_epub(tmp_path, JUNCTION_EPUB)
    orig = write_orig(tmp_path, parts)
    cfg = SourceConfig(epub=EpubConfig(epub), work_dir=str(tmp_path / "w"), orig_dir=orig)
    return run_source_stage(cfg), json.loads(rd(cfg.work_dir, "epub/placement.json"))


def test_sentence_end_junction_inside_paragraph_with_leftover_words(tmp_path):
    t = toks(0, 20, sent=10)
    parts = {
        "a1": sentence(t[:8]) + "\n",
        "a2": sentence(t[10:]) + "\n\n" + sentence(toks(20, 40)) + "\n",
        "a3": sentence(toks(40, 60)) + "\n",
    }
    report, pl = junction_run(tmp_path, parts)
    assert pl["parts"]["a1"]["end"] == {"inside_paragraph": True, "mid_sentence": True, "next": "paragraph"}
    assert pl["parts"]["a2"]["start"] == {"inside_paragraph": True, "mid_sentence": False}
    assert pl["parts"]["a2"]["end"] == {"inside_paragraph": False, "mid_sentence": False, "next": "paragraph"}
    assert pl["parts"]["a3"]["start"] == {"inside_paragraph": False, "mid_sentence": False}
    assert pl["parts"]["a3"]["end"]["next"] == "none"
    assert report["gaps"][0] == {"after": "a1", "before": "a2", "words_between": 2, "items": []}
    assert len(report["gaps"]) == 1


def test_sentence_end_junction_both_sides(tmp_path):
    t = toks(0, 20, sent=10)
    parts = {"a1": sentence(t[:10]) + "\n", "a2": sentence(t[10:]) + "\n\n" + sentence(toks(20, 60)) + "\n"}
    report, pl = junction_run(tmp_path, parts)
    assert pl["parts"]["a1"]["end"] == {"inside_paragraph": True, "mid_sentence": False, "next": "paragraph"}
    assert pl["parts"]["a2"]["start"] == {"inside_paragraph": True, "mid_sentence": False}
    assert report["gaps"] == []


def test_partspec_by_index(tmp_path):
    files = {"a.xhtml": xhtml(
        ("h2", "Headx"), ("p", "alpha beta"), ("p", "t\tu"), ("p", "gamma delta"), ("h3", "Heady"), ("p", "eps zeta"))}
    cfg = SourceConfig(
        epub=EpubConfig(write_epub(tmp_path, files)), work_dir=str(tmp_path / "w"),
        parts=[PartSpec("one", 1), PartSpec("two", 4)])
    report = run_source_stage(cfg)
    assert rd(cfg.work_dir, "epub/one.txt") == "alpha beta\n\ngamma delta\n"
    assert rd(cfg.work_dir, "epub/two.txt") == "# Heady\n\neps zeta\n"
    assert not os.path.exists(os.path.join(cfg.work_dir, "source", "orig"))
    pl = json.loads(rd(cfg.work_dir, "epub/placement.json"))
    assert pl["parts"]["one"] == {
        "chapter_starts": [], "start": {"inside_paragraph": False, "mid_sentence": False},
        "end": {"inside_paragraph": False, "mid_sentence": False, "next": "heading"}}
    assert pl["parts"]["two"]["end"]["next"] == "none"
    assert report["gaps"] == [{"after": None, "before": "one", "words_between": 1,
                               "items": [{"index": 0, "kind": "heading", "excluded": False, "words": 1}]}]


def test_partspec_by_first_words_and_ambiguous(tmp_path):
    files = {"a.xhtml": xhtml(
        ("p", "Alpha beta one"), ("p", "alpha beta two"), ("p", "“Gamma” delta"), ("p", "last"))}
    epub = write_epub(tmp_path, files)
    cfg = SourceConfig(
        epub=EpubConfig(epub), work_dir=str(tmp_path / "w"),
        parts=[PartSpec("x", "alpha beta two"), PartSpec("y", "gamma  Delta")])
    run_source_stage(cfg)
    assert rd(cfg.work_dir, "epub/x.txt") == "alpha beta two\n"
    assert rd(cfg.work_dir, "epub/y.txt") == "“Gamma” delta\n\nlast\n"
    bad = SourceConfig(
        epub=EpubConfig(epub), work_dir=str(tmp_path / "w2"),
        parts=[PartSpec("x", "alpha beta"), PartSpec("y", 3)])
    with pytest.raises(ValueError) as e:
        run_source_stage(bad)
    assert "0" in str(e.value) and "1" in str(e.value)
    none = SourceConfig(epub=EpubConfig(epub), work_dir=str(tmp_path / "w3"), parts=[PartSpec("x", "nothing here")])
    with pytest.raises(ValueError):
        run_source_stage(none)


def test_no_orig_and_no_parts_is_an_error(tmp_path):
    cfg = SourceConfig(epub=EpubConfig(write_epub(tmp_path, {"a.xhtml": xhtml(("p", "x"))})), work_dir=str(tmp_path / "w"))
    with pytest.raises(ValueError):
        run_source_stage(cfg)


def test_import_pulls_in_no_third_party_package():
    code = (
        "import sys\n"
        "sys.path.insert(0, %r)\n"
        "before = set(sys.modules)\n"
        "import src.data.prepare.source\n"
        "new = {m.split('.')[0] for m in set(sys.modules) - before}\n"
        "bad = sorted(m for m in new if m not in sys.stdlib_module_names and not m.startswith('_') and m != 'src')\n"
        "print(bad)\n" % REPO
    )
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
