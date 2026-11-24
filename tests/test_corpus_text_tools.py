"""Synthetic CLI checks against the project's actual eSpeak NG tokenizer."""
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from src.text.phonemizer import CzechSlovakPhonemizer, SPECIAL_TOKENS, tokens_to_ids

ROOT = Path(__file__).resolve().parents[1]


def run(tool, *args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / tool), *map(str, args)],
                          capture_output=True, text=True, env={**os.environ, "OMP_NUM_THREADS": "1"})


def manifest(tmp_path, rows, name="rows.jsonl"):
    path = tmp_path / name
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return path


def test_vocab_cli_spans_lexicon_coverage_quotes_and_provenance(tmp_path):
    row = {"id": "fixture", "lang": "sk", "text": '„Kôň“ a <en>a cup of tea</en>.',
           "quote_before": 2, "quote_after": 1}
    row["n_tokens"] = len(CzechSlovakPhonemizer("sk").tokenize(
        row["text"], quote_before=2, quote_after=1))
    source = manifest(tmp_path, [row])
    lexicon = tmp_path / "lexicon.json"
    lexicon.write_text(json.dumps({"cs": {"cvičný": "ʘ a"}, "en": {"sample": "ˈ ʙ"}}), encoding="utf-8")
    coverage = tmp_path / "coverage.txt"
    coverage.write_text("Řeka teče.\n", encoding="utf-8")
    output = tmp_path / "vocab.json"
    result = run("build_vocab.py", "--manifests", source, "--out", output,
                 "--lexicon", lexicon, "--coverage", f"cs={coverage}")
    assert result.returncode == 0, result.stderr
    vocab = json.loads(output.read_text())
    assert [vocab[s] for s in SPECIAL_TOKENS] == list(range(len(SPECIAL_TOKENS)))
    assert "ʘ" in vocab and "ʙ" in vocab and "ˈ" in vocab
    tokens = CzechSlovakPhonemizer("sk").tokenize(row["text"], quote_before=2, quote_after=1)
    assert len(tokens_to_ids(tokens, vocab)[0]) == row["n_tokens"]
    meta = json.loads(Path(str(output) + ".meta.json").read_text())
    assert meta["languages"] == ["cs", "en", "sk"]
    assert meta["rows"] == 1 and meta["vocab_sha256"]
    assert meta["manifests"][0]["sha256"] and meta["lexicon"]["sha256"]
    assert row["text"] not in result.stdout + result.stderr + json.dumps(meta)


def test_vocab_cli_deterministic_and_overwrite_guard(tmp_path):
    rows = [{"lang": "cs", "text": "Dobrý den."}, {"lang": "sk", "text": "Kôň ide."}]
    a = manifest(tmp_path, rows, "a.jsonl")
    b = manifest(tmp_path, list(reversed(rows)), "b.jsonl")
    out_a, out_b = tmp_path / "a.json", tmp_path / "b.json"
    for source, out in [(a, out_a), (b, out_b)]:
        result = run("build_vocab.py", "--manifests", source, "--out", out)
        assert result.returncode == 0, result.stderr
    assert out_a.read_bytes() == out_b.read_bytes()
    saved = out_a.read_bytes()
    assert run("build_vocab.py", "--manifests", a, "--out", out_a).returncode != 0
    assert out_a.read_bytes() == saved
    assert run("build_vocab.py", "--manifests", a, "--out", out_a, "--force").returncode == 0
    assert run("build_vocab.py", "--manifests", a, "--out", a, "--force").returncode != 0


@pytest.mark.parametrize("change", [{"n_tokens": 999}, {"lang": "de"}, {"quote_before": True},
                                     {"text": "privatefixture & forbidden"}, {"n_tokens": True}])
def test_manifest_errors_are_safe_and_do_not_write(tmp_path, change):
    source = manifest(tmp_path, [{"lang": "cs", "text": "privatefixture", **change}])
    for tool, flag in [("build_vocab.py", "--manifests"), ("g2p_review.py", "--manifest")]:
        output = tmp_path / tool
        result = run(tool, flag, source, "--out", output)
        assert result.returncode != 0
        assert "privatefixture" not in result.stdout + result.stderr
        assert not output.exists()


def test_review_cli_frequency_variants_spans_and_tsv_roundtrip(tmp_path):
    text = '„Zorim“\tčíta <en>"a cup of tea"</en>.\nKôň a kôň, sh qwerty. Brat ide.'
    source = manifest(tmp_path, [{"id": "fixture", "lang": "sk", "text": text},
                                 {"id": "second", "lang": "sk", "text": "kôň kôň ide."}])
    output = tmp_path / "review.tsv"
    result = run("g2p_review.py", "--manifest", source, "--out", output, "--top-n", 1)
    assert result.returncode == 0, result.stderr
    rows = list(csv.DictReader(io.StringIO(output.read_text()), delimiter="\t"))
    assert list(rows[0]) == ["language", "word", "count", "phones", "reason", "example_id", "example_text"]
    by_word = {r["word"]: r for r in rows}
    assert by_word["kôň"]["count"] == "2"
    assert "kôň" in by_word and "Kôň" in by_word
    assert "casing_variant" in by_word["Kôň"]["reason"]
    assert "frequent" in by_word["kôň"]["reason"]
    assert by_word["Zorim"]["example_text"] == text
    assert "a cup of tea" in by_word
    assert by_word["a cup of tea"]["language"] == "en"
    assert not {"en", "sk", "cs"} & by_word.keys()
    assert "context:" in by_word["Brat"]["phones"]
    assert "context_voicing" in by_word["Brat"]["reason"]
    assert "qwerty" not in result.stdout + result.stderr
    saved = output.read_bytes()
    assert run("g2p_review.py", "--manifest", source, "--out", output).returncode != 0
    assert output.read_bytes() == saved


def test_review_deterministic_across_manifest_order_and_multiple_languages(tmp_path):
    a = manifest(tmp_path, [{"id": "a", "lang": "cs", "text": "<sk>Brat ide.</sk> <cs>Den.</cs>"}], "a.jsonl")
    b = manifest(tmp_path, [{"id": "b", "lang": "cs", "text": "Brat ide."}], "b.jsonl")
    output_a, output_b = tmp_path / "a.tsv", tmp_path / "b.tsv"
    first = run("g2p_review.py", "--manifest", a, b, "--out", output_a)
    second = run("g2p_review.py", "--manifest", b, a, "--out", output_b)
    assert first.returncode == second.returncode == 0, first.stderr + second.stderr
    assert output_a.read_bytes() == output_b.read_bytes()
    rows = list(csv.DictReader(io.StringIO(output_a.read_text()), delimiter="\t"))
    assert {(r["language"], r["word"]) for r in rows} >= {("sk", "Brat"), ("cs", "Brat"), ("cs", "Den")}


def test_force_replaces_output_without_mutating_hardlinked_input(tmp_path):
    source = manifest(tmp_path, [{"lang": "cs", "text": "Dobrý den."}])
    saved = source.read_bytes()
    output = tmp_path / "vocab.json"
    os.link(source, output)
    result = run("build_vocab.py", "--manifests", source, "--out", output, "--force")
    assert result.returncode == 0, result.stderr
    assert source.read_bytes() == saved
    assert "<pad>" in json.loads(output.read_text())


@pytest.mark.parametrize("text", ['{"lang": [], "text": "privatefixture"}',
                                 '{"lang": "cs", "text": "privatefixture", "text": "other"}',
                                 '["privatefixture"]', '{bad privatefixture'])
def test_invalid_manifest_structure_has_helpful_private_error(tmp_path, text):
    source = tmp_path / "bad.jsonl"
    source.write_text(text, encoding="utf-8")
    result = run("build_vocab.py", "--manifests", source, "--out", tmp_path / "vocab.json")
    assert result.returncode != 0 and "manifest 1, line 1" in result.stderr
    assert "Traceback" not in result.stderr and "privatefixture" not in result.stderr


@pytest.mark.parametrize("text", ["<en>a <sk>kôň</sk></en>", "&lt;en&gt;hello&lt;/en&gt;"])
def test_review_rejects_nested_or_escaped_markup_like_tokenizer(tmp_path, text):
    with pytest.raises(ValueError):
        CzechSlovakPhonemizer("cs").tokenize(text)
    source = manifest(tmp_path, [{"lang": "cs", "text": text}])
    result = run("g2p_review.py", "--manifest", source, "--out", tmp_path / "review.tsv")
    assert result.returncode != 0 and "<sk>" not in result.stderr


@pytest.mark.parametrize("lexicon", [{"de": {"entry": "a"}}, {"sk": {"two words": "a"}},
                                      {"sk": {"entry": ""}}, {"sk": ["bad"]}])
def test_vocab_rejects_malformed_lexicon(tmp_path, lexicon):
    source = manifest(tmp_path, [{"lang": "cs", "text": "Dobrý den."}])
    lex = tmp_path / "lexicon.json"
    lex.write_text(json.dumps(lexicon), encoding="utf-8")
    result = run("build_vocab.py", "--manifests", source, "--lexicon", lex, "--out", tmp_path / "vocab.json")
    assert result.returncode != 0 and not (tmp_path / "vocab.json").exists()
