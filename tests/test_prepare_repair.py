import copy
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

from src.data.prepare.config import load_config
from src.data.prepare.asr import ASR_BACKENDS, transcription_identity
from src.data.prepare.terms import BookTerms
from src.data.prepare.repair import apply_repairs, propose_repairs
from src.data.prepare.text import _words

JUDGES = ("parakeet-tdt-0.6b-v3", "canary-1b-v2")


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def producer_report(work, cfg):
    models = {j: ASR_BACKENDS[j].model_id for j in cfg.audit.judges}
    put(work / "audit/report.json", {"config": dataclasses.asdict(cfg.audit), "judges": models,
                                      "asr_provenance": {j: transcription_identity(j, cfg.language) for j in models},
                                      "terminology": {"policy_sha256": BookTerms(cfg.language, cfg.text.terms).policy_sha256},
                                      "asr_model": " + ".join(models.values()),
                                      "parts": {part: {} for part in cfg.part_names()}})


@pytest.fixture
def scene(tmp_path):
    work = tmp_path / "work"
    text = '„Na louce svítí,“ řekl.'
    words, spans = _words(text)
    unit = {"id": "p-00000", "part": "p", "index": 0, "text": text, "words": words, "word_spans": spans,
            "sentences": [{"text": text, "boundary": "sentence"}], "boundary": "sentence",
            "source": {"line_start": 1, "line_end": 1}, "tags": ["quote_then_narration"], "est_s": 3}
    put(work / "text/p.units.jsonl", unit)
    pcm = np.arange(1200, dtype=np.int16) * 10
    (work / "audio").mkdir()
    (work / "cut/clips").mkdir(parents=True)
    sf.write(work / "audio/p.24k.wav", pcm, 24000, subtype="PCM_16")
    sf.write(work / "audio/p.16k.wav", pcm[:800], 16000, subtype="PCM_16")
    sf.write(work / "cut/clips/p-00000.wav", pcm[200:1000], 24000, subtype="PCM_16")
    sha = hashlib.sha256(pcm[200:1000].tobytes()).hexdigest()
    put(work / "cut/p.cuts.jsonl", {"id": unit["id"], "units": [unit["id"]], "text": text, "sha256": sha,
                                    "start_sample": 200, "end_sample": 1000})
    put(work / "audit/p.audit.jsonl", {"id": unit["id"], "sha256": sha, "verdict": "drop", "hyp": "Na louce září řekl",
                                      "hyps": {JUDGES[0]: "Na louce září řekl", JUDGES[1]: "Na louce září řekl"}})
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump({"work_dir": str(work), "parts": ["p"]}))
    cfg = load_config(cfg_path)
    # Trusted vocabulary can come from another book part even when only p is
    # selected. This is source evidence, not a claimed Czech dictionary.
    put(work / "text/q.units.jsonl", {"text": "září"})
    producer_report(work, cfg)
    return cfg, work, unit


def reviewed(cfg):
    proposal = propose_repairs(cfg)
    assert len(proposal["edits"]) == 1
    assert proposal["edits"][0]["reviewed"] is False
    assert proposal["edits"][0]["judges"] == sorted(JUDGES)
    proposal["edits"][0].update(reviewed=True, reviewer="listener")
    return proposal


def test_repair_preserves_dialogue_and_provenance_and_reruns_gates(scene, tmp_path):
    cfg, work, unit = scene
    proposal = reviewed(cfg)
    before = {p.relative_to(work): p.read_bytes() for p in work.rglob("*") if p.is_file()}
    output, stages = tmp_path / "candidate", []

    def run(stage, candidate):
        assert candidate.work_dir == str(output)
        assert not (output / "audit/p.audit.jsonl").exists()
        stages.append(stage)

    receipt = apply_repairs(cfg, proposal, output, run)
    row = json.loads((output / "text/p.units.jsonl").read_text())
    assert row["text"] == '„Na louce září,“ řekl.'
    assert row["sentences"][0]["text"] == row["text"]
    assert (row["words"], row["word_spans"]) == _words(row["text"])
    assert row["source"] == unit["source"] and row["tags"] == unit["tags"]
    assert row["repair"]["reviewers"] == ["listener"]
    assert json.loads((output / "repair/source-text/p.units.jsonl").read_text()) == unit
    assert receipt["complete"] is True and stages == ["align", "cut", "audit", "manifest"]
    assert {p.relative_to(work): p.read_bytes() for p in work.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("change", ["units", "audio", "audit", "config", "original", "punctuation", "duplicate", "reviewer", "unknown", "clip_audio", "clip_id"])
def test_repair_rejects_stale_or_unsafe_proposals_before_writing(scene, tmp_path, change):
    cfg, work, _ = scene
    proposal = reviewed(cfg)
    edit = proposal["edits"][0]
    if change in ("units", "audio", "audit"):
        path = {"units": "text/p.units.jsonl", "audio": "audio/p.16k.wav", "audit": "audit/p.audit.jsonl"}[change]
        (work / path).write_bytes((work / path).read_bytes() + b"\n")
    elif change == "clip_audio":
        (work / "cut/clips/p-00000.wav").write_text("changed")
    elif change == "clip_id":
        edit["clip_id"] = "unrelated"
    elif change == "config":
        cfg.align.margin_s += 1
    elif change == "original":
        edit["original"] = "jiné"
    elif change == "punctuation":
        edit["replacement"] = 'září“'
    elif change == "duplicate":
        proposal["edits"].append(copy.deepcopy(edit))
    elif change == "reviewer":
        edit["reviewer"] = " "
    else:
        edit["unit_id"] = "unknown"
    output = tmp_path / "candidate"
    with pytest.raises(ValueError):
        apply_repairs(cfg, proposal, output, lambda *_: pytest.fail("must not run"))
    assert not output.exists()


def test_unreviewed_proposal_does_not_apply(scene, tmp_path):
    cfg, _, _ = scene
    with pytest.raises(ValueError, match="review"):
        apply_repairs(cfg, propose_repairs(cfg), tmp_path / "candidate", lambda *_: None)


def test_unknown_word_is_grouped_for_book_review_without_language_guess(scene):
    cfg, work, _ = scene
    (work / "text/q.units.jsonl").unlink()
    proposal = propose_repairs(cfg)
    assert proposal["edits"] == []
    assert len(proposal["term_review"]) == 1
    term = proposal["term_review"][0]
    assert term["original"] == "svítí" and term["candidates"] == ["září"]
    assert term["language"] == "cs" and term["reasons"] == ["new_vocabulary"]


def test_reviewed_glossary_can_admit_a_new_ordinary_word(scene, tmp_path):
    cfg, work, _ = scene
    (work / "text/q.units.jsonl").unlink()
    path = tmp_path / "terms.tsv"
    path.write_text("canonical\talias\tlanguage\tkind\nzáří\t\tcs\tword\n")
    cfg.text.terms = cfg.audit.terms = str(path)
    producer_report(work, cfg)
    proposal = reviewed(cfg)
    assert proposal["term_review"] == []
    path.write_text(path.read_text() + "jinak\t\tcs\tword\n")
    with pytest.raises(ValueError, match="producer"):
        propose_repairs(cfg)
    with pytest.raises(ValueError, match="producer"):
        apply_repairs(cfg, proposal, tmp_path / "candidate", lambda *_: None)


def test_unselected_book_vocabulary_changes_invalidate_a_reviewed_proposal(scene, tmp_path):
    cfg, work, _ = scene
    proposal = reviewed(cfg)
    put(work / "text/q.units.jsonl", {"text": "jinak"})
    with pytest.raises(ValueError, match="stale"):
        apply_repairs(cfg, proposal, tmp_path / "candidate", lambda *_: None)


def test_protected_spoken_form_routes_consensus_out_of_edits_and_cannot_be_forged(scene, tmp_path):
    cfg, work, unit = scene
    text = "Ten mladej host čeká."
    unit.update(text=text, words=_words(text)[0], word_spans=_words(text)[1], sentences=[{"text": text}])
    put(work / "text/p.units.jsonl", unit)
    cut = json.loads((work / "cut/p.cuts.jsonl").read_text())
    cut["text"] = text
    put(work / "cut/p.cuts.jsonl", cut)
    audit = json.loads((work / "audit/p.audit.jsonl").read_text())
    audit["hyps"] = dict.fromkeys(JUDGES, "Ten mladý host čeká.")
    put(work / "audit/p.audit.jsonl", audit)
    put(work / "text/q.units.jsonl", {"text": "mladý"})
    path = tmp_path / "terms.tsv"
    path.write_text("canonical\talias\tlanguage\tkind\nmladej\tmladý\tcs\tprotected\n")
    cfg.text.terms = cfg.audit.terms = str(path)
    producer_report(work, cfg)
    proposal = propose_repairs(cfg)
    assert proposal["edits"] == []
    assert proposal["term_review"][0]["reasons"] == ["protected_source"]
    edit = proposal["term_review"][0]["occurrences"][0]
    edit.update(reviewed=True, reviewer="listener")
    proposal["edits"] = [edit]
    with pytest.raises(ValueError, match="unprotected"):
        apply_repairs(cfg, proposal, tmp_path / "candidate", lambda *_: None)


def test_competing_replacements_across_book_occurrences_are_grouped_once(scene):
    import shutil

    cfg, work, unit = scene
    put(work / "text/q.units.jsonl", {"text": "září prší"})
    second = {**unit, "id": "p-00001"}
    with (work / "text/p.units.jsonl").open("a") as stream:
        stream.write(json.dumps(second) + "\n")
    cut = json.loads((work / "cut/p.cuts.jsonl").read_text())
    with (work / "cut/p.cuts.jsonl").open("a") as stream:
        stream.write(json.dumps({**cut, "id": second["id"], "units": [second["id"]]}) + "\n")
    audit = json.loads((work / "audit/p.audit.jsonl").read_text())
    with (work / "audit/p.audit.jsonl").open("a") as stream:
        stream.write(json.dumps({**audit, "id": second["id"], "hyps": dict.fromkeys(JUDGES, "Na louce prší řekl")}) + "\n")
    shutil.copyfile(work / "cut/clips/p-00000.wav", work / "cut/clips/p-00001.wav")
    proposal = propose_repairs(cfg)
    assert proposal["edits"] == [] and len(proposal["term_review"]) == 1
    term = proposal["term_review"][0]
    assert term["candidates"] == ["prší", "září"]
    assert term["reasons"] == ["competing_replacements"] and len(term["occurrences"]) == 2


def test_audit_failure_stops_before_manifest_and_records_incomplete_candidate(scene, tmp_path):
    cfg, _, _ = scene
    output, calls = tmp_path / "candidate", []

    def run(stage, candidate):
        calls.append(stage)
        if stage == "audit":
            raise ValueError("synthetic gate failure")

    with pytest.raises(ValueError, match="gate failure"):
        apply_repairs(cfg, reviewed(cfg), output, run)
    receipt = json.loads((output / "repair/receipt.json").read_text())
    assert calls == ["align", "cut", "audit"]
    assert receipt["complete"] is False and receipt["stages"] == ["align", "cut"]
    assert not (output / "manifest").exists()


def test_repair_never_overwrites_source_or_existing_candidate(scene):
    cfg, work, _ = scene
    for output in (work, work / "nested", work.parent):
        with pytest.raises(ValueError, match="new directory"):
            apply_repairs(cfg, reviewed(cfg), output, lambda *_: None)


def test_repair_cli_dispatches_existing_stages_after_explicit_review(scene, tmp_path, monkeypatch):
    from scripts import prepare_corpus, repair_corpus

    cfg, _, _ = scene
    proposal_path = tmp_path / "proposal.json"
    config_path = tmp_path / "cfg.yaml"
    assert repair_corpus.main(["propose", "--config", str(config_path), "--proposal", str(proposal_path)]) == 0
    proposal = json.loads(proposal_path.read_text())
    proposal["edits"][0].update(reviewed=True, reviewer="listener")
    put(proposal_path, proposal)
    calls = []
    for stage in ("align", "cut", "audit", "manifest"):
        monkeypatch.setitem(prepare_corpus.RUNNERS, stage,
                            lambda candidate, hooks, stage=stage: calls.append((stage, candidate.work_dir)))
    output = tmp_path / "candidate"
    assert repair_corpus.main(["apply", "--config", str(config_path), "--proposal", str(proposal_path),
                               "--output", str(output)]) == 0
    assert calls == [(stage, str(output)) for stage in ("align", "cut", "audit", "manifest")]


@pytest.mark.parametrize("case", ["ok", "case_only"])
def test_proposals_exclude_accepted_clips_and_casing_only_disagreements(scene, case):
    cfg, work, unit = scene
    audit = json.loads((work / "audit/p.audit.jsonl").read_text())
    if case == "ok":
        audit["verdict"] = "ok"
    else:
        audit["hyp"] = unit["text"].lower()
        audit["hyps"] = {JUDGES[0]: unit["text"].lower(), JUDGES[1]: unit["text"]}
    put(work / "audit/p.audit.jsonl", audit)
    assert propose_repairs(cfg)["edits"] == []


def test_proposal_word_comparison_accepts_raw_asr_symbols_without_full_text_normalization(scene):
    cfg, work, _ = scene
    audit = json.loads((work / "audit/p.audit.jsonl").read_text())
    audit["hyp"] = "Na louce září @ řekl"
    audit["hyps"] = {JUDGES[0]: audit["hyp"], JUDGES[1]: "na louce ZÁŘÍ řekl"}
    put(work / "audit/p.audit.jsonl", audit)
    edits = propose_repairs(cfg)["edits"]
    assert len(edits) == 1
    assert edits[0]["judges"] == sorted(JUDGES)
    assert edits[0]["replacement"].casefold() == "září"
    assert edits[0]["reviewed"] is False


def test_ordinary_word_repair_preserves_source_case(scene):
    cfg, work, _ = scene
    audit = json.loads((work / "audit/p.audit.jsonl").read_text())
    audit["hyps"] = dict.fromkeys(JUDGES, "NA LOUCE ZÁŘÍ ŘEKL")
    put(work / "audit/p.audit.jsonl", audit)
    assert propose_repairs(cfg)["edits"][0]["replacement"] == "září"


def test_proposals_require_every_configured_judge_and_support_single_judge_audit(scene):
    cfg, work, unit = scene
    audit = json.loads((work / "audit/p.audit.jsonl").read_text())
    audit["hyps"][JUDGES[1]] = unit["text"]
    put(work / "audit/p.audit.jsonl", audit)
    assert propose_repairs(cfg)["edits"] == []
    cfg.audit.judges = (JUDGES[0],)
    producer_report(work, cfg)
    audit.pop("hyps")
    put(work / "audit/p.audit.jsonl", audit)
    assert propose_repairs(cfg)["edits"][0]["judges"] == [JUDGES[0]]


def test_missing_configured_judge_does_not_turn_partial_evidence_into_consensus(scene):
    cfg, work, _ = scene
    audit = json.loads((work / "audit/p.audit.jsonl").read_text())
    audit["hyps"].pop(JUDGES[1])
    put(work / "audit/p.audit.jsonl", audit)
    assert propose_repairs(cfg)["edits"] == []


@pytest.mark.parametrize("change", ["model", "missing_producer", "audit_config", "single_judge_relabel"])
def test_propose_rejects_stale_producer_identity(scene, change):
    cfg, work, _ = scene
    report = json.loads((work / "audit/report.json").read_text())
    if change == "model":
        report["judges"][JUDGES[0]] = ASR_BACKENDS[JUDGES[1]].model_id
    elif change == "missing_producer":
        report.pop("judges")
    elif change == "audit_config":
        cfg.audit.cer_margin += .01
    else:
        cfg.audit.judges = (JUDGES[1],)
        producer_report(work, cfg)
        report = json.loads((work / "audit/report.json").read_text())
        cfg.audit.judges = (JUDGES[0],)
        audit = json.loads((work / "audit/p.audit.jsonl").read_text())
        audit.pop("hyps")
        put(work / "audit/p.audit.jsonl", audit)
    put(work / "audit/report.json", report)
    with pytest.raises(ValueError, match="producer"):
        propose_repairs(cfg)


@pytest.mark.parametrize("change", ["clip", "source", "range", "rate", "channels"])
def test_propose_rejects_preexisting_audio_identity_mismatch(scene, change):
    cfg, work, _ = scene
    if change in ("clip", "source"):
        path = work / ("cut/clips/p-00000.wav" if change == "clip" else "audio/p.24k.wav")
        pcm, rate = sf.read(path, dtype="int16")
        pcm[300] += 1
        sf.write(path, pcm, rate, subtype="PCM_16")
    elif change == "range":
        cut = json.loads((work / "cut/p.cuts.jsonl").read_text())
        cut["end_sample"] = 1400
        put(work / "cut/p.cuts.jsonl", cut)
    else:
        pcm = np.zeros((800, 2) if change == "channels" else 800, dtype=np.int16)
        sf.write(work / "cut/clips/p-00000.wav", pcm, 16000 if change == "rate" else 24000, subtype="PCM_16")
    with pytest.raises(ValueError, match="audio|range|PCM"):
        propose_repairs(cfg)


def test_apply_rechecks_source_consistency_despite_manually_rebound_input_hash(scene, tmp_path):
    cfg, work, _ = scene
    proposal = reviewed(cfg)
    path = work / "audio/p.24k.wav"
    pcm, rate = sf.read(path, dtype="int16")
    pcm[300] += 1
    sf.write(path, pcm, rate, subtype="PCM_16")
    proposal["inputs"]["audio/p.24k.wav"] = hashlib.sha256(path.read_bytes()).hexdigest()
    output = tmp_path / "candidate"
    with pytest.raises(ValueError, match="audio|PCM"):
        apply_repairs(cfg, proposal, output, lambda *_: None)
    assert not output.exists()


def test_apply_rechecks_producer_consistency_despite_manually_rebound_config_hash(scene, tmp_path):
    cfg, work, _ = scene
    proposal = reviewed(cfg)
    cfg.audit.cer_margin += .01
    proposal["config_sha256"] = hashlib.sha256(
        json.dumps(dataclasses.asdict(cfg), sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    output = tmp_path / "candidate"
    with pytest.raises(ValueError, match="producer"):
        apply_repairs(cfg, proposal, output, lambda *_: None)
    assert not output.exists()


@pytest.mark.parametrize("parts", [None, {"other": {}}])
def test_propose_rejects_audit_report_that_does_not_cover_requested_parts(scene, parts):
    cfg, work, _ = scene
    report = json.loads((work / "audit/report.json").read_text())
    report["parts"] = parts
    put(work / "audit/report.json", report)
    with pytest.raises(ValueError, match="producer.*parts"):
        propose_repairs(cfg)


def test_propose_accepts_requested_subset_of_audit_report(scene):
    cfg, work, _ = scene
    report = json.loads((work / "audit/report.json").read_text())
    report["parts"]["other"] = {}
    put(work / "audit/report.json", report)
    assert len(propose_repairs(cfg)["edits"]) == 1
