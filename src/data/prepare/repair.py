"""Reviewed word substitutions in an isolated corpus candidate, followed by normal quality gates.

Proposals are suggestions, never approvals. Only individual normalized words can be
replaced; punctuation, dialogue marks, tags and source locations remain intact.
"""
from __future__ import annotations

import copy
import dataclasses
import difflib
import hashlib
import json
import shutil
import unicodedata
from pathlib import Path

import soundfile as sf

from src.data.prepare.text import _words
from src.data.prepare.audit import spell_digits
from src.data.prepare.asr import ASR_BACKENDS
from src.text.normalizer import TextNormalizer
from src.text.units import estimate_s


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _inputs(cfg):
    work = Path(cfg.work_dir)
    paths = []
    for part in cfg.part_names():
        paths += [f"text/{part}.units.jsonl", f"audio/{part}.16k.wav", f"audio/{part}.24k.wav",
                  f"cut/{part}.cuts.jsonl", f"audit/{part}.audit.jsonl"]
    paths += ["audit/report.json"]
    return {p: _sha(work / p) for p in paths}



def _validate_producer(cfg):
    """Bind the saved hypotheses to the actual configured audit producer."""
    try:
        report = json.loads((Path(cfg.work_dir) / "audit" / "report.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ValueError("audit producer metadata is missing or malformed") from None
    models = {judge: ASR_BACKENDS[judge].model_id for judge in cfg.audit.judges}
    if (report.get("judges") != models
            or report.get("asr_model") != " + ".join(models.values())
            or _json_hash(report.get("config")) != _json_hash(dataclasses.asdict(cfg.audit))):
        raise ValueError("audit producer judges, model identities or configuration differ; rerun audit")


def _verify_cut_audio(work, part, cut):
    """Check a bounded source range and its saved clip against the declared PCM hash."""
    start, end = cut.get("start_sample"), cut.get("end_sample")
    if type(start) is not int or type(end) is not int or not 0 <= start < end:
        raise ValueError("cut audio range must contain ordered nonnegative integer samples")
    try:
        with sf.SoundFile(work / "audio" / f"{part}.24k.wav") as source:
            if (source.samplerate != 24000 or source.channels != 1 or source.subtype != "PCM_16"
                    or source.format not in ("WAV", "WAVEX")):
                raise ValueError("source audio must be mono 24k PCM16 WAV")
            if end > source.frames:
                raise ValueError("cut audio range exceeds source audio")
            source.seek(start)
            pcm = source.read(end - start, dtype="int16")
            if hashlib.sha256(pcm.tobytes()).hexdigest() != cut["sha256"]:
                raise ValueError("source range audio differs from the declared cut PCM hash; rerun cut/audit")
        with sf.SoundFile(work / "cut" / "clips" / f"{cut['id']}.wav") as clip:
            if (clip.samplerate != 24000 or clip.channels != 1 or clip.subtype != "PCM_16"
                    or clip.format not in ("WAV", "WAVEX") or clip.frames != end - start):
                raise ValueError("clip audio format or length differs from its source range")
            if hashlib.sha256(clip.read(dtype="int16").tobytes()).hexdigest() != cut["sha256"]:
                raise ValueError("clip audio differs from the declared cut PCM hash; rerun cut/audit")
    except sf.LibsndfileError:
        raise ValueError("source or clip audio is missing or is not a readable WAV") from None


def propose_repairs(cfg):
    """Suggest unanimous, case-insensitive word changes in dropped single-unit clips."""
    work = Path(cfg.work_dir).resolve()
    _validate_producer(cfg)
    result = {"version": 1, "source_work_dir": str(work), "config_sha256": _json_hash(dataclasses.asdict(cfg)),
              "inputs": _inputs(cfg), "edits": []}
    for part in cfg.part_names():
        units = {u["id"]: u for u in _rows(work / "text" / f"{part}.units.jsonl")}
        audits = {a["id"]: a for a in _rows(work / "audit" / f"{part}.audit.jsonl")}
        for cut in _rows(work / "cut" / f"{part}.cuts.jsonl"):
            if len(cut["units"]) != 1 or cut["units"][0] not in units:
                continue
            unit, audit = units[cut["units"][0]], audits[cut["id"]]
            if audit["verdict"] != "drop":
                continue
            if audit["sha256"] != cut["sha256"] or cut["text"] != unit["text"]:
                raise ValueError("stale cut/audit text or audio; regenerate before proposing repairs")
            _verify_cut_audio(work, part, cut)
            words, _ = _words(unit["text"])
            hypotheses = audit.get("hyps")
            if hypotheses is None:
                if len(cfg.audit.judges) != 1:
                    continue
                hypotheses = {cfg.audit.judges[0]: audit["hyp"]}
            if any(judge not in hypotheses for judge in cfg.audit.judges):
                continue
            hypotheses = {judge: hypotheses[judge] for judge in cfg.audit.judges}
            key = lambda word: unicodedata.normalize("NFC", word).casefold()
            candidates = {}
            for judge, hyp in hypotheses.items():
                # Audit already defines digit spelling; raw ASR is not book prose
                # and may contain symbols with no TextNormalizer reading.
                hyp_words, _ = _words(unicodedata.normalize("NFC", spell_digits(hyp, cfg.language)))
                for op, a, b, c, d in difflib.SequenceMatcher(
                        a=[key(w) for w in words], b=[key(w) for w in hyp_words], autojunk=False).get_opcodes():
                    if op != "replace" or b - a != d - c:
                        continue
                    for index, replacement in zip(range(a, b), hyp_words[c:d]):
                        if key(replacement) != key(words[index]):
                            candidate = candidates.setdefault((index, key(replacement)),
                                                              {"replacement": replacement, "judges": []})
                            candidate["judges"].append(judge)
            for (index, _), candidate in sorted(candidates.items()):
                judges, replacement = candidate["judges"], candidate["replacement"]
                if len(judges) != len(cfg.audit.judges):
                    continue
                result["edits"].append({"part": part, "unit_id": unit["id"], "word_index": index,
                                        "original": words[index], "replacement": replacement,
                                        "clip_id": cut["id"], "clip_file_sha256": _sha(work / "cut" / "clips" / f"{cut['id']}.wav"),
                                        "judges": sorted(judges),
                                        "reviewed": False, "reviewer": ""})
    return result


def _replace_unit(unit, edits, normalizer, estimate):
    row = copy.deepcopy(unit)
    words, spans = _words(row["text"])
    if row["words"] != words or row["word_spans"] != spans:
        raise ValueError("unit word metadata differs from its text")
    replacements = {}
    for edit in edits:
        index, replacement = edit["word_index"], edit["replacement"]
        if type(index) is not int or not 0 <= index < len(words) or index in replacements:
            raise ValueError("invalid or duplicate word index")
        if words[index] != edit["original"]:
            raise ValueError("proposal original word differs from source")
        if (not isinstance(replacement, str) or replacement == words[index]
                or _words(replacement) != ([replacement], [[0, len(replacement)]])
                or normalizer.normalize(replacement) != replacement):
            raise ValueError("replacement must be one changed normalized word without punctuation")
        replacements[index] = replacement

    def substitute(text, word_spans, changes):
        for index in sorted(changes, reverse=True):
            start, end = word_spans[index]
            text = text[:start] + changes[index] + text[end:]
        return text

    sentence_words, offset = [], 0
    for sentence in row.get("sentences", []):
        sw, ss = _words(sentence["text"])
        sentence_words.extend(sw)
        changes = {i - offset: value for i, value in replacements.items() if offset <= i < offset + len(sw)}
        sentence["text"] = substitute(sentence["text"], ss, changes)
        offset += len(sw)
    if row.get("sentences") and sentence_words != words:
        raise ValueError("sentence metadata cannot be mapped to unit words")
    row["text"] = substitute(row["text"], spans, replacements)
    row["words"], row["word_spans"] = _words(row["text"])
    row["est_s"] = round(estimate_s(row["text"], **estimate), 3)
    row["repair"] = {"original_text_sha256": _json_hash(unit["text"]),
                     "reviewers": sorted({e["reviewer"].strip() for e in edits})}
    return row


def apply_repairs(cfg, proposal, output_dir, run_stage):
    """Validate reviewed edits, create a new candidate, then align/cut/audit/manifest.

    run_stage(name, candidate_cfg) is the existing pipeline stage dispatcher. A
    failed gate leaves a receipt with complete=false and preserves diagnostic output.
    """
    source, output = Path(cfg.work_dir).resolve(), Path(output_dir).resolve()
    if output.exists() or output == source or source in output.parents:
        raise ValueError("repair output must be a new directory outside the source work directory")
    _validate_producer(cfg)
    if (proposal.get("version") != 1 or proposal.get("source_work_dir") != str(source)
            or proposal.get("config_sha256") != _json_hash(dataclasses.asdict(cfg))
            or proposal.get("inputs") != _inputs(cfg)):
        raise ValueError("stale or mismatched repair proposal")
    edits = [e for e in proposal["edits"] if e.get("reviewed") is True]
    if not edits or any(not isinstance(e.get("reviewer"), str) or not e["reviewer"].strip() for e in edits):
        raise ValueError("each applied edit requires explicit review and a reviewer")
    cuts = {(part, c["id"]): c for part in cfg.part_names()
            for c in _rows(source / "cut" / f"{part}.cuts.jsonl")}
    audits = {(part, a["id"]): a for part in cfg.part_names()
              for a in _rows(source / "audit" / f"{part}.audit.jsonl")}
    by_unit, verified = {}, set()
    for edit in edits:
        cut = cuts.get((edit["part"], edit["clip_id"]))
        if (cut is None or cut["units"] != [edit["unit_id"]]
                or edit.get("clip_file_sha256") != _sha(source / "cut" / "clips" / f"{cut['id']}.wav")):
            raise ValueError("proposal clip is mismatched or its audio changed")
        audit = audits.get((edit["part"], cut["id"]))
        if audit is None or audit["sha256"] != cut["sha256"] or audit["verdict"] != "drop":
            raise ValueError("proposal cut audio identity or verdict differs from its saved audit")
        identity = (edit["part"], cut["id"])
        if identity not in verified:
            _verify_cut_audio(source, edit["part"], cut)
            verified.add(identity)
        by_unit.setdefault((edit["part"], edit["unit_id"]), []).append(edit)
    normalizer = TextNormalizer(cfg.language, cfg.text.book_config)
    candidate_rows = {}
    for part in cfg.part_names():
        candidate_rows[part] = []
        for unit in _rows(source / "text" / f"{part}.units.jsonl"):
            changes = by_unit.pop((part, unit["id"]), [])
            candidate_rows[part].append(_replace_unit(unit, changes, normalizer, cfg.text.estimate) if changes else unit)
    if by_unit:
        raise ValueError("proposal references an unknown part or unit")
    # All proposal checks finish before creating the candidate.
    (output / "text").mkdir(parents=True)
    (output / "repair").mkdir()
    shutil.copytree(source / "text", output / "repair" / "source-text")
    _write(output / "repair" / "proposal.json", proposal)
    receipt = {"complete": False, "source_work_dir": str(source), "inputs": proposal["inputs"],
               "applied_edits": len(edits), "stages": []}
    _write(output / "repair" / "receipt.json", receipt)
    for part, rows in candidate_rows.items():
        (output / "text" / f"{part}.units.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    (output / "audio").symlink_to(source / "audio", target_is_directory=True)
    # Reuse only audio-identified transcript caches, never old spans or verdicts.
    for directory, pattern in (("align", "*.asr.json"), ("audit", "*.asr*.jsonl")):
        (output / directory).mkdir()
        for path in (source / directory).glob(pattern):
            shutil.copy2(path, output / directory / path.name)
    candidate = dataclasses.replace(cfg, work_dir=str(output), text=dataclasses.replace(cfg.text, work_dir=str(output)))
    try:
        for stage in ("align", "cut", "audit", "manifest") + (("gold",) if cfg.gold is not None else ()):
            run_stage(stage, candidate)
            receipt["stages"].append(stage)
            _write(output / "repair" / "receipt.json", receipt)
        receipt["complete"] = True
    finally:
        _write(output / "repair" / "receipt.json", receipt)
    return receipt
