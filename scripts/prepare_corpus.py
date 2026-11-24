#!/usr/bin/env python3
"""Run one stage of the corpus pipeline, or all of them, from one YAML config.

Usage:
    python scripts/prepare_corpus.py <stage> --config <yaml>      stage: source lexicon text audio align cut audit manifest gold all
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

STAGES = ("source", "lexicon", "text", "audio", "align", "cut", "audit", "manifest", "gold")
OPTIONAL = {"source": "source", "lexicon": "lexicon", "gold": "gold"}


class PipelineError(Exception):
    pass


def _require(cfg, parts, *needs):
    for rel, first in needs:
        for part in parts:
            path = os.path.join(cfg.work_dir, rel.format(part=part))
            if not os.path.isfile(path):
                raise PipelineError(f"{path} is missing: run the {first} stage first")


def _section(cfg, name):
    section = getattr(cfg, name)
    if section is None:
        raise PipelineError(f"{name}: the config has no {name} section")
    return section


def stage_source(cfg, hooks):
    from src.data.prepare.source import run_source_stage

    run_source_stage(_section(cfg, "source"))


def stage_lexicon(cfg, hooks):
    from src.data.prepare.lexicon import derive_to_file
    from src.text.normalizer import TextNormalizer

    lex = _section(cfg, "lexicon")
    if not (os.path.isdir(lex.book_dir) and any(f.endswith(lex.book_suffix) for f in os.listdir(lex.book_dir))):
        raise PipelineError(f"no <part>{lex.book_suffix} file in {lex.book_dir}: run the source stage first")
    os.makedirs(os.path.dirname(lex.path), exist_ok=True)
    derive_to_file(book_dir=lex.book_dir, rewrite_dir=lex.rewrite_dir, path=lex.path, book_suffix=lex.book_suffix,
                   rewrite_suffix=lex.rewrite_suffix, normalizer=TextNormalizer(cfg.language, cfg.text.book_config))


def stage_text(cfg, hooks):
    import dataclasses

    from src.data.prepare.text import run_text_stage

    parts = cfg.text.parts if cfg.text.parts is not None else cfg.part_names()
    run_text_stage(dataclasses.replace(cfg.text, parts=parts))


def stage_audio(cfg, hooks):
    from src.data.prepare.audio import run_audio

    run_audio(cfg.raw_dir, cfg.work_dir, cfg.part_names(), cfg.audio)


def _sha256(path):
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_cache(path, sha, config, model_id):
    try:
        with open(path, encoding="utf-8") as f:
            cache = json.load(f)
    except (OSError, ValueError):
        return None
    if (cache.get("audio_sha256") == sha) and (cache.get("config") == config) and (cache.get("model_id") == model_id):
        return cache["words"]
    return None


def _count(rows, key):
    counts = {}
    for r in rows:
        for v in ([r[key]] if isinstance(r[key], str) else r[key]):
            counts[v] = counts.get(v, 0) + 1
    return dict(sorted(counts.items()))


def _merge(counts):
    out = {}
    for c in counts:
        for k, v in c.items():
            out[k] = out.get(k, 0) + v
    return dict(sorted(out.items()))


def _align_markdown(report):
    rows = ["| part | units | duration s | asr words | asr cache | blocks | suspect blocks | seconds |",
            "|---|---|---|---|---|---|---|---|"]
    for name, p in {**report["parts"], "total": report["total"]}.items():
        rows.append(f"| {name} | {p['units']} | {p['duration_s']:.1f} | {p['asr_words']} | {p.get('asr_cache')} | "
                    f"{p['n_blocks']} | {p['n_suspect_blocks']} | {p['seconds']:.1f} |")
    rows += ["", "Status: " + json.dumps(report["total"]["status"], sort_keys=True),
             "Flags: " + json.dumps(report["total"]["flags"], sort_keys=True), ""]
    return "\n".join(rows)


def _free_gpu():
    import gc

    gc.collect()
    torch = sys.modules.get("torch")
    if (torch is not None) and torch.cuda.is_available():
        torch.cuda.empty_cache()


def stage_align(cfg, hooks):
    import dataclasses

    import soundfile as sf

    from src.data.prepare.align import align
    from src.data.prepare.align.ctc import BACKENDS, CtcAligner
    from src.data.prepare.asr import ASR_BACKENDS, SAMPLE_RATE, AsrRunner, AsrWord

    parts = cfg.part_names()
    _require(cfg, parts, ("text/{part}.units.jsonl", "text"), ("audio/{part}.16k.wav", "audio"))
    acfg, asr_hook = cfg.align, hooks.get("align_asr")
    work = Path(cfg.work_dir)
    (work / "align").mkdir(parents=True, exist_ok=True)
    wav = {p: str(work / "audio" / f"{p}.16k.wav") for p in parts}
    need = {p: (acfg.longform == "always") or (sf.info(wav[p]).frames / SAMPLE_RATE > acfg.block_max_s) for p in parts}
    model_id = asr_hook.model_id if asr_hook is not None else ASR_BACKENDS[acfg.asr.backend].model_id
    asr_config = {"window_s": acfg.asr.window_s, "overlap_s": acfg.asr.overlap_s,
                  "cut_search_s": acfg.asr.cut_search_s}
    words, cache_state, seconds = {}, {}, {}
    runner = None
    try:
        for p in (p for p in parts if need[p]):
            started = time.perf_counter()
            path, sha = work / "align" / f"{p}.asr.json", _sha256(wav[p])
            cached = _read_cache(path, sha, asr_config, model_id)
            if cached is not None:
                words[p], cache_state[p] = [AsrWord(*w) for w in cached], "hit"
            else:
                if asr_hook is not None:
                    source = asr_hook
                else:
                    runner = runner or AsrRunner(acfg.asr)
                    source = runner
                audio, _ = sf.read(wav[p], dtype="float32")
                words[p], cache_state[p] = list(source.transcribe_words(audio)), "miss"
                path.write_text(json.dumps({"audio_sha256": sha, "config": asr_config, "model_id": model_id,
                                            "words": [[w.text, w.start, w.end] for w in words[p]]},
                                           sort_keys=True, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            seconds[p] = time.perf_counter() - started
    finally:
        if runner is not None:
            runner.close()
            runner = None

    backend = hooks.get("align_backend")
    created = backend is None
    if created:
        backend = CtcAligner(BACKENDS[acfg.backend], acfg.language, acfg.device, acfg.window_s, acfg.context_s)
    report = {"config": dataclasses.asdict(acfg), "parts": {}}
    try:
        for p in parts:
            started = time.perf_counter()
            audio, _ = sf.read(wav[p], dtype="float32")
            with open(work / "text" / f"{p}.units.jsonl", encoding="utf-8") as f:
                units = [json.loads(line) for line in f if line.strip()]
            res = align(audio, units, acfg, asr_words=words.get(p), backend=backend, asr=asr_hook).to_dict()
            (work / "align" / f"{p}.align.jsonl").write_text("".join(
                json.dumps(u, ensure_ascii=False, sort_keys=True) + "\n" for u in res["units"]), encoding="utf-8")
            (work / "align" / f"{p}.meta.json").write_text(json.dumps(
                {"gaps": res["gaps"], "meta": res["meta"]}, sort_keys=True, indent=1, ensure_ascii=False) + "\n",
                encoding="utf-8")
            report["parts"][p] = {
                "units": len(units), "status": _count(res["units"], "status"), "flags": _count(res["units"], "flags"),
                "duration_s": res["meta"]["duration_s"], "asr_words": len(words[p]) if p in words else None,
                "asr_cache": cache_state.get(p), "n_blocks": res["meta"]["n_blocks"],
                "n_suspect_blocks": res["meta"]["n_suspect_blocks"],
                "first_words_repaired": res["meta"]["first_words_repaired"],
                "seconds": round(seconds.get(p, 0.0) + time.perf_counter() - started, 3)}
    finally:
        if created:
            backend = None
            _free_gpu()
    ps = list(report["parts"].values())
    report["total"] = {
        "units": sum(x["units"] for x in ps), "status": _merge(x["status"] for x in ps),
        "flags": _merge(x["flags"] for x in ps), "duration_s": sum(x["duration_s"] for x in ps),
        "asr_words": sum(x["asr_words"] for x in ps if x["asr_words"] is not None),
        "n_blocks": sum(x["n_blocks"] for x in ps), "n_suspect_blocks": sum(x["n_suspect_blocks"] for x in ps),
        "first_words_repaired": sum(x["first_words_repaired"] for x in ps),
        "seconds": round(sum(x["seconds"] for x in ps), 3)}
    (work / "align" / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    (work / "align" / "report.md").write_text(_align_markdown(report))


def stage_cut(cfg, hooks):
    from src.data.prepare.cut import run_cut

    parts = cfg.part_names()
    _require(cfg, parts, ("text/{part}.units.jsonl", "text"), ("align/{part}.align.jsonl", "align"),
             ("audio/{part}.24k.wav", "audio"))
    run_cut(cfg.work_dir, parts, cfg.cut)


def _aligner_model_id(cfg, parts):
    from src.data.prepare.align.ctc import BACKENDS

    found = {}
    for p in parts:
        path = os.path.join(cfg.work_dir, "align", f"{p}.meta.json")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                found[p] = json.load(f)["meta"]["model_id"]
    ids = sorted(set(found.values()))
    if len(ids) > 1:
        raise PipelineError(f"the align meta files name different aligner models: {found}")
    return ids[0] if ids else BACKENDS[cfg.align.backend].model_id


def stage_audit(cfg, hooks):
    from src.data.prepare.audit import run_audit

    parts = cfg.part_names()
    _require(cfg, parts, ("cut/{part}.cuts.jsonl", "cut"))
    run_audit(cfg.work_dir, parts, cfg.audit, runner=hooks.get("audit_runners"),
              aligner_model_id=_aligner_model_id(cfg, parts))


def stage_manifest(cfg, hooks):
    from src.data.prepare.manifest import run_manifest

    parts = cfg.part_names()
    _require(cfg, parts, ("text/{part}.units.jsonl", "text"), ("cut/{part}.cuts.jsonl", "cut"),
             ("align/{part}.align.jsonl", "align"), ("audit/{part}.audit.jsonl", "audit"))
    run_manifest(cfg.work_dir, parts, cfg.manifest, g2p=hooks.get("g2p"))


def stage_gold(cfg, hooks):
    from src.data.prepare.gold import run_gold

    run_gold(cfg.work_dir, cfg.part_names(), _section(cfg, "gold"))


RUNNERS = {"source": stage_source, "lexicon": stage_lexicon, "text": stage_text, "audio": stage_audio,
           "align": stage_align, "cut": stage_cut, "audit": stage_audit, "manifest": stage_manifest,
           "gold": stage_gold}


def main(argv=None, *, align_backend=None, align_asr=None, audit_runners=None, g2p=None) -> int:
    parser = argparse.ArgumentParser(description="Run one stage of the corpus pipeline from a YAML config.")
    parser.add_argument("stage", choices=[*STAGES, "all"])
    parser.add_argument("--config", required=True, help="the pipeline YAML")
    args = parser.parse_args(argv)
    hooks = {"align_backend": align_backend, "align_asr": align_asr, "audit_runners": audit_runners, "g2p": g2p}

    from src.data.prepare.config import ConfigError, load_config
    from src.data.prepare.text import TextStageError

    try:
        cfg = load_config(args.config)
        names = [s for s in STAGES if (s not in OPTIONAL) or (getattr(cfg, OPTIONAL[s]) is not None)] \
            if args.stage == "all" else [args.stage]
        for name in names:
            started = time.perf_counter()
            RUNNERS[name](cfg, hooks)
            print(f"{name}: done in {time.perf_counter() - started:.1f} s", flush=True)
    except (ConfigError, PipelineError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except TextStageError as e:
        print(e, file=sys.stderr)
        for f in e.failures[:20]:
            print(f"{f['part']} {f.get('line', '')} {' '.join(f.get('codepoints', []))}".rstrip(), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
