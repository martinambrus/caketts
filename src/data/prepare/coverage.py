"""
Coverage and split stage of the corpus pipeline (design D5): clips -> train, val, test and one contiguous held-out block
so that no paragraph, neighbouring text or identical audio sits on both sides, plus counts and hours per tag and split.

Pure functions over items in book order: {id, part, chapter, paragraph, tags, duration_s, eligible, sha256, text}.
A group is one paragraph (part, chapter, paragraph); durations are summed in seconds and reported in hours.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

SPLITS = ("train", "val", "test", "heldout")
STRUCTURAL = ("number", "date", "time", "roman", "abbreviation")
EPS = 1e-9


class SplitError(ValueError):
    """A split invariant failed; the message names item ids and splits, never text."""


@dataclass
class SplitConfig:
    seed: int = 0
    quota_max: int = 5
    quota_share: float = 0.10
    min_tag_clips: int = 3
    heldout_share: float = 0.04
    text_overlap_words: int = 4
    thin_clips: int = 30
    thin_hours: float = 0.1


@dataclass
class SplitResult:
    assignment: dict
    excluded: list
    unmet: dict
    skipped_tags: dict
    quotas: dict
    heldout: Optional[dict] = None


def group_key(item: dict) -> tuple:
    return (item["part"], item["chapter"], item["paragraph"])


def clip_tags(clip: dict, units_by_id: dict) -> list:
    tags = set()
    for unit_id in clip["units"]:
        tags.update(units_by_id[unit_id.split(".")[0]]["tags"])
    return sorted(tags)


def _hash(seed: int, first_id: str) -> str:
    return hashlib.sha1(f"{seed}:{first_id}".encode()).hexdigest()


def _tokens(text: str) -> list:
    return re.findall(r"\w+", text.lower())


def _paragraphs(items: Sequence[dict]) -> list:
    out = []
    for x in items:
        if out and group_key(out[-1][0]) == group_key(x):
            out[-1].append(x)
        else:
            out.append([x])
    return out


def _runs(items: Sequence[dict]) -> list:
    runs = []
    for para in _paragraphs(items):
        if runs and (runs[-1][-1][0]["part"], runs[-1][-1][0]["chapter"]) == (para[0]["part"], para[0]["chapter"]):
            runs[-1].append(para)
        else:
            runs.append([para])
    return runs


def _flat(paras: Iterable[list]) -> list:
    return [x for para in paras for x in para]


def _seconds(items: Iterable[dict]) -> float:
    return sum(x["duration_s"] for x in items)


def _shares(items: Sequence[dict]) -> dict:
    counts = Counter(t for x in items for t in x["tags"])
    return {t: n / len(items) for t, n in counts.items()}


def _distance(a: dict, b: dict) -> float:
    return round(sum(abs(a.get(t, 0.0) - b.get(t, 0.0)) for t in set(a) | set(b)), 12)


def _heldout_block(items: Sequence[dict], cfg: SplitConfig) -> Optional[list]:
    total_s = _seconds(items)
    target_s = cfg.heldout_share * total_s
    if not items or total_s < target_s - EPS:
        return None
    runs = _runs(items)
    candidates = [_flat(run) for run in runs if 0.5 * target_s - EPS <= _seconds(_flat(run)) <= 2 * target_s + EPS]
    if not candidates:
        for run in runs:
            for start in range(len(run)):
                acc = 0.0
                for end in range(start, len(run)):
                    acc += _seconds(run[end])
                    if acc >= target_s - EPS:
                        candidates.append(_flat(run[start:end + 1]))
                        break
    if not candidates:
        return None
    corpus = _shares(items)
    return min(candidates, key=lambda c: (_distance(_shares(c), corpus), _hash(cfg.seed, c[0]["id"])))


def make_split(items: Sequence[dict], cfg: SplitConfig) -> SplitResult:
    block = _heldout_block(items, cfg)
    groups: dict = {}
    for x in items:
        groups.setdefault(group_key(x), []).append(x)
    group_split = {}
    heldout = None
    if block:
        for x in block:
            group_split[group_key(x)] = "heldout"
        heldout = {"part": block[0]["part"], "chapter": block[0]["chapter"], "clips": len(block),
                   "hours": _seconds(block) / 3600, "paragraphs": len({group_key(x) for x in block})}

    counts = Counter(t for x in items for t in x["tags"])
    quotas, skipped = {}, {}
    for tag in sorted(counts):
        if counts[tag] >= cfg.min_tag_clips:
            quotas[tag] = min(cfg.quota_max, math.ceil(round(cfg.quota_share * counts[tag], 9)))
        else:
            skipped[tag] = counts[tag]

    group_counts = {g: Counter(t for x in xs for t in x["tags"]) for g, xs in groups.items()}
    unmet = {"val": {}, "test": {}}
    for split in ("val", "test"):
        pool = [g for g, xs in groups.items() if g not in group_split and all(x["eligible"] for x in xs)]
        have = Counter()
        for tag in sorted(quotas, key=lambda t: (counts[t], t)):
            while have[tag] < quotas[tag]:
                cands = [g for g in pool if tag in group_counts[g]]
                if not cands:
                    unmet[split][tag] = quotas[tag] - have[tag]
                    break

                def score(g):
                    gain = sum(min(c, quotas[t] - have[t]) for t, c in group_counts[g].items()
                               if t in quotas and quotas[t] > have[t])
                    return gain / len(groups[g])

                best = min(cands, key=lambda g: (-score(g), _hash(cfg.seed, groups[g][0]["id"])))
                pool.remove(best)
                group_split[best] = split
                have.update(group_counts[best])

    assignment = {x["id"]: group_split.get(group_key(x), "train") for x in items}
    held_texts = {" ".join(_tokens(x["text"])) for x in items if assignment[x["id"]] != "train"}
    excluded = []
    for x in items:
        tokens = _tokens(x["text"])
        if (assignment[x["id"]] == "train" and len(tokens) >= cfg.text_overlap_words
                and " ".join(tokens) in held_texts):
            del assignment[x["id"]]
            excluded.append({"id": x["id"], "reason": "text_overlap"})
    return SplitResult(assignment, excluded, unmet, skipped, quotas, heldout)


def check_split(items: Sequence[dict], assignment: dict, hashes: dict, excluded: Iterable = ()) -> None:
    skip = {e["id"] if isinstance(e, dict) else e for e in excluded}
    missing = [x["id"] for x in items if x["id"] not in assignment and x["id"] not in skip]
    if missing:
        raise SplitError(f"items without a split: {missing[:5]}")
    by_group, by_hash = {}, {}
    for x in items:
        split = assignment.get(x["id"])
        if split is None:
            continue
        by_group.setdefault(group_key(x), set()).add(split)
        if x["id"] in hashes:
            by_hash.setdefault(hashes[x["id"]], set()).add(split)
    bad = [g for g, s in by_group.items() if len(s) > 1]
    if bad:
        raise SplitError(f"groups in more than one split: {bad[:5]}")
    bad = [h[:12] for h, s in by_hash.items() if len(s) > 1]
    if bad:
        raise SplitError(f"sha256 in more than one split: {bad[:5]}")
    pos = [i for i, x in enumerate(items) if assignment.get(x["id"]) == "heldout"]
    if pos and (pos != list(range(pos[0], pos[-1] + 1))
                or len({(items[i]["part"], items[i]["chapter"]) for i in pos}) != 1):
        raise SplitError(f"heldout is not consecutive items of one (part, chapter): {[items[i]['id'] for i in pos[:5]]}")


def _hours(seconds: float) -> float:
    return round(seconds / 3600, 6)


def coverage_table(items: Sequence[dict], assignment: dict, cfg: SplitConfig = SplitConfig()) -> dict:
    placed = [x for x in items if x["id"] in assignment]
    tags: dict = {}
    for x in placed:
        for t in x["tags"]:
            row = tags.setdefault(t, {"s": 0.0, "n": 0, "by": {s: [0, 0.0] for s in SPLITS}})
            row["n"] += 1
            row["s"] += x["duration_s"]
            row["by"][assignment[x["id"]]][0] += 1
            row["by"][assignment[x["id"]]][1] += x["duration_s"]
    out_tags = {t: {"clips": r["n"], "hours": _hours(r["s"]),
                    "by_split": {s: {"clips": n, "hours": _hours(sec)} for s, (n, sec) in r["by"].items()}}
                for t, r in sorted(tags.items())}
    splits = {}
    for s in SPLITS:
        xs = [x for x in placed if assignment[x["id"]] == s]
        splits[s] = {"clips": len(xs), "hours": _hours(_seconds(xs)), "paragraphs": len({group_key(x) for x in xs}),
                     "parts": dict(sorted(Counter(x["part"] for x in xs).items()))}
    thin = [t for t, r in out_tags.items() if r["clips"] < cfg.thin_clips or r["hours"] < cfg.thin_hours]
    structural = {t: {"clips": out_tags[t]["clips"], "hours": out_tags[t]["hours"]} if t in out_tags
                  else {"clips": 0, "hours": 0} for t in STRUCTURAL}
    return {"tags": out_tags, "splits": splits, "thin": thin, "structural": structural}


def coverage_markdown(table: dict) -> str:
    lines = ["# Coverage", "", "## Splits", "", "| split | clips | hours | paragraphs | parts |", "|---|---|---|---|---|"]
    for s, r in table["splits"].items():
        parts = ", ".join(f"{p}: {n}" for p, n in r["parts"].items())
        lines.append(f"| {s} | {r['clips']} | {r['hours']} | {r['paragraphs']} | {parts} |")
    lines += ["", "## Tags (clips / hours)", "", "| tag | total | " + " | ".join(SPLITS) + " |", "|---|---|" + "---|" * len(SPLITS)]
    for t, r in table["tags"].items():
        cells = " | ".join(f"{r['by_split'][s]['clips']} / {r['by_split'][s]['hours']}" for s in SPLITS)
        lines.append(f"| {t} | {r['clips']} / {r['hours']} | {cells} |")
    lines += ["", "## Structural tags", "", "| tag | clips | hours |", "|---|---|---|"]
    for t, r in table["structural"].items():
        lines.append(f"| {t} | {r['clips']} | {r['hours']} |")
    lines += ["", "## Thin tags", "", ", ".join(table["thin"]) if table["thin"] else "none", ""]
    return "\n".join(lines)
