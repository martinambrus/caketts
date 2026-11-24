"""Offline word/MAS tempo comparison; this module never selects or writes label policy.

Minimal JSONL contracts (one object per clip):
* Manifest: id, sha256, n_tokens, duration_s, session, chapter; optional paired
  articulation_rate/pause_ratio from run_manifest's word alignment labels.
* MAS: id, sha256, kinds, durations_frames. Export the unpadded token sequence and
  valid mel frames for each clip, excluding collator batch padding. kinds must be
  in TTSDataset token order from the SAME phonemizer/config/phonetic lexicon and
  quote_before/quote_after context. Counts must match manifest n_tokens. Kinds
  alone cannot verify token symbols or checkpoint quality; export provenance is
  the caller's responsibility. Durations are JSON integers >= 0, not floats/bools.
* Cuts: id, sha256, pad_start_s, pad_end_s from the original cut records. Multiple
  cut files may include clips absent from the manifest; all ids must be unique.

MAS ids must equal manifest ids exactly. Digests are the pipeline's SHA256 of
int16 PCM, NOT WAV-file hashes. sum(durations_frames) must equal the existing
BigVGAN mel contract floor(duration_s/hop_s). A final partial frame is unrepresented;
we intersect each token's timeline with [pad_start_s, duration_s-pad_end_s],
without snapping pads to frames or assuming all padding belongs to one token.
The manifest word labels already exclude clip padding; they are not recomputed.
Both policies use the existing tempo API; only local quote->boundary mapping
adds quotes >= min_pause_s to pauses. Durations within 1e-12 seconds of the pause
threshold are normalized to it to absorb float cancellation when trimming pads.
Output contains ids/groups/counts/numbers,
never source text or audio paths. It cannot establish that an export is trained.
"""
from __future__ import annotations

import math
import re
import statistics
from collections import defaultdict
from typing import Sequence

from src.data.tempo import tempo_from_alignment

KINDS = {"phone", "stress", "punct", "boundary", "quote"}
METRICS = ("articulation_rate", "pause_ratio", "delta_articulation_rate", "delta_pause_ratio")
POLICIES = ("quote_as_speech", "quote_as_pause")


def _number(row: dict, key: str, context: str, *, positive: bool = False) -> float:
    value = row.get(key)
    if (isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value)
            or value < 0 or (positive and value == 0)):
        raise ValueError(f"{context}: {key} must be finite and {'positive' if positive else 'nonnegative'}")
    return float(value)


def _index(rows: Sequence[dict], name: str) -> dict:
    indexed = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"].strip():
            raise ValueError(f"{name}: each row requires a nonempty string id")
        cid = row["id"]
        if cid in indexed:
            raise ValueError(f"{name}: duplicate id {cid}")
        if not isinstance(row.get("sha256"), str) or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None:
            raise ValueError(f"{name} {cid}: sha256 must be 64 lowercase hex characters")
        indexed[cid] = row
    return indexed


def _median(rows: list[dict], path: tuple[str, ...]):
    values = []
    for row in rows:
        value = row
        for key in path:
            value = value[key]
        if value is not None:
            values.append(value)
    return statistics.median(values) if values else None


def _summary(rows: list[dict]) -> dict:
    compared = sum(row["word"] is not None for row in rows)
    return {"count": len(rows), "compared_count": compared, "missing_word_labels": len(rows) - compared,
            "word": {key: _median(rows, ("word_metrics", key)) for key in METRICS[:2]},
            **{policy: {key: _median(rows, (policy, key)) for key in METRICS} for policy in POLICIES},
            "quote_policy_delta": {key: _median(rows, ("quote_policy_delta", key)) for key in METRICS[:2]}}


def compare_tempo(manifest: Sequence[dict], mas: Sequence[dict], cuts: Sequence[dict], *,
                  hop_s: float, min_pause_s: float = 0.12) -> dict:
    """Return numeric comparisons and medians; raise ValueError on incompatible inputs.

    No checkpoint is loaded and no trained-export claim is inferred. Supply a
    meaningful trained MAS export before interpreting the result empirically.
    """
    hop_s = _number({"hop_s": hop_s}, "hop_s", "settings", positive=True)
    min_pause_s = _number({"min_pause_s": min_pause_s}, "min_pause_s", "settings", positive=True)
    manifests, alignments, cut_records = (_index(manifest, "manifest"), _index(mas, "MAS"), _index(cuts, "cuts"))
    if not manifests:
        raise ValueError("manifest must contain at least one clip")
    if manifests.keys() != alignments.keys():
        raise ValueError(f"MAS ids must equal manifest ids (missing={len(manifests.keys() - alignments.keys())}, "
                         f"extra={len(alignments.keys() - manifests.keys())})")
    if not manifests.keys() <= cut_records.keys():
        raise ValueError(f"cut ids missing for {len(manifests.keys() - cut_records.keys())} manifest clips")

    rows, sessions, chapters = [], defaultdict(list), defaultdict(list)
    for cid, manifest_row in manifests.items():
        alignment, cut = alignments[cid], cut_records[cid]
        if not manifest_row["sha256"] == alignment["sha256"] == cut["sha256"]:
            raise ValueError(f"{cid}: sha256 mismatch between manifest, MAS and cut records")
        session, chapter = manifest_row.get("session"), manifest_row.get("chapter")
        if not isinstance(session, str) or not session.strip():
            raise ValueError(f"{cid}: session must be a nonempty string")
        if isinstance(chapter, bool) or not (isinstance(chapter, int) and chapter >= 0 or
                                           isinstance(chapter, str) and chapter.strip()):
            raise ValueError(f"{cid}: chapter must be a nonnegative integer or nonempty string")
        kinds, durations = alignment.get("kinds"), alignment.get("durations_frames")
        if not isinstance(kinds, list) or not isinstance(durations, list) or len(kinds) != len(durations):
            raise ValueError(f"{cid}: kinds and durations_frames require arrays of equal length")
        n_tokens = manifest_row.get("n_tokens")
        if type(n_tokens) is not int or n_tokens <= 0 or n_tokens != len(kinds):
            raise ValueError(f"{cid}: MAS token length must equal positive manifest n_tokens")
        if any(not isinstance(kind, str) or kind not in KINDS for kind in kinds):
            raise ValueError(f"{cid}: unknown token kind")
        if "phone" not in kinds:
            raise ValueError(f"{cid}: at least one phone kind is required")
        if any(type(duration) is not int for duration in durations):
            raise ValueError(f"{cid}: durations_frames must be integral JSON values")
        if any(duration < 0 for duration in durations):
            raise ValueError(f"{cid}: duration frames must be nonnegative")
        duration_s = _number(manifest_row, "duration_s", cid, positive=True)
        frames = sum(durations)
        if frames != math.floor(duration_s / hop_s + 1e-6):
            raise ValueError(f"{cid}: MAS frame count differs from floor(duration_s/hop_s)")
        start_pad = _number(cut, "pad_start_s", cid)
        end_pad = _number(cut, "pad_end_s", cid)
        if start_pad + end_pad >= duration_s:
            raise ValueError(f"{cid}: pads must leave positive unpadded duration")
        left, right = start_pad / hop_s, (duration_s - end_pad) / hop_s
        corrected, cursor = [], 0
        for duration in durations:
            end = cursor + duration
            corrected.append(max(0.0, min(end, right) - max(cursor, left)))
            cursor = end
        retained_s = sum(corrected) * hop_s
        if retained_s <= 0 or not any(d > 0 for kind, d in zip(kinds, corrected) if kind == "phone"):
            raise ValueError(f"{cid}: pad-corrected MAS must contain positive phone duration")
        # Subtraction at a pad edge can put an exact 120 ms token a float ULP
        # below the threshold. Normalize only picosecond-level threshold noise.
        corrected_s = [duration * hop_s for duration in corrected]
        corrected_s = [min_pause_s if math.isclose(duration, min_pause_s, rel_tol=0, abs_tol=1e-12)
                       else duration for duration in corrected_s]
        has_rate, has_pause = "articulation_rate" in manifest_row, "pause_ratio" in manifest_row
        if has_rate != has_pause:
            raise ValueError(f"{cid}: word labels articulation_rate/pause_ratio must occur together")
        word = None
        if has_rate:
            word = {"articulation_rate": _number(manifest_row, "articulation_rate", cid, positive=True),
                    "pause_ratio": _number(manifest_row, "pause_ratio", cid)}
            if word["pause_ratio"] > 1:
                raise ValueError(f"{cid}: pause_ratio must be <= 1")
        row = {"id": cid, "session": session, "chapter": chapter, "word": word,
               "retained_s": retained_s, "pad_start_s": start_pad, "pad_end_s": end_pad,
               "unrepresented_tail_s": max(0.0, duration_s - frames * hop_s)}
        for policy in POLICIES:
            policy_kinds = ["boundary" if policy == "quote_as_pause" and kind == "quote" else kind for kind in kinds]
            label = tempo_from_alignment(policy_kinds, corrected_s, 1.0, min_pause_s)
            values = {"articulation_rate": label.articulation_rate, "pause_ratio": label.pause_ratio}
            row[policy] = {**values, **{f"delta_{key}": value - word[key] if word else None
                                      for key, value in values.items()}}
        row["quote_policy_delta"] = {key: row["quote_as_pause"][key] - row["quote_as_speech"][key]
                                     for key in METRICS[:2]}
        # Null placeholders make the same median code handle unlabeled manifest rows.
        row["word_metrics"] = word or {key: None for key in METRICS[:2]}
        rows.append(row)
        sessions[session].append(row)
        chapters[(session, chapter)].append(row)
    report = {"hop_s": hop_s, "min_pause_s": min_pause_s,
              "delta_direction": "MAS minus manifest word label; quote policy delta is pause minus speech",
              "summary": _summary(rows),
              "sessions": [{"session": key, **_summary(group)} for key, group in sessions.items()],
              "chapters": [{"session": key[0], "chapter": key[1], **_summary(group)} for key, group in chapters.items()],
              "rows": rows}
    for row in rows:
        del row["word_metrics"]
    return report
