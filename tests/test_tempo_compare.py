"""Offline tempo controls use numeric synthetic alignments, never corpus text/audio."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.data.tempo import PAUSE_KINDS
from src.data.tempo_compare import compare_tempo


def inputs():
    # Pads span several edge tokens; the internal quote crosses the pause threshold.
    kinds = ["quote", "boundary", "phone", "quote", "phone", "punct", "quote"]
    frames = [5, 10, 30, 20, 30, 5, 10]
    manifest = [{"id": "a-001", "sha256": "a" * 64, "n_tokens": len(kinds),
                 "duration_s": 1.1, "session": "a", "chapter": 1,
                 "articulation_rate": 2.5, "pause_ratio": 0.0,
                 "text": "PRIVATE_SYNTHETIC_MARKER", "audio": "/private/marker.wav"}]
    mas = [{"id": "a-001", "sha256": "a" * 64, "kinds": kinds, "durations_frames": frames}]
    cuts = [{"id": "a-001", "sha256": "a" * 64, "pad_start_s": 0.15, "pad_end_s": 0.15}]
    return manifest, mas, cuts


def test_pad_intersection_and_both_quote_policies():
    report = compare_tempo(*inputs(), hop_s=0.01)
    row = report["rows"][0]
    assert row["retained_s"] == pytest.approx(0.8)
    speech, pause = row["quote_as_speech"], row["quote_as_pause"]
    assert speech["articulation_rate"] == pytest.approx(2.5)
    assert speech["pause_ratio"] == 0
    assert speech["delta_articulation_rate"] == pytest.approx(0)
    assert pause["articulation_rate"] == pytest.approx(2 / 0.6)
    assert pause["pause_ratio"] == pytest.approx(0.25)
    assert pause["delta_pause_ratio"] == pytest.approx(0.25)
    assert row["quote_policy_delta"]["articulation_rate"] == pytest.approx(2 / 0.6 - 2.5)
    assert PAUSE_KINDS == {"punct", "boundary"}
    assert "PRIVATE_SYNTHETIC_MARKER" not in json.dumps(report)
    assert "/private/marker.wav" not in json.dumps(report)


def test_groups_counts_and_missing_word_labels():
    manifest, mas, cuts = inputs()
    for i, session, chapter, rate in [(2, "a", 2, 1.5), (3, "b", 1, None)]:
        m, a, c = copy.deepcopy(manifest[0]), copy.deepcopy(mas[0]), copy.deepcopy(cuts[0])
        m.update(id=f"{session}-{i:03}", session=session, chapter=chapter)
        a["id"] = c["id"] = m["id"]
        if rate is None:
            del m["articulation_rate"], m["pause_ratio"]
        else:
            m["articulation_rate"] = rate
        manifest.append(m); mas.append(a); cuts.append(c)
    report = compare_tempo(manifest, mas, cuts, hop_s=0.01)
    assert report["summary"]["count"] == 3
    assert report["summary"]["compared_count"] == 2
    assert report["summary"]["missing_word_labels"] == 1
    assert report["summary"]["quote_as_speech"]["delta_articulation_rate"] == pytest.approx(0.5)
    assert len(report["sessions"]) == 2
    assert len(report["chapters"]) == 3  # chapters are scoped by session
    assert report["rows"][2]["quote_as_pause"]["delta_articulation_rate"] is None


def test_short_quote_stays_speech_and_subframe_pad_is_exact():
    manifest, mas, cuts = inputs()
    mas[0]["durations_frames"] = [5, 10, 30, 11, 39, 5, 10]
    manifest[0]["duration_s"] = 1.105  # mel drops the last half frame
    cuts[0]["pad_start_s"] = 0.155
    report = compare_tempo(manifest, mas, cuts, hop_s=0.01)
    row = report["rows"][0]
    assert row["retained_s"] == pytest.approx(0.8)
    assert row["quote_policy_delta"]["articulation_rate"] == pytest.approx(0)


@pytest.mark.parametrize("mutation,match", [
    (lambda m, a, c: a[0]["kinds"].pop(), "length"),
    (lambda m, a, c: m[0].update(n_tokens=8), "n_tokens"),
    (lambda m, a, c: a[0]["durations_frames"].__setitem__(0, -1), "duration"),
    (lambda m, a, c: a[0]["durations_frames"].__setitem__(0, 5.0), "integral"),
    (lambda m, a, c: a[0]["durations_frames"].__setitem__(0, True), "integral"),
    (lambda m, a, c: a[0]["durations_frames"].__setitem__(0, float("nan")), "integral"),
    (lambda m, a, c: a[0]["durations_frames"].__setitem__(0, float("inf")), "integral"),
    (lambda m, a, c: a[0]["durations_frames"].__setitem__(0, 6), "frame count"),
    (lambda m, a, c: a[0]["kinds"].__setitem__(0, "unknown"), "kind"),
    (lambda m, a, c: a[0].update(sha256="b" * 64), "sha256"),
    (lambda m, a, c: c[0].update(sha256="b" * 64), "sha256"),
    (lambda m, a, c: c[0].update(pad_start_s=float("nan")), "pad_start_s"),
    (lambda m, a, c: c[0].update(pad_end_s=-0.1), "pad_end_s"),
    (lambda m, a, c: c[0].update(pad_start_s=1), "pads"),
    (lambda m, a, c: m[0].update(articulation_rate=float("inf")), "articulation_rate"),
    (lambda m, a, c: m[0].update(pause_ratio=1.1), "pause_ratio"),
    (lambda m, a, c: m[0].pop("pause_ratio"), "word labels"),
    (lambda m, a, c: a[0].pop("id"), "id"),
    (lambda m, a, c: a[0].update(id=""), "id"),
    (lambda m, a, c: a.append(copy.deepcopy(a[0])), "duplicate"),
    (lambda m, a, c: m.append(copy.deepcopy(m[0])), "duplicate"),
    (lambda m, a, c: c.append(copy.deepcopy(c[0])), "duplicate"),
    (lambda m, a, c: a.clear(), "MAS.*ids"),
    (lambda m, a, c: a[0].update(id="other"), "MAS.*ids"),
    (lambda m, a, c: c.clear(), "cut.*ids"),
])
def test_invalid_exports_fail(mutation, match):
    data = inputs()
    mutation(*data)
    with pytest.raises(ValueError, match=match):
        compare_tempo(*data, hop_s=0.01)


def test_cli_real_input_boundary_and_missing_export(tmp_path):
    root = Path(__file__).resolve().parents[1]
    paths = []
    for name, rows in zip(("manifest", "mas", "cuts"), inputs()):
        path = tmp_path / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        paths.append(path)
    command = [sys.executable, str(root / "scripts/compare_tempo.py"), "--manifest", str(paths[0]),
               "--mas", str(paths[1]), "--cuts", str(paths[2]), "--hop-s", "0.01"]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["summary"]["compared_count"] == 1
    paths[1].unlink()
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "MAS export missing" in result.stderr
    assert not result.stdout


def test_quote_at_threshold_and_extra_cut_rows():
    manifest, mas, cuts = inputs()
    mas[0]["durations_frames"] = [5, 10, 30, 12, 38, 5, 10]
    cuts.append({"id": "dropped-001", "sha256": "b" * 64,
                 "pad_start_s": 0.1, "pad_end_s": 0.1})
    report = compare_tempo(manifest, mas, cuts, hop_s=0.01)
    row = report["rows"][0]
    assert row["quote_as_pause"]["pause_ratio"] == pytest.approx(0.12 / 0.8)
    assert row["quote_as_speech"]["pause_ratio"] == 0


@pytest.mark.parametrize("hop,threshold", [(0, 0.12), (-1, 0.12), (float("nan"), 0.12),
                                          (0.01, float("inf")), (0.01, 0)])
def test_invalid_settings(hop, threshold):
    with pytest.raises(ValueError, match="settings"):
        compare_tempo(*inputs(), hop_s=hop, min_pause_s=threshold)


def test_zero_frame_tokens_are_valid_but_positive_phone_time_is_required():
    manifest, mas, cuts = inputs()
    mas[0]["durations_frames"][0:2] = [0, 15]
    assert compare_tempo(manifest, mas, cuts, hop_s=0.01)["rows"][0]["retained_s"] == pytest.approx(0.8)
    mas[0]["durations_frames"] = [5, 10, 0, 80, 0, 5, 10]
    with pytest.raises(ValueError, match="positive phone duration"):
        compare_tempo(manifest, mas, cuts, hop_s=0.01)


def test_trimmed_quote_at_threshold_is_not_lost_to_float_roundoff():
    manifest, mas, cuts = inputs()
    mas[0].update(kinds=["quote", "phone", "boundary"], durations_frames=[26, 44, 40])
    manifest[0]["n_tokens"] = 3
    cuts[0].update(pad_start_s=0.14, pad_end_s=0)
    row = compare_tempo(manifest, mas, cuts, hop_s=0.01)["rows"][0]
    assert row["quote_as_pause"]["pause_ratio"] == pytest.approx(0.52 / 0.96)
    assert row["quote_as_speech"]["pause_ratio"] == pytest.approx(0.4 / 0.96)
