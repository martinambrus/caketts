#!/usr/bin/env python3
"""Compare manifest word tempo with an existing MAS export; no model loading.

Example (use a meaningful trained checkpoint export, not random weights):
  python scripts/compare_tempo.py --manifest manifest/train.jsonl --mas trained-mas.jsonl \
    --cuts cut/session-a.cuts.jsonl cut/session-b.cuts.jsonl --hop-s 0.010666666666666666

See src/data/tempo_compare.py for the exact minimal row contracts. JSON goes to
stdout or --output; source files are read only. Missing MAS input is an error.
No choice of quote policy is made and checkpoint quality is not verified here.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.tempo_compare import compare_tempo


def _read(path: Path, name: str) -> list:
    if not path.is_file():
        if name == "MAS":
            raise ValueError("MAS export missing: export per-token durations from a trained checkpoint first")
        raise ValueError(f"{name} input missing")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                raise ValueError(f"{name}: invalid JSON at line {line_number}") from None
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mas", type=Path, required=True, help="existing per-token MAS JSONL export")
    parser.add_argument("--cuts", type=Path, nargs="+", required=True, help="original cut JSONL files")
    parser.add_argument("--hop-s", type=float, required=True, help="mel hop in seconds used for the export")
    parser.add_argument("--min-pause-s", type=float, default=0.12, help="match the manifest word-label threshold")
    parser.add_argument("--output", type=Path, help="report JSON destination (default stdout)")
    args = parser.parse_args()
    if args.output and args.output.resolve() in {p.resolve() for p in (args.manifest, args.mas, *args.cuts)}:
        parser.error("output must differ from all input files")
    try:
        mas = _read(args.mas, "MAS")
        report = compare_tempo(_read(args.manifest, "manifest"), mas,
                               [row for path in args.cuts for row in _read(path, "cuts")],
                               hop_s=args.hop_s, min_pause_s=args.min_pause_s)
        output = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(output, encoding="utf-8")
        else:
            print(output, end="")
    except (OSError, ValueError) as error:
        parser.exit(2, f"tempo comparison failed: {error}\n")


if __name__ == "__main__":
    main()
