#!/usr/bin/env python3
"""Propose reviewed word repairs, or apply them in a new, fully rechecked candidate.

propose --config corpus.yaml --proposal repairs.json
apply --config corpus.yaml --proposal reviewed.json --output new-work-dir

Propose includes only dropped, unsplit single-unit clips and word changes agreed
by every configured judge in trusted ordinary book vocabulary. New, protected
or competing terminology is grouped in term_review; update the reviewed book
policy (text.terms) or orthographic lexicon and regenerate instead of guessing
pronunciation or correcting colloquial grammar. Case-only differences are ignored.
Review individual edits by setting reviewed=true and reviewer to an accountable
name after listening. Apply reruns align, cut, audit, manifest and configured gold.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    from src.data.prepare.config import load_config
    from src.data.prepare.repair import apply_repairs, propose_repairs
    from scripts.prepare_corpus import RUNNERS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("propose", "apply"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--proposal", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if args.action == "apply" and not args.output:
        parser.error("apply requires --output")
    cfg = load_config(args.config)
    if args.action == "propose":
        proposal = propose_repairs(cfg)
        with Path(args.proposal).open("x", encoding="utf-8") as stream:
            json.dump(proposal, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
        print(f"Suggested {len(proposal['edits'])} word edits; every edit requires review. "
              f"Grouped {len(proposal['term_review'])} book terminology decisions.")
    else:
        proposal = json.loads(Path(args.proposal).read_text(encoding="utf-8"))
        receipt = apply_repairs(cfg, proposal, args.output, lambda stage, candidate: RUNNERS[stage](candidate, {}))
        print(f"Applied {receipt['applied_edits']} reviewed edits; candidate checks completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
