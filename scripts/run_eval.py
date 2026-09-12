"""Run one condition over ProcessBench.

    python scripts/run_eval.py --config configs/baseline.yaml     # Condition A
    python scripts/run_eval.py --config configs/retrieval.yaml    # Condition B

Handy overrides for pilots:

    --limit 20            cap solutions per subset
    --subsets gsm8k math  evaluate only these subsets
    --name pilot-a        write to runs/pilot-a instead of the config's run name
"""

from __future__ import annotations

import argparse
import json

from _bootstrap import setup_logging  # noqa: F401

from rapfprm.config import load_config
from rapfprm.eval.runner import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None, help="cap solutions per subset")
    parser.add_argument("--subsets", nargs="+", default=None)
    parser.add_argument("--name", default=None, help="override run.name")
    parser.add_argument("--backend", default=None, choices=["mock", "hf", "hf4bit", "int4"])
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip solutions already in the run's predictions.jsonl and append to it",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    overrides: dict = {}
    if args.limit is not None:
        overrides["data.limit_per_subset"] = args.limit
    if args.subsets:
        overrides["data.subsets"] = args.subsets
    if args.name:
        overrides["run.name"] = args.name
    if args.backend:
        overrides["prm.backend"] = args.backend
    if args.resume:
        overrides["run.resume"] = True

    cfg = load_config(args.config, overrides or None)
    summary = run_evaluation(cfg)

    print(json.dumps(summary["metrics"], indent=2))
    print(f"\nRun directory: {cfg.run_dir}")


if __name__ == "__main__":
    main()
