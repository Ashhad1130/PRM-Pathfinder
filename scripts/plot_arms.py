"""Draw every arm of the experiment on one axis.

    python scripts/plot_arms.py --runs A=runs/int4-a C=runs/int4-c B=runs/int4-b \
        --out runs/comparison/arms.png

Each argument is LABEL=RUN_DIR, drawn left to right in the order given, so put the
baseline first. The pairwise tables in runs/comparison*/ carry the statistics; this is the
one picture that shows what happened.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import setup_logging  # noqa: F401

from rapfprm.analysis.plots import plot_arms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, metavar="LABEL=RUN_DIR")
    parser.add_argument("--out", default="runs/comparison/arms.png")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    setup_logging()

    arms = []
    for spec in args.runs:
        if "=" not in spec:
            raise SystemExit(f"Expected LABEL=RUN_DIR, got {spec!r}")
        label, run_dir = spec.split("=", 1)
        metrics_path = Path(run_dir) / "metrics.json"
        if not metrics_path.exists():
            raise SystemExit(f"No metrics.json in {run_dir}. Did that run finish?")
        arms.append((label, json.loads(metrics_path.read_text(encoding="utf-8"))))

    path = plot_arms(arms, args.out, title=args.title)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
