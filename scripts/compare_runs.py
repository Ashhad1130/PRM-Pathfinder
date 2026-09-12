"""Compare two finished runs and write the report artefacts.

    python scripts/compare_runs.py --a runs/baseline --b runs/retrieval

Outputs into `--out` (default runs/comparison):
    comparison.json   full numbers, CIs, McNemar, OOD trend
    comparison.md     the table to paste into the report
    f1_by_subset.png  grouped bars, A vs B
    ood_trend.png     Δ F1 with 95% CI against OOD severity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import setup_logging  # noqa: F401

from rapfprm.analysis.compare import compare, to_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True, help="baseline run directory (Condition A)")
    parser.add_argument("--b", required=True, help="retrieval run directory (Condition B)")
    parser.add_argument("--out", default="runs/comparison")
    parser.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    setup_logging()

    result = compare(args.a, args.b, seed=args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    markdown = to_markdown(result)
    (out_dir / "comparison.md").write_text(markdown, encoding="utf-8")

    if not args.no_plots:
        from rapfprm.analysis.plots import plot_f1_comparison, plot_ood_trend

        plot_f1_comparison(result, out_dir / "f1_by_subset.png")
        plot_ood_trend(result, out_dir / "ood_trend.png")

    print(markdown)
    print(f"\nWritten to {out_dir}")


if __name__ == "__main__":
    main()
