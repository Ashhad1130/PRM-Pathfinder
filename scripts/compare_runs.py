"""Compare two finished runs and write the report artefacts.

    python scripts/compare_runs.py --a runs/baseline --b runs/retrieval

    # also write the table with pool-contaminated eval solutions removed from BOTH runs
    python scripts/compare_runs.py --a runs/baseline --b runs/retrieval \
        --exclude-contaminated runs/contamination.json

Outputs into `--out` (default runs/comparison):
    comparison.json   full numbers, CIs, McNemar, OOD trend
    comparison.md     the table to paste into the report
    f1_by_subset.png  grouped bars, A vs B
    ood_trend.png     Δ F1 with 95% CI against OOD severity

With --exclude-contaminated, the same four files are written again with an
`uncontaminated-` prefix. Report both: the retrieval-time guard filters a contaminated
question's *neighbours*, but the question is still graded, so the filtered table is the one
that shows the subset without that asymmetry.

Any run pair works — this is also how Condition B is compared against Condition C, the
random-reference control (`--a runs/control-random --b runs/retrieval`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _bootstrap import setup_logging  # noqa: F401

from rapfprm.analysis.compare import compare, load_contaminated_uids, to_markdown

# The report table prints deltas as "Δ F1". Printing that to a cp1252 console raises
# UnicodeEncodeError and loses the whole run's analysis over one character — the files
# themselves are always written as UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def write_report(result: dict, out_dir: Path, prefix: str = "", plots: bool = True) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{prefix}comparison.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    markdown = to_markdown(result)
    (out_dir / f"{prefix}comparison.md").write_text(markdown, encoding="utf-8")

    if plots:
        from rapfprm.analysis.plots import plot_f1_comparison, plot_ood_trend

        for plot, name in (
            (plot_f1_comparison, f"{prefix}f1_by_subset.png"),
            (plot_ood_trend, f"{prefix}ood_trend.png"),
        ):
            try:
                plot(result, out_dir / name)
            except ValueError as exc:
                # A pilot with no defined F1 has nothing to draw. That is a property of the
                # sample, not a failure of the run — the numbers are already written, so
                # say so and carry on rather than killing the pipeline at the last step.
                print(f"[skipped {name}] {exc}")

    return markdown


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True, help="baseline run directory (Condition A)")
    parser.add_argument("--b", required=True, help="retrieval run directory (Condition B)")
    parser.add_argument("--out", default="runs/comparison")
    parser.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--exclude-contaminated",
        default=None,
        metavar="CONTAMINATION_JSON",
        help="also write a second report with the flagged eval solutions dropped from both runs",
    )
    args = parser.parse_args()

    setup_logging()

    out_dir = Path(args.out)
    result = compare(args.a, args.b, seed=args.seed)
    markdown = write_report(result, out_dir, plots=not args.no_plots)
    print(markdown)

    if args.exclude_contaminated:
        excluded = load_contaminated_uids(args.exclude_contaminated)
        if not excluded:
            print(
                f"\n{args.exclude_contaminated} flags no solutions — nothing to exclude, so "
                "no second report was written."
            )
        else:
            clean = compare(
                args.a,
                args.b,
                seed=args.seed,
                exclude_uids=excluded,
                exclusion_label=(
                    f"pool-contaminated solutions removed, per {args.exclude_contaminated}"
                ),
            )
            clean_markdown = write_report(
                clean, out_dir, prefix="uncontaminated-", plots=not args.no_plots
            )
            print("\n\n" + "=" * 78)
            print("SECOND TABLE — contaminated eval solutions excluded from both conditions")
            print("=" * 78 + "\n")
            print(clean_markdown)

    print(f"\nWritten to {out_dir}")


if __name__ == "__main__":
    main()
