"""How each condition's verdicts differ, not just how well they scored.

    python scripts/error_analysis.py --runs A=runs/int4-a C=runs/int4-c B=runs/int4-b

Reports false alarms, misses and misplaced blame side by side over the solutions all the
runs share. Use it to say *what* a condition did wrong; the F1 tables in runs/comparison*/
only say how much.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import setup_logging  # noqa: F401

from rapfprm.analysis.verdicts import profile_runs, similarity_bins

ROWS = (
    ("false alarms (clean flagged)", "false_alarm_rate", "%"),
    ("misses (errors waved through)", "miss_rate", "%"),
    ("exact index, of those flagged", "exact_rate", "%"),
    ("flagged too early", "early_rate", "%"),
    ("flagged too late", "late_rate", "%"),
    ("mean index offset", "mean_index_offset", "f"),
    ("mean steps scored", "mean_steps_scored", "f"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, metavar="LABEL=RUN_DIR")
    parser.add_argument("--out", default=None, help="also write the profile as JSON")
    parser.add_argument(
        "--similarity-bins",
        nargs=2,
        default=None,
        metavar=("TREATMENT_RUN", "BASELINE_RUN"),
        help="within the treatment arm, does the damage depend on retrieval quality?",
    )
    args = parser.parse_args()

    setup_logging()

    runs = []
    for spec in args.runs:
        if "=" not in spec:
            raise SystemExit(f"Expected LABEL=RUN_DIR, got {spec!r}")
        label, run_dir = spec.split("=", 1)
        runs.append((label, run_dir))

    profiles = profile_runs(runs)
    labels = [label for label, _ in runs]
    first = profiles[labels[0]]

    print(
        f"{first['n_solutions']} shared solutions "
        f"({first['n_error']} with an error, {first['n_clean']} clean)\n"
    )
    width = max(len(r[0]) for r in ROWS) + 2
    print(" " * width + "".join(f"{label:>12}" for label in labels))
    for title, key, kind in ROWS:
        cells = []
        for label in labels:
            value = profiles[label][key]
            if value is None:
                cells.append(f"{'n/a':>12}")
            elif kind == "%":
                cells.append(f"{100 * value:>11.1f}%")
            else:
                cells.append(f"{value:>12.2f}")
        print(f"{title:<{width}}" + "".join(cells))

    if args.similarity_bins:
        treatment, baseline = args.similarity_bins
        print()
        print("Solutions binned by mean reference similarity (within the treatment arm)")
        print()
        print(f"{'bin':>4}{'n':>6}{'mean sim':>11}{'baseline':>11}{'treatment':>12}{'delta':>9}")
        for row in similarity_bins(treatment, baseline):
            print(
                f"{row['bin']:>4}{row['n']:>6}{row['mean_similarity']:>11.3f}"
                f"{100 * row['baseline_accuracy']:>10.1f}%{100 * row['treatment_accuracy']:>11.1f}%"
                f"{100 * row['delta']:>+9.1f}"
            )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(profiles, indent=2), encoding="utf-8")
        print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
