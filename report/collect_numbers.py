"""Every number the report quotes, for all five arms, in one pass.

Run from the repository root once all arms are in results/:
    python report/collect_numbers.py

Writes the contrasts under results/<a>-vs-<b>/, the verdict profile to
results/verdict_profile.json, the arm figure to results/arms.png and docs/figures/arms.png,
then regenerates the stage figure and prints its step-level statistics.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ARMS = {
    "A": "int4-a",  # no references
    "D": "int4-d",  # retrieved, no labels
    "E": "int4-e",  # retrieved, labels as words
    "B": "int4-b",  # retrieved, labels as <+>/<->
    "C": "int4-c",  # random, labels as <+>/<->
}
CONTRASTS = [("a", "b"), ("a", "c"), ("c", "b"), ("a", "d"), ("d", "b"), ("a", "e"), ("d", "e"), ("e", "b")]


def run(*args: str) -> str:
    result = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"{' '.join(args)} failed:\n{result.stderr[-2000:]}")
    return result.stdout


def main() -> None:
    for label, run_dir in ARMS.items():
        summary = json.loads((RESULTS / run_dir / "summary.json").read_text(encoding="utf-8"))
        f1 = {k: round(100 * v["f1"], 1) for k, v in summary["metrics"]["per_subset"].items()}
        print(f"{label}  avg {100 * summary['metrics']['average_f1']:.1f}  {f1}  "
              f"retrieval={summary.get('retrieval', {}).get('mean_step_similarity')}")

    print()
    for first, second in CONTRASTS:
        out = RESULTS / f"{first}-vs-{second}"
        run("scripts/compare_runs.py", "--a", f"results/int4-{first}", "--b", f"results/int4-{second}",
            "--out", str(out), "--exclude-contaminated", "results/contamination.json")
        for name in ("comparison", "uncontaminated-comparison"):
            data = json.loads((out / f"{name}.json").read_text(encoding="utf-8"))
            overall = data["overall"]
            lo, hi = overall["average_delta_ci95"]
            test = overall["mcnemar_pooled"]
            print(f"{first.upper()}->{second.upper()} {name:26s} delta {100 * data['average_delta']:+.1f} "
                  f"[{100 * lo:+.1f}, {100 * hi:+.1f}]  {test['b_only']}/{test['a_only']} "
                  f"disc {test['discordant']}  p={test['p_value']:.2g}  reliable={test['reliable']}")

    runs = [f"{label}=results/{run_dir}" for label, run_dir in ARMS.items()]
    print()
    print(run("scripts/error_analysis.py", "--runs", *runs, "--out", "results/verdict_profile.json",
              "--similarity-bins", "results/int4-b", "results/int4-a"))
    plot_runs = [
        "A: no refs=results/int4-a", "D: refs, no labels=results/int4-d",
        "E: refs, word labels=results/int4-e", "B: refs, token labels=results/int4-b",
        "C: random refs, token labels=results/int4-c",
    ]
    run("scripts/plot_arms.py", "--out", "results/arms.png", "--runs", *plot_runs)
    run("scripts/plot_arms.py", "--out", "docs/figures/arms.png", "--runs", *plot_runs)
    print(run("report/make_figures.py"))


if __name__ == "__main__":
    main()
