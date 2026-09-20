"""Print the whole study on one screen once every arm has finished.

    python scripts/summarize_study.py runs A B C D E

Reads each arm's metrics.json and each contrast's comparison.json and prints the average
F1 per arm followed by every contrast with its pooled interval. Anything missing is
skipped rather than fatal: this runs after hours of compute, and a partial study should
still report what it has.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

#: Arm label -> the run directory name its config writes to.
ARM_RUNS = {
    "A": "baseline",
    "B": "retrieval",
    "C": "control-random",
    "D": "nolabels",
    "E": "wordlabels",
}

#: Contrast directory suffix -> what that comparison isolates.
CONTRASTS = (
    ("a-vs-b", "A -> B   retrieval, all in"),
    ("a-vs-c", "A -> C   extra text alone"),
    ("c-vs-b", "C -> B   relevance, length fixed"),
    ("a-vs-d", "A -> D   unlabelled references"),
    ("d-vs-b", "D -> B   adding the labels"),
    ("e-vs-b", "E -> B   labels as tokens vs words"),
    ("d-vs-e", "D -> E   the judgement, as words"),
)

PUBLISHED_BF16_AVERAGE = 69.5


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def arm_scores(out_dir: Path, arms: list[str]) -> dict[str, float]:
    scores: dict[str, float] = {}
    print("  Average F1 by arm")
    print("  " + "-" * 60)
    for arm in arms:
        run_name = ARM_RUNS.get(arm)
        metrics = read_json(out_dir / run_name / "metrics.json") if run_name else None
        if metrics is None:
            print(f"    {arm}  (no metrics; that arm did not finish)")
            continue
        average = metrics.get("average_f1")
        if average is None:
            undefined = ", ".join(metrics.get("undefined_subsets") or []) or "every subset"
            print(f"    {arm}  undefined -- {undefined} had only one population")
            continue
        scores[arm] = 100 * average
        print(f"    {arm}  {100 * average:5.1f}   ({run_name})")
    return scores


def contrast_lines(out_dir: Path) -> None:
    print("\n  Contrasts (positive means the second arm scored higher)")
    print("  " + "-" * 60)
    printed = False
    for suffix, question in CONTRASTS:
        payload = read_json(out_dir / f"comparison-{suffix}" / "comparison.json")
        if payload is None or payload.get("average_delta") is None:
            continue
        printed = True
        delta = 100 * payload["average_delta"]
        overall = payload.get("overall") or {}
        low, high = overall.get("average_delta_ci95", [None, None])
        mcnemar = overall.get("mcnemar_pooled") or {}

        interval = (
            f"[{100 * low:+6.1f}, {100 * high:+6.1f}]"
            if low is not None and high is not None
            else "[     n/a     ]"
        )
        verdict = "significant" if overall.get("significant") else "no detectable effect"
        caveat = "" if mcnemar.get("reliable") else "  (pooled test under-powered)"
        print(f"    {question:34} {delta:+6.1f}  {interval}  {verdict}{caveat}")
    if not printed:
        print("    (no contrasts available yet)")


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: summarize_study.py <out_dir> <arm> [<arm> ...]")
    out_dir = Path(sys.argv[1])
    arms = sys.argv[2:]

    scores = arm_scores(out_dir, arms)

    if "A" in scores:
        gap = scores["A"] - PUBLISHED_BF16_AVERAGE
        print(
            f"\n    published PathFinder-PRM-7B at bf16: {PUBLISHED_BF16_AVERAGE}   "
            f"this run's A: {scores['A']:.1f}  ({gap:+.1f})"
        )

    contrast_lines(out_dir)

    print(
        f"""
  Written to {out_dir}/:
    comparison-*/comparison.md        per-contrast tables, each with an uncontaminated- twin
    comparison/arms.png               every arm on one axis
    comparison/verdict_profile.json   false alarms, misses, misplaced blame

  Before quoting any number: state the backend beside it, check n_resumed in each
  summary.json before quoting throughput, and read the MATH rows against
  {out_dir}/contamination.json. docs/RESULTS.md shows how the int4 run was written up.
"""
    )


if __name__ == "__main__":
    main()
