"""What the grader actually did, as opposed to how well it scored.

F1 says a condition got worse. It does not say whether the model stopped finding errors,
started inventing them, or found them in the wrong place — and those are three different
failures with three different explanations. This module reads a run's predictions back and
splits its behaviour into the pieces the ProcessBench metric folds together:

    false alarms   clean solutions the model flagged anyway
    misses         erroneous solutions it waved through
    early / late   erroneous solutions it flagged at the wrong step, and in which direction

`n_steps_scored` carries extra signal for free. With `early_stop` on, scoring halts at the
first step the model rejects, so a condition that scores fewer steps per solution is one
that pulls the trigger sooner.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_rows(run_dir: str | Path) -> list[dict]:
    path = Path(run_dir) / "predictions.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No predictions at {path}. Did that run finish?")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def verdict_profile(rows: list[dict]) -> dict:
    """Break one run's predictions into behaviours rather than scores."""
    if not rows:
        raise ValueError("No predictions to profile.")

    clean = [r for r in rows if r["gold_label"] == -1]
    erroneous = [r for r in rows if r["gold_label"] != -1]

    false_alarms = [r for r in clean if r["predicted_label"] != -1]
    missed = [r for r in erroneous if r["predicted_label"] == -1]
    flagged = [r for r in erroneous if r["predicted_label"] != -1]

    early = [r for r in flagged if r["predicted_label"] < r["gold_label"]]
    late = [r for r in flagged if r["predicted_label"] > r["gold_label"]]
    exact = [r for r in flagged if r["predicted_label"] == r["gold_label"]]

    def rate(part: list, whole: list) -> float | None:
        return len(part) / len(whole) if whole else None

    return {
        "n_solutions": len(rows),
        "n_clean": len(clean),
        "n_error": len(erroneous),
        #: Clean solutions the model flagged. Drives `correct_acc` down directly.
        "false_alarm_rate": rate(false_alarms, clean),
        #: Erroneous solutions the model let through.
        "miss_rate": rate(missed, erroneous),
        #: Of the erroneous solutions it did flag, where it put the blame.
        "exact_rate": rate(exact, flagged),
        "early_rate": rate(early, flagged),
        "late_rate": rate(late, flagged),
        #: How far off the flagged index was, signed, averaged over flagged solutions.
        "mean_index_offset": (
            sum(r["predicted_label"] - r["gold_label"] for r in flagged) / len(flagged)
            if flagged
            else None
        ),
        #: With early_stop on, fewer steps scored means a quicker trigger.
        "mean_steps_scored": (
            sum(r["n_steps_scored"] for r in rows) / len(rows) if rows else None
        ),
        "mean_steps_available": sum(r["n_steps"] for r in rows) / len(rows),
    }


def profile_runs(runs: list[tuple[str, str | Path]]) -> dict[str, dict]:
    """Profile several runs at once, keyed by label, over their shared solutions.

    Restricting to shared uids matters: comparing one arm's false-alarm rate against
    another's is only meaningful if both met the same solutions.
    """
    loaded = {label: {r["uid"]: r for r in load_rows(run_dir)} for label, run_dir in runs}
    shared = set.intersection(*(set(rows) for rows in loaded.values())) if loaded else set()
    if not shared:
        raise ValueError("The runs share no solutions, so their behaviour is not comparable.")
    return {
        label: verdict_profile([rows[uid] for uid in sorted(shared)])
        for label, rows in loaded.items()
    }


def similarity_bins(
    treatment_dir: str | Path, baseline_dir: str | Path, n_bins: int = 3
) -> list[dict]:
    """Does the damage depend on how relevant the retrieved references actually were?

    The C-vs-B contrast answers this between arms. This answers it *within* the treatment
    arm, which is independent evidence and free: the traces record every reference's
    similarity, so solutions can be ranked by how good their retrieval was and compared
    against the baseline's verdicts on those same solutions.

    If retrieval helps, the top bin should lose less than the bottom one. Flat bins say the
    quality of the retrieval had nothing to do with the outcome.
    """
    treatment_dir, baseline_dir = Path(treatment_dir), Path(baseline_dir)

    similarity: dict[str, list[float]] = {}
    with (treatment_dir / "traces.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            trace = json.loads(line)
            similarity.setdefault(trace["uid"], []).extend(trace.get("reference_similarities") or [])

    treatment = {r["uid"]: r for r in load_rows(treatment_dir)}
    baseline = {r["uid"]: r for r in load_rows(baseline_dir)}

    scored = [
        (uid, sum(sims) / len(sims))
        for uid, sims in similarity.items()
        if sims and uid in treatment and uid in baseline
    ]
    if len(scored) < n_bins:
        raise ValueError(
            f"Only {len(scored)} solutions carry reference similarities; cannot fill "
            f"{n_bins} bins."
        )

    scored.sort(key=lambda pair: pair[1])
    edges = [round(i * len(scored) / n_bins) for i in range(n_bins + 1)]

    bins = []
    for i in range(n_bins):
        chunk = scored[edges[i] : edges[i + 1]]
        uids = [uid for uid, _ in chunk]
        hit_t = sum(treatment[u]["predicted_label"] == treatment[u]["gold_label"] for u in uids)
        hit_b = sum(baseline[u]["predicted_label"] == baseline[u]["gold_label"] for u in uids)
        bins.append(
            {
                "bin": i + 1,
                "n": len(uids),
                "mean_similarity": sum(sim for _, sim in chunk) / len(chunk),
                "baseline_accuracy": hit_b / len(uids),
                "treatment_accuracy": hit_t / len(uids),
                "delta": (hit_t - hit_b) / len(uids),
            }
        )
    return bins
