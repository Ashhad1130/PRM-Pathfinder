"""Condition A vs Condition B.

Produces the table the report is built around, plus the two things a delta needs before it
can be believed: a paired significance test per subset, and the OOD-severity trend across
subsets. RetrievalPRM's case rested on that trend, so it is tested explicitly rather than
eyeballed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import OOD_ORDER
from ..eval.metrics import bootstrap_f1_ci, harmonic_f1, mcnemar, score_subset


@dataclass
class AlignedSubset:
    """Both conditions' predictions for one subset, aligned by solution uid."""

    subset: str
    uids: list[str]
    gold: list[int]
    predicted_a: list[int]
    predicted_b: list[int]


def load_predictions(run_dir: str | Path) -> dict[str, dict]:
    path = Path(run_dir) / "predictions.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No predictions at {path}. Did that run finish?")
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {row["uid"]: row for row in rows}


def load_contaminated_uids(path: str | Path) -> set[str]:
    """Read the uids `check_contamination.py` flagged as overlapping the retrieval pool."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "flagged_uids" not in payload:
        raise ValueError(
            f"{path} has no `flagged_uids` — it was written by an older version of "
            "check_contamination.py. Re-run it to record which eval solutions are affected."
        )
    return set(payload["flagged_uids"])


def align(
    run_a: str | Path, run_b: str | Path, exclude_uids: set[str] | None = None
) -> list[AlignedSubset]:
    """Pair the two runs solution by solution, refusing to compare mismatched runs.

    `exclude_uids` drops solutions from **both** conditions — the only sound way to do it.
    Dropping them from one would compare different data and the delta would be noise.
    """
    a_rows, b_rows = load_predictions(run_a), load_predictions(run_b)

    shared = sorted(set(a_rows) & set(b_rows))
    if not shared:
        raise ValueError("The two runs share no solution ids — they evaluated different data.")

    missing = (len(a_rows) - len(shared)) + (len(b_rows) - len(shared))
    if missing:
        raise ValueError(
            f"{missing} solution(s) appear in only one run ({len(a_rows)} in A, {len(b_rows)} "
            f"in B, {len(shared)} shared). Comparing a subset of one run against all of the "
            "other would bias the delta. Re-run both conditions over identical data."
        )

    if exclude_uids:
        kept = [uid for uid in shared if uid not in exclude_uids]
        if not kept:
            raise ValueError(
                f"All {len(shared)} shared solution(s) are on the exclusion list, so there "
                "is nothing left to compare."
            )
        shared = kept

    by_subset: dict[str, AlignedSubset] = {}
    for uid in shared:
        a_row, b_row = a_rows[uid], b_rows[uid]
        if a_row["gold_label"] != b_row["gold_label"]:
            raise ValueError(f"{uid}: gold labels disagree between runs — data mismatch.")

        bucket = by_subset.setdefault(
            a_row["subset"], AlignedSubset(a_row["subset"], [], [], [], [])
        )
        bucket.uids.append(uid)
        bucket.gold.append(a_row["gold_label"])
        bucket.predicted_a.append(a_row["predicted_label"])
        bucket.predicted_b.append(b_row["predicted_label"])

    return sorted(by_subset.values(), key=lambda s: OOD_ORDER.get(s.subset, 99))


def bootstrap_delta_ci(
    gold: list[int],
    predicted_a: list[int],
    predicted_b: list[int],
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Paired bootstrap CI for F1(B) - F1(A): resample solutions, not conditions."""
    rng = np.random.default_rng(seed)
    gold_array = np.asarray(gold, dtype=int)
    a_array = np.asarray(predicted_a, dtype=int)
    b_array = np.asarray(predicted_b, dtype=int)
    n = len(gold_array)

    def f1_of(g: np.ndarray, p: np.ndarray) -> float | None:
        is_error = g != -1
        hits = g == p
        error_acc = float(hits[is_error].mean()) if is_error.any() else None
        correct_acc = float(hits[~is_error].mean()) if (~is_error).any() else None
        return harmonic_f1(error_acc, correct_acc)

    deltas = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        g = gold_array[idx]
        f1_b, f1_a = f1_of(g, b_array[idx]), f1_of(g, a_array[idx])
        # Both conditions see the same resample, so either both are defined or neither is.
        deltas[i] = np.nan if f1_a is None or f1_b is None else f1_b - f1_a

    if np.all(np.isnan(deltas)):
        return float("nan"), float("nan")
    return (
        float(np.nanquantile(deltas, alpha / 2)),
        float(np.nanquantile(deltas, 1 - alpha / 2)),
    )


def bootstrap_average_delta_ci(
    aligned: list[AlignedSubset],
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Paired bootstrap CI for the macro-average delta — the headline number itself.

    Resampling happens *within* each subset, which is how the benchmark is built: the
    average is a mean over four per-subset F1s, not a score over one pooled population.
    Resampling across the pool instead would let a draw change the subset balance and would
    give an interval for a quantity nobody reports.
    """
    rng = np.random.default_rng(seed)
    arrays = [
        (
            np.asarray(s.gold, dtype=int),
            np.asarray(s.predicted_a, dtype=int),
            np.asarray(s.predicted_b, dtype=int),
        )
        for s in aligned
    ]

    def f1_of(g: np.ndarray, p: np.ndarray) -> float | None:
        is_error = g != -1
        hits = g == p
        error_acc = float(hits[is_error].mean()) if is_error.any() else None
        correct_acc = float(hits[~is_error].mean()) if (~is_error).any() else None
        return harmonic_f1(error_acc, correct_acc)

    samples = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        deltas = []
        for gold, a, b in arrays:
            idx = rng.integers(0, len(gold), size=len(gold))
            g = gold[idx]
            f1_a, f1_b = f1_of(g, a[idx]), f1_of(g, b[idx])
            if f1_a is None or f1_b is None:
                deltas = None
                break
            deltas.append(f1_b - f1_a)
        samples[i] = np.nan if deltas is None else float(np.mean(deltas))

    if np.all(np.isnan(samples)):
        return float("nan"), float("nan")
    return (
        float(np.nanquantile(samples, alpha / 2)),
        float(np.nanquantile(samples, 1 - alpha / 2)),
    )


def pooled_contrast(aligned: list[AlignedSubset], seed: int = 0) -> dict:
    """The whole experiment as one test, alongside the per-subset breakdown.

    Per-subset McNemar tests run out of discordant solutions long before they run out of
    solutions: at 50 per subset there are typically 10-20, under the ~25 the test needs to
    be trustworthy. Pooling the paired decisions across all four subsets fixes that without
    pretending the sample is larger than it is — it answers "did B and A decide differently
    over these 200 solutions", which is a different and weaker question than the per-subset
    F1 delta, and is reported as such.
    """
    gold = [g for s in aligned for g in s.gold]
    a = [p for s in aligned for p in s.predicted_a]
    b = [p for s in aligned for p in s.predicted_b]

    low, high = bootstrap_average_delta_ci(aligned, seed=seed)
    return {
        "n_solutions": len(gold),
        "average_delta_ci95": [low, high],
        #: True when the interval for the macro-average delta stays on one side of zero.
        "significant": bool(not np.isnan(low) and not np.isnan(high) and (low > 0 or high < 0)),
        "mcnemar_pooled": mcnemar(gold, a, b),
    }


def ood_trend(subsets: list[str], deltas: list[float]) -> dict:
    """Does the gain grow with OOD severity? Slope + Spearman over the severity ranks."""
    ranks = np.array([OOD_ORDER.get(s, 99) for s in subsets], dtype=float)
    values = np.array(deltas, dtype=float)

    if len(ranks) < 3:
        return {
            "testable": False,
            "reason": f"only {len(ranks)} subset(s); the trend needs at least 3",
        }

    slope, intercept = np.polyfit(ranks, values, 1)

    # Spearman = Pearson on ranks; computed here to avoid a scipy dependency.
    # Ties MUST get their shared average rank. Breaking them by array order would
    # manufacture a perfect correlation out of a flat, all-zero delta vector — which is
    # exactly what a null result looks like.
    def rank_of(values_array: np.ndarray) -> np.ndarray:
        order = values_array.argsort(kind="stable")
        out = np.empty(len(values_array), dtype=float)
        out[order] = np.arange(len(values_array), dtype=float)
        for value in np.unique(values_array):
            tied = values_array == value
            if tied.sum() > 1:
                out[tied] = out[tied].mean()
        return out

    x, y = rank_of(ranks), rank_of(values)
    spearman = 0.0 if x.std() == 0 or y.std() == 0 else float(np.corrcoef(x, y)[0, 1])

    # A trend claim needs the deltas to actually vary. With every subset landing on the
    # same delta there is no pattern to report, however tidy the arithmetic looks.
    spread = float(values.max() - values.min())
    degenerate = spread < 1e-9

    if degenerate:
        interpretation = "no pattern — every subset has the same delta"
    elif slope > 0 and spearman > 0.5:
        interpretation = "consistent with retrieval fixing distribution shift"
    else:
        interpretation = "no clear difficulty-scaling pattern"

    return {
        "testable": True,
        "slope_per_ood_rank": float(slope),
        "intercept": float(intercept),
        "spearman": spearman,
        "delta_spread": spread,
        "degenerate": degenerate,
        "monotonic_increasing": bool(np.all(np.diff(values) >= 0)) and not degenerate,
        "interpretation": interpretation,
    }


def compare(
    run_a: str | Path,
    run_b: str | Path,
    seed: int = 0,
    exclude_uids: set[str] | None = None,
    exclusion_label: str | None = None,
) -> dict:
    """Full A-vs-B analysis.

    Pass `exclude_uids` to recompute everything on a subset of the solutions — used for the
    contamination-excluded table, where the flagged eval items are dropped from both
    conditions. `exclusion_label` is carried into the output so a table can never be
    mistaken for the unfiltered one.
    """
    aligned = align(run_a, run_b, exclude_uids=exclude_uids)

    rows = []
    for subset in aligned:
        metrics_a = score_subset(subset.subset, subset.gold, subset.predicted_a)
        metrics_b = score_subset(subset.subset, subset.gold, subset.predicted_b)

        # A subset with no error cases (or no clean cases) has no defined F1, so it has no
        # defined delta either. Carry it through as None instead of inventing a number.
        defined = metrics_a.well_defined and metrics_b.well_defined
        delta = metrics_b.f1 - metrics_a.f1 if defined else None
        low, high = (
            bootstrap_delta_ci(subset.gold, subset.predicted_a, subset.predicted_b, seed=seed)
            if defined
            else (None, None)
        )

        rows.append(
            {
                "subset": subset.subset,
                "n": metrics_a.n,
                "ood_rank": OOD_ORDER.get(subset.subset, 99),
                "a": metrics_a.to_dict(),
                "b": metrics_b.to_dict(),
                "well_defined": defined,
                "delta_f1": delta,
                "delta_ci95": [low, high],
                "significant": bool(defined and (low > 0 or high < 0)),
                "mcnemar": mcnemar(subset.gold, subset.predicted_a, subset.predicted_b),
                "a_f1_ci95": list(bootstrap_f1_ci(subset.gold, subset.predicted_a, seed=seed))
                if metrics_a.well_defined
                else [None, None],
                "b_f1_ci95": list(bootstrap_f1_ci(subset.gold, subset.predicted_b, seed=seed))
                if metrics_b.well_defined
                else [None, None],
            }
        )

    usable = [r for r in rows if r["well_defined"]]
    average_a = float(np.mean([r["a"]["f1"] for r in usable])) if usable else None
    average_b = float(np.mean([r["b"]["f1"] for r in usable])) if usable else None

    return {
        "run_a": str(run_a),
        "run_b": str(run_b),
        "excluded_uids": len(exclude_uids) if exclude_uids else 0,
        "exclusion_label": exclusion_label,
        "per_subset": rows,
        "average_f1_a": average_a,
        "average_f1_b": average_b,
        "average_delta": (average_b - average_a) if usable else None,
        "overall": pooled_contrast(aligned, seed=seed),
        "undefined_subsets": [r["subset"] for r in rows if not r["well_defined"]],
        "ood_trend": ood_trend(
            [r["subset"] for r in usable], [r["delta_f1"] for r in usable]
        ),
    }


def to_markdown(result: dict) -> str:
    """Report-ready table. F1 shown on the 0-100 scale both papers use."""
    lines = [
        "# Condition A (baseline) vs Condition B (retrieval)",
        "",
        f"- A: `{result['run_a']}`",
        f"- B: `{result['run_b']}`",
    ]

    if result.get("exclusion_label"):
        lines += [
            f"- **Filtered: {result['exclusion_label']}** "
            f"({result.get('excluded_uids', 0)} solution(s) removed from both conditions). "
            "The `n` column below is what survived, so these numbers are not comparable "
            "with the unfiltered table.",
        ]

    lines += [
        "",
        "| Subset | n | OOD | F1 (A) | F1 (B) | Δ F1 | 95% CI on Δ | McNemar p | Sig. |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    severity = {0: "in-dist", 1: "moderate", 2: "severe", 3: "extreme"}

    def pct(value, spec=".1f") -> str:
        return "n/a" if value is None else format(100 * value, spec)

    for row in result["per_subset"]:
        low, high = row["delta_ci95"]
        ci = "n/a" if low is None or high is None else f"[{pct(low, '+.1f')}, {pct(high, '+.1f')}]"
        lines.append(
            f"| {row['subset']} | {row['n']} | {severity.get(row['ood_rank'], '?')} "
            f"| {pct(row['a']['f1'])} | {pct(row['b']['f1'])} | {pct(row['delta_f1'], '+.1f')} "
            f"| {ci} | {row['mcnemar']['p_value']:.3f} "
            f"| {'yes' if row['significant'] else 'no'} |"
        )

    lines += [
        f"| **average** | | | **{pct(result['average_f1_a'])}** "
        f"| **{pct(result['average_f1_b'])}** | **{pct(result['average_delta'], '+.1f')}** | | | |",
        "",
    ]

    if result.get("undefined_subsets"):
        lines += [
            "> **Undefined F1**: "
            + ", ".join(result["undefined_subsets"])
            + ". Those subsets contain only error cases or only clean cases, so the "
            "harmonic mean is undefined and they are excluded from the average — this is "
            "a property of the sample, not a score of zero. Expected on `--limit` pilots.",
            "",
        ]

    overall = result.get("overall")
    if overall:
        lo, hi = overall["average_delta_ci95"]
        mc = overall["mcnemar_pooled"]
        ci = "n/a" if lo is None or hi is None else f"[{pct(lo, '+.1f')}, {pct(hi, '+.1f')}]"
        lines += [
            "## Whole experiment",
            "",
            f"- Solutions compared: **{overall['n_solutions']}** (paired, identical in both arms)",
            f"- 95% CI on the average Δ: **{ci}** "
            f"— {'excludes' if overall['significant'] else 'includes'} zero",
            f"- Pooled McNemar: **p = {mc['p_value']:.4f}**, {mc['discordant']} discordant "
            f"({mc['b_only']} fixed by B, {mc['a_only']} broken by B), "
            f"reliable = {mc.get('reliable')}",
            "",
            "> Pooling is what makes the paired test usable at this sample size; the "
            "per-subset tests above rarely reach the ~25 discordant solutions they need. "
            "The pooled test asks whether the two arms decide differently overall, which is "
            "weaker than the per-subset F1 deltas it sits beside.",
            "",
        ]

    lines += ["## OOD-severity trend", ""]

    trend = result["ood_trend"]
    if not trend.get("testable"):
        lines.append(f"Not testable: {trend.get('reason')}.")
    else:
        lines += [
            f"- Slope per OOD rank: **{100 * trend['slope_per_ood_rank']:+.2f} F1**",
            f"- Spearman(severity, Δ): **{trend['spearman']:+.2f}**",
            f"- Monotonically increasing: **{trend['monotonic_increasing']}**",
            f"- Reading: {trend['interpretation']}",
        ]

    lines += [
        "",
        "> A McNemar test with fewer than ~25 discordant solutions is not trustworthy; check "
        "the `reliable` flag in `comparison.json` before quoting a p-value.",
    ]
    return "\n".join(lines)
