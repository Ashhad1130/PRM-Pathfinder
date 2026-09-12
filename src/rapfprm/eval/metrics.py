"""ProcessBench's official metric.

For each solution the model must name the index of the FIRST erroneous step, or -1 if the
solution is clean. Scores are reported separately for the two populations and then combined:

    error_acc   = accuracy over solutions that DO contain an error   (exact index match)
    correct_acc = accuracy over solutions that do NOT                (predicting -1)
    F1          = harmonic mean of the two

The harmonic mean is what makes the benchmark hard: a grader that flags everything gets
error_acc 1.0 and correct_acc 0.0, for an F1 of 0. Both papers report the mean of this F1
across the four subsets ("average F1" = 65.8 for RetrievalPRM-7B, 69.5 for PathFinder-PRM-7B).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np

from ..config import OOD_ORDER


@dataclass(frozen=True)
class SubsetMetrics:
    subset: str
    n: int
    n_error: int
    n_correct: int
    #: None when the corresponding population is empty — an accuracy over zero samples is
    #: undefined, not zero.
    error_acc: float | None
    correct_acc: float | None
    #: None when either accuracy is undefined. Scoring an absent population as 0.0 would
    #: drag the harmonic mean to 0 and make every small pilot look like a total failure.
    f1: float | None

    @property
    def well_defined(self) -> bool:
        return self.f1 is not None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["well_defined"] = self.well_defined
        return payload


def harmonic_f1(error_acc: float | None, correct_acc: float | None) -> float | None:
    """Harmonic mean of the two accuracies.

    Returns 0.0 when an accuracy is genuinely zero — that is a real, meaningful score (a
    grader that flags everything earns it). Returns None when an accuracy is *undefined*
    because its population was empty; those two cases must not be conflated.
    """
    if error_acc is None or correct_acc is None:
        return None
    if error_acc <= 0.0 or correct_acc <= 0.0:
        return 0.0
    return 2.0 * error_acc * correct_acc / (error_acc + correct_acc)


def score_subset(subset: str, gold: list[int], predicted: list[int]) -> SubsetMetrics:
    if len(gold) != len(predicted):
        raise ValueError(f"{subset}: {len(gold)} gold labels but {len(predicted)} predictions")
    if not gold:
        raise ValueError(f"{subset}: nothing to score")

    gold_array = np.asarray(gold, dtype=int)
    predicted_array = np.asarray(predicted, dtype=int)

    is_error = gold_array != -1
    hits = gold_array == predicted_array

    n_error = int(is_error.sum())
    n_correct = int((~is_error).sum())

    # Undefined, not zero, when a population is absent. The full benchmark always has
    # both, but a `--limit` pilot or a filtered slice easily has only one.
    error_acc = float(hits[is_error].mean()) if n_error else None
    correct_acc = float(hits[~is_error].mean()) if n_correct else None

    return SubsetMetrics(
        subset=subset,
        n=len(gold),
        n_error=n_error,
        n_correct=n_correct,
        error_acc=error_acc,
        correct_acc=correct_acc,
        f1=harmonic_f1(error_acc, correct_acc),
    )


def score_all(by_subset: dict[str, tuple[list[int], list[int]]]) -> dict:
    """Score every subset and add the macro average both papers headline.

    Subsets whose F1 is undefined (an empty error or clean population) are excluded from
    the average and named in `undefined_subsets`, rather than being silently counted as 0.
    """
    per_subset = {
        subset: score_subset(subset, gold, predicted)
        for subset, (gold, predicted) in by_subset.items()
    }
    ordered = sorted(per_subset, key=lambda s: OOD_ORDER.get(s, 99))

    defined = [s for s in ordered if per_subset[s].well_defined]
    undefined = [s for s in ordered if not per_subset[s].well_defined]
    average_f1 = float(np.mean([per_subset[s].f1 for s in defined])) if defined else None

    return {
        "per_subset": {s: per_subset[s].to_dict() for s in ordered},
        "average_f1": average_f1,
        "subset_order": ordered,
        "undefined_subsets": undefined,
        "average_over": defined,
    }


def bootstrap_f1_ci(
    gold: list[int],
    predicted: list[int],
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a subset's F1, resampling solutions with replacement."""
    rng = np.random.default_rng(seed)
    gold_array = np.asarray(gold, dtype=int)
    predicted_array = np.asarray(predicted, dtype=int)
    n = len(gold_array)

    samples = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        g, p = gold_array[idx], predicted_array[idx]
        is_error = g != -1
        hits = g == p
        error_acc = float(hits[is_error].mean()) if is_error.any() else None
        correct_acc = float(hits[~is_error].mean()) if (~is_error).any() else None
        f1 = harmonic_f1(error_acc, correct_acc)
        # A resample can miss one population entirely; that draw carries no information
        # about F1, so drop it rather than scoring it 0 and dragging the interval down.
        samples[i] = np.nan if f1 is None else f1

    if np.all(np.isnan(samples)):
        return float("nan"), float("nan")
    return (
        float(np.nanquantile(samples, alpha / 2)),
        float(np.nanquantile(samples, 1 - alpha / 2)),
    )


def mcnemar(gold: list[int], predicted_a: list[int], predicted_b: list[int]) -> dict:
    """Exact-ish McNemar test on paired per-solution correctness.

    A and B grade the *same* solutions, so the paired test is the right one: it looks only
    at solutions where exactly one condition was right. Uses the normal approximation with
    continuity correction, plus the raw discordant counts so a tiny-n result can be judged
    by eye rather than by a p-value that the approximation does not support.
    """
    gold_array = np.asarray(gold, dtype=int)
    a_hits = gold_array == np.asarray(predicted_a, dtype=int)
    b_hits = gold_array == np.asarray(predicted_b, dtype=int)

    b_only = int((~a_hits & b_hits).sum())  # B fixed what A got wrong
    a_only = int((a_hits & ~b_hits).sum())  # B broke what A got right
    discordant = a_only + b_only

    if discordant == 0:
        return {"b_only": 0, "a_only": 0, "discordant": 0, "statistic": 0.0, "p_value": 1.0}

    statistic = (abs(b_only - a_only) - 1) ** 2 / discordant
    p_value = math.erfc(math.sqrt(statistic / 2.0))

    return {
        "b_only": b_only,
        "a_only": a_only,
        "discordant": discordant,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "reliable": discordant >= 25,
    }
