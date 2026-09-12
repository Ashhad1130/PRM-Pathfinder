"""A-vs-B analysis: alignment safety and the OOD-trend claim."""

import json

import pytest

from rapfprm.analysis.compare import align, compare, ood_trend


def write_run(tmp_path, name, rows):
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return run_dir


def row(uid, subset, gold, predicted):
    return {
        "uid": uid,
        "subset": subset,
        "gold_label": gold,
        "predicted_label": predicted,
        "n_steps": 3,
        "n_steps_scored": 3,
        "predicted_error_type": None,
        "n_references_used": 0,
    }


def test_alignment_pairs_by_uid(tmp_path):
    a = write_run(tmp_path, "a", [row("x", "gsm8k", 1, 1), row("y", "gsm8k", -1, -1)])
    b = write_run(tmp_path, "b", [row("y", "gsm8k", -1, 0), row("x", "gsm8k", 1, 1)])
    aligned = align(a, b)
    assert len(aligned) == 1
    assert aligned[0].uids == ["x", "y"]
    assert aligned[0].predicted_b == [1, 0]


def test_partial_overlap_is_refused(tmp_path):
    """Comparing 10 solutions of A against 100 of B would silently bias the delta."""
    a = write_run(tmp_path, "a", [row("x", "gsm8k", 1, 1)])
    b = write_run(tmp_path, "b", [row("x", "gsm8k", 1, 1), row("z", "gsm8k", 0, 0)])
    with pytest.raises(ValueError, match="only one run"):
        align(a, b)


def test_disagreeing_gold_labels_are_refused(tmp_path):
    a = write_run(tmp_path, "a", [row("x", "gsm8k", 1, 1)])
    b = write_run(tmp_path, "b", [row("x", "gsm8k", 2, 1)])
    with pytest.raises(ValueError, match="gold labels disagree"):
        align(a, b)


def test_compare_reports_a_positive_delta_when_b_is_better(tmp_path):
    gold = [-1, 0, 1, -1, 2, -1]
    a_predictions = [-1, 0, 9, -1, 9, -1]
    b_predictions = [-1, 0, 1, -1, 2, -1]
    a = write_run(tmp_path, "a", [row(f"u{i}", "gsm8k", g, p) for i, (g, p) in enumerate(zip(gold, a_predictions))])
    b = write_run(tmp_path, "b", [row(f"u{i}", "gsm8k", g, p) for i, (g, p) in enumerate(zip(gold, b_predictions))])

    result = compare(a, b, seed=0)
    assert result["average_delta"] > 0
    assert result["per_subset"][0]["mcnemar"]["b_only"] == 2


def test_flat_deltas_are_not_reported_as_a_trend():
    """Three identical zero deltas must not become 'gains grow with difficulty'."""
    trend = ood_trend(["gsm8k", "math", "olympiadbench", "omnimath"], [0.0, 0.0, 0.0, 0.0])
    assert trend["degenerate"] is True
    assert trend["monotonic_increasing"] is False
    assert "no pattern" in trend["interpretation"]


def test_a_real_increasing_trend_is_detected():
    trend = ood_trend(
        ["gsm8k", "math", "olympiadbench", "omnimath"], [0.00, 0.02, 0.05, 0.09]
    )
    assert trend["slope_per_ood_rank"] > 0
    assert trend["spearman"] == pytest.approx(1.0)
    assert trend["monotonic_increasing"] is True
    assert "distribution shift" in trend["interpretation"]


def test_a_decreasing_trend_is_not_dressed_up():
    trend = ood_trend(
        ["gsm8k", "math", "olympiadbench", "omnimath"], [0.09, 0.05, 0.02, -0.01]
    )
    assert trend["slope_per_ood_rank"] < 0
    assert "no clear" in trend["interpretation"]


def test_two_subsets_are_not_enough_for_a_trend():
    trend = ood_trend(["gsm8k", "math"], [0.0, 0.1])
    assert trend["testable"] is False
