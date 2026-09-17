"""A-vs-B analysis: alignment safety and the OOD-trend claim."""

import json

import pytest

from rapfprm.analysis.compare import (
    align,
    compare,
    load_contaminated_uids,
    ood_trend,
    to_markdown,
)


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


# --------------------------------------------------------------------------------------
# Contamination-excluded analysis.
#
# The retrieval-time guard filters a contaminated question's NEIGHBOURS; the question is
# still graded. So Condition B meets a contaminated MATH item with more distant references
# than it meets a clean one — an asymmetry the headline table cannot show. Recomputing with
# those items dropped from BOTH conditions is what separates the two effects.
# --------------------------------------------------------------------------------------


def write_contamination(tmp_path, uids, name="contamination.json"):
    path = tmp_path / name
    path.write_text(
        json.dumps({"flagged_uids": {uid: {"verbatim": True} for uid in uids}}),
        encoding="utf-8",
    )
    return path


def test_excluded_solutions_are_dropped_from_both_conditions(tmp_path):
    rows_a = [row("keep", "math", 1, 1), row("dirty", "math", -1, -1)]
    rows_b = [row("keep", "math", 1, 0), row("dirty", "math", -1, -1)]
    a, b = write_run(tmp_path, "a", rows_a), write_run(tmp_path, "b", rows_b)

    full = compare(a, b)
    clean = compare(a, b, exclude_uids={"dirty"})

    assert full["per_subset"][0]["n"] == 2
    assert clean["per_subset"][0]["n"] == 1
    assert clean["excluded_uids"] == 1


def test_exclusion_is_refused_when_it_would_empty_the_comparison(tmp_path):
    a = write_run(tmp_path, "a", [row("x", "gsm8k", 1, 1)])
    b = write_run(tmp_path, "b", [row("x", "gsm8k", 1, 1)])
    with pytest.raises(ValueError, match="nothing left to compare"):
        compare(a, b, exclude_uids={"x"})


def test_an_unfiltered_table_never_claims_to_be_filtered(tmp_path):
    a = write_run(tmp_path, "a", [row("x", "gsm8k", 1, 1), row("y", "gsm8k", -1, -1)])
    b = write_run(tmp_path, "b", [row("x", "gsm8k", 1, 1), row("y", "gsm8k", -1, -1)])

    assert "Filtered" not in to_markdown(compare(a, b))

    labelled = compare(a, b, exclude_uids={"y"}, exclusion_label="pool-contaminated removed")
    assert "Filtered: pool-contaminated removed" in to_markdown(labelled)


def test_contaminated_uids_are_read_from_the_report(tmp_path):
    path = write_contamination(tmp_path, ["a1", "b2"])
    assert load_contaminated_uids(path) == {"a1", "b2"}


def test_an_old_contamination_report_is_rejected_rather_than_read_as_empty(tmp_path):
    """Silently excluding nothing would look exactly like a clean benchmark."""
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"per_subset": [], "total": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="flagged_uids"):
        load_contaminated_uids(path)


# --------------------------------------------------------------------------------------
# The multi-arm figure. It is the one picture the report leans on, so it must refuse to
# draw a misleading one rather than quietly produce a plausible-looking chart.
# --------------------------------------------------------------------------------------


def metrics_for(f1_by_subset):
    return {
        "per_subset": {
            s: {"subset": s, "n": 50, "f1": f1, "well_defined": f1 is not None}
            for s, f1 in f1_by_subset.items()
        },
        "average_f1": None,
    }


def test_arms_figure_is_written(tmp_path):
    from rapfprm.analysis.plots import plot_arms

    arms = [
        ("A", metrics_for({"gsm8k": 0.70, "omnimath": 0.57})),
        ("C", metrics_for({"gsm8k": 0.54, "omnimath": 0.34})),
        ("B", metrics_for({"gsm8k": 0.54, "omnimath": 0.30})),
    ]
    out = plot_arms(arms, tmp_path / "arms.png")
    assert out.exists() and out.stat().st_size > 0


def test_arms_figure_skips_a_subset_missing_from_one_arm(tmp_path):
    """Drawing a group with a hole invites comparing two arms over different data."""
    from rapfprm.analysis.plots import plot_arms

    arms = [
        ("A", metrics_for({"gsm8k": 0.70, "omnimath": 0.57})),
        ("B", metrics_for({"gsm8k": 0.54, "omnimath": None})),
    ]
    out = plot_arms(arms, tmp_path / "arms.png")
    assert out.exists()


def test_arms_figure_refuses_when_nothing_is_comparable(tmp_path):
    from rapfprm.analysis.plots import plot_arms

    arms = [
        ("A", metrics_for({"gsm8k": None})),
        ("B", metrics_for({"gsm8k": 0.54})),
    ]
    with pytest.raises(ValueError, match="nothing comparable"):
        plot_arms(arms, tmp_path / "arms.png")


def test_arms_figure_refuses_an_empty_experiment(tmp_path):
    from rapfprm.analysis.plots import plot_arms

    with pytest.raises(ValueError, match="no arms"):
        plot_arms([], tmp_path / "arms.png")


# --------------------------------------------------------------------------------------
# Pooled contrast. Per-subset McNemar runs out of discordant solutions at this sample size;
# pooling is what lets the experiment say "no effect" instead of "not enough data".
# --------------------------------------------------------------------------------------


def test_pooled_test_has_more_discordant_pairs_than_any_subset(tmp_path):
    rows_a, rows_b = [], []
    for subset in ("gsm8k", "math", "olympiadbench", "omnimath"):
        for i in range(10):
            gold = 1 if i % 2 else -1
            rows_a.append(row(f"{subset}-{i}", subset, gold, gold))
            # B disagrees on one solution per subset.
            rows_b.append(row(f"{subset}-{i}", subset, gold, 0 if i == 3 else gold))
    a, b = write_run(tmp_path, "a", rows_a), write_run(tmp_path, "b", rows_b)

    result = compare(a, b)
    pooled = result["overall"]["mcnemar_pooled"]["discordant"]
    per_subset_max = max(r["mcnemar"]["discordant"] for r in result["per_subset"])
    assert pooled == 4
    assert pooled > per_subset_max
    assert result["overall"]["n_solutions"] == 40


def test_pooled_contrast_reports_no_effect_when_the_arms_agree(tmp_path):
    rows = [row(f"x{i}", "gsm8k", 1 if i % 2 else -1, 1 if i % 2 else -1) for i in range(20)]
    a, b = write_run(tmp_path, "a", rows), write_run(tmp_path, "b", list(rows))

    overall = compare(a, b)["overall"]
    assert overall["significant"] is False
    assert overall["mcnemar_pooled"]["p_value"] == 1.0


def test_average_delta_interval_excludes_zero_when_one_arm_is_clearly_worse(tmp_path):
    rows_a, rows_b = [], []
    for subset in ("gsm8k", "math", "olympiadbench", "omnimath"):
        for i in range(30):
            gold = 1 if i % 2 else -1
            rows_a.append(row(f"{subset}-{i}", subset, gold, gold))
            rows_b.append(row(f"{subset}-{i}", subset, gold, 0 if i % 3 else gold))
    a, b = write_run(tmp_path, "a", rows_a), write_run(tmp_path, "b", rows_b)

    overall = compare(a, b)["overall"]
    low, high = overall["average_delta_ci95"]
    assert high < 0, "B is worse on every subset; the interval must not straddle zero"
    assert overall["significant"] is True


def test_average_delta_interval_is_reported_in_the_markdown(tmp_path):
    rows = [row(f"x{i}", "gsm8k", 1 if i % 2 else -1, 1 if i % 2 else -1) for i in range(12)]
    a, b = write_run(tmp_path, "a", rows), write_run(tmp_path, "b", list(rows))

    text = to_markdown(compare(a, b))
    assert "Whole experiment" in text
    assert "Pooled McNemar" in text


# --------------------------------------------------------------------------------------
# Verdict profiling. F1 folds three different failures into one number; these tests pin
# that the profile takes them apart the way the report claims it does.
# --------------------------------------------------------------------------------------


def pred(uid, subset, gold, predicted, n_steps=6, n_steps_scored=6):
    r = row(uid, subset, gold, predicted)
    r["n_steps"] = n_steps
    r["n_steps_scored"] = n_steps_scored
    return r


def test_false_alarms_are_counted_over_clean_solutions_only():
    from rapfprm.analysis.verdicts import verdict_profile

    rows = [
        pred("c1", "gsm8k", -1, 2),   # clean, flagged -> false alarm
        pred("c2", "gsm8k", -1, -1),  # clean, passed
        pred("e1", "gsm8k", 3, 3),    # error, caught exactly
    ]
    profile = verdict_profile(rows)
    assert profile["false_alarm_rate"] == 0.5
    assert profile["n_clean"] == 2 and profile["n_error"] == 1


def test_early_and_late_blame_are_distinguished():
    """Flagging step 1 of a solution that breaks at step 4 is a different error than 6."""
    from rapfprm.analysis.verdicts import verdict_profile

    rows = [
        pred("e1", "gsm8k", 4, 1),
        pred("e2", "gsm8k", 4, 6),
        pred("e3", "gsm8k", 4, 4),
    ]
    profile = verdict_profile(rows)
    assert profile["early_rate"] == 1 / 3
    assert profile["late_rate"] == 1 / 3
    assert profile["exact_rate"] == 1 / 3
    assert profile["mean_index_offset"] == pytest.approx(((1 - 4) + (6 - 4) + 0) / 3)


def test_misses_do_not_count_as_misplaced_blame():
    from rapfprm.analysis.verdicts import verdict_profile

    rows = [pred("e1", "gsm8k", 4, -1), pred("e2", "gsm8k", 4, 4)]
    profile = verdict_profile(rows)
    assert profile["miss_rate"] == 0.5
    assert profile["exact_rate"] == 1.0, "rates over flagged solutions exclude the miss"


def test_profiles_compare_only_shared_solutions(tmp_path):
    from rapfprm.analysis.verdicts import profile_runs

    a = write_run(tmp_path, "a", [pred("x", "gsm8k", -1, -1), pred("y", "gsm8k", -1, 1)])
    b = write_run(tmp_path, "b", [pred("x", "gsm8k", -1, 1)])
    profiles = profile_runs([("A", a), ("B", b)])
    assert profiles["A"]["n_solutions"] == 1 == profiles["B"]["n_solutions"]
    assert profiles["A"]["false_alarm_rate"] == 0.0
    assert profiles["B"]["false_alarm_rate"] == 1.0


def test_profiling_disjoint_runs_is_refused(tmp_path):
    from rapfprm.analysis.verdicts import profile_runs

    a = write_run(tmp_path, "a", [pred("x", "gsm8k", -1, -1)])
    b = write_run(tmp_path, "b", [pred("z", "gsm8k", -1, -1)])
    with pytest.raises(ValueError, match="share no solutions"):
        profile_runs([("A", a), ("B", b)])


def write_traces(run_dir, rows):
    with (run_dir / "traces.jsonl").open("w", encoding="utf-8") as handle:
        for uid, sims in rows:
            handle.write(json.dumps({"uid": uid, "reference_similarities": sims}) + "\n")


def test_similarity_bins_rank_by_retrieval_quality(tmp_path):
    """A flat delta across bins is the claim; the binning itself has to be right."""
    from rapfprm.analysis.verdicts import similarity_bins

    treat_rows = [pred(f"u{i}", "gsm8k", 1, 1 if i % 2 else 0) for i in range(6)]
    base_rows = [pred(f"u{i}", "gsm8k", 1, 1) for i in range(6)]
    t = write_run(tmp_path, "t", treat_rows)
    b = write_run(tmp_path, "b", base_rows)
    write_traces(t, [(f"u{i}", [i / 10]) for i in range(6)])

    bins = similarity_bins(t, b, n_bins=3)
    assert [bin_["n"] for bin_ in bins] == [2, 2, 2]
    assert bins[0]["mean_similarity"] < bins[-1]["mean_similarity"]
    assert bins[0]["baseline_accuracy"] == 1.0


def test_similarity_bins_need_enough_solutions(tmp_path):
    from rapfprm.analysis.verdicts import similarity_bins

    t = write_run(tmp_path, "t", [pred("u0", "gsm8k", 1, 1)])
    b = write_run(tmp_path, "b", [pred("u0", "gsm8k", 1, 1)])
    write_traces(t, [("u0", [0.5])])
    with pytest.raises(ValueError, match="cannot fill"):
        similarity_bins(t, b, n_bins=3)
