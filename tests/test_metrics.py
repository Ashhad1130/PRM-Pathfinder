"""ProcessBench metric behaviour, including the failure modes it is designed to punish."""

import pytest

from rapfprm.eval.metrics import bootstrap_f1_ci, harmonic_f1, mcnemar, score_all, score_subset


def test_perfect_predictions_score_one():
    gold = [-1, 0, 2, -1]
    metrics = score_subset("gsm8k", gold, list(gold))
    assert metrics.error_acc == 1.0
    assert metrics.correct_acc == 1.0
    assert metrics.f1 == 1.0


def test_flagging_everything_scores_zero():
    """The whole point of the harmonic mean: a paranoid grader gets nothing."""
    gold = [-1, -1, 1, 2]
    predicted = [0, 0, 1, 2]  # every clean solution wrongly flagged
    metrics = score_subset("math", gold, predicted)
    assert metrics.error_acc == 1.0
    assert metrics.correct_acc == 0.0
    assert metrics.f1 == 0.0


def test_flagging_nothing_scores_zero():
    gold = [-1, -1, 1, 2]
    predicted = [-1, -1, -1, -1]
    metrics = score_subset("math", gold, predicted)
    assert metrics.correct_acc == 1.0
    assert metrics.error_acc == 0.0
    assert metrics.f1 == 0.0


def test_right_error_wrong_index_counts_as_wrong():
    """ProcessBench needs the exact first-error index, not just 'something is wrong'."""
    metrics = score_subset("omnimath", [2], [1])
    assert metrics.error_acc == 0.0


def test_harmonic_f1_matches_hand_computation():
    assert harmonic_f1(0.6, 0.4) == pytest.approx(0.48)
    assert harmonic_f1(0.0, 0.9) == 0.0


def test_score_all_orders_by_ood_severity_and_averages():
    result = score_all(
        {
            "omnimath": ([-1, 0], [-1, 0]),
            "gsm8k": ([-1, 0], [-1, 1]),
        }
    )
    assert result["subset_order"] == ["gsm8k", "omnimath"]
    assert result["average_f1"] == pytest.approx((0.0 + 1.0) / 2)


def test_f1_is_undefined_not_zero_when_there_are_no_clean_solutions():
    """A --limit pilot can sample only error cases. That is not a score of 0.

    Scoring the absent population as 0 would drag the harmonic mean to 0 and make every
    small pilot look like total failure — which is exactly what it used to do.
    """
    metrics = score_subset("gsm8k", [1, 1], [1, -1])
    assert metrics.n_error == 2
    assert metrics.n_correct == 0
    assert metrics.error_acc == 0.5
    assert metrics.correct_acc is None
    assert metrics.f1 is None
    assert metrics.well_defined is False


def test_f1_is_undefined_when_there_are_no_error_solutions():
    metrics = score_subset("gsm8k", [-1, -1], [-1, 0])
    assert metrics.error_acc is None
    assert metrics.correct_acc == 0.5
    assert metrics.f1 is None


def test_genuine_zero_is_kept_distinct_from_undefined():
    """error_acc == 0.0 with a non-empty population is a real score, and must stay 0.0."""
    metrics = score_subset("gsm8k", [-1, 1], [-1, 0])
    assert metrics.error_acc == 0.0
    assert metrics.correct_acc == 1.0
    assert metrics.f1 == 0.0
    assert metrics.well_defined is True


def test_undefined_subsets_are_excluded_from_the_average():
    result = score_all(
        {
            "gsm8k": ([-1, 0], [-1, 0]),   # well defined, F1 = 1.0
            "math": ([1, 1], [1, 1]),      # no clean cases -> undefined
        }
    )
    assert result["undefined_subsets"] == ["math"]
    assert result["average_over"] == ["gsm8k"]
    assert result["average_f1"] == pytest.approx(1.0)


def test_average_is_none_when_nothing_is_defined():
    result = score_all({"gsm8k": ([1, 1], [1, 1])})
    assert result["average_f1"] is None


def test_harmonic_f1_distinguishes_none_from_zero():
    assert harmonic_f1(None, 0.9) is None
    assert harmonic_f1(0.9, None) is None
    assert harmonic_f1(0.0, 0.9) == 0.0


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError):
        score_subset("gsm8k", [-1, 0], [-1])


def test_bootstrap_ci_brackets_the_point_estimate():
    gold = [-1, 0, 1, -1, 2, -1, 0, 1]
    predicted = [-1, 0, 1, -1, 1, -1, 0, 0]
    point = score_subset("math", gold, predicted).f1
    low, high = bootstrap_f1_ci(gold, predicted, n_resamples=400, seed=1)
    assert low <= point <= high


def test_mcnemar_counts_only_discordant_pairs():
    gold = [0, 1, 2, -1]
    a = [0, 9, 9, -1]  # right on 2
    b = [0, 1, 9, 9]   # right on 2, but a different 2
    result = mcnemar(gold, a, b)
    assert result["b_only"] == 1   # B fixed index 1
    assert result["a_only"] == 1   # B broke index 3
    assert result["discordant"] == 2
    assert result["reliable"] is False


def test_mcnemar_is_neutral_when_conditions_agree():
    gold = [0, 1, -1]
    result = mcnemar(gold, gold, list(gold))
    assert result["discordant"] == 0
    assert result["p_value"] == 1.0
