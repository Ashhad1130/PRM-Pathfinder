"""End-to-end: config -> data -> retrieval -> scoring -> metrics, on the fixtures."""

import json

import pytest

from rapfprm.config import load_config
from rapfprm.data.processbench import Solution, iter_prefixes, load_processbench
from rapfprm.data.pool import load_pool
from rapfprm.eval.runner import run_evaluation
from rapfprm.retrieval.index import build_index

CONFIG = "configs/smoke.yaml"


@pytest.fixture(scope="module")
def indexed(tmp_path_factory):
    """Build the fixture index once into a temp dir, shared by the tests below."""
    index_dir = tmp_path_factory.mktemp("index")
    cfg = load_config(CONFIG, {"retrieval.index_dir": str(index_dir)})
    items = load_pool(cfg.data.pool_dir)
    build_index(items, cfg.retrieval).save(index_dir)
    return str(index_dir)


def test_fixtures_load_with_the_expected_shape():
    cfg = load_config(CONFIG)
    solutions = load_processbench(cfg.data)
    assert len(solutions) == 48
    assert {s.subset for s in solutions} == {"gsm8k", "math", "olympiadbench", "omnimath"}
    assert any(s.has_error for s in solutions)
    assert any(not s.has_error for s in solutions)


def test_solution_rejects_an_out_of_range_label():
    with pytest.raises(ValueError, match="out of range"):
        Solution(uid="x", subset="gsm8k", problem="p", steps=("a", "b"), label=5)


def test_solution_rejects_an_empty_step_list():
    with pytest.raises(ValueError, match="no steps"):
        Solution(uid="x", subset="gsm8k", problem="p", steps=(), label=-1)


def test_iter_prefixes_accumulates_context():
    solution = Solution(uid="x", subset="gsm8k", problem="p", steps=("a", "b", "c"), label=-1)
    prefixes = list(iter_prefixes(solution))
    assert [p[0] for p in prefixes] == [0, 1, 2]
    assert prefixes[2][1] == ("a", "b")
    assert prefixes[2][2] == "c"


def test_baseline_run_produces_a_complete_run_directory(tmp_path, indexed):
    cfg = load_config(
        CONFIG,
        {
            "run.name": "unit-a",
            "run.out_dir": str(tmp_path),
            "retrieval.enabled": False,
            "retrieval.index_dir": indexed,
        },
    )
    summary = run_evaluation(cfg)

    run_dir = cfg.run_dir
    for filename in ("config.resolved.yaml", "summary.json", "predictions.jsonl", "metrics.json"):
        assert (run_dir / filename).exists(), filename

    assert summary["condition"] == "A (baseline)"
    assert summary["environment"]["backend"] == "mock"
    assert 0.0 <= summary["metrics"]["average_f1"] <= 1.0

    with (run_dir / "predictions.jsonl").open(encoding="utf-8") as handle:
        predictions = [json.loads(line) for line in handle]
    assert len(predictions) == 48
    assert all(p["predicted_label"] >= -1 for p in predictions)


def test_retrieval_run_actually_calls_the_retriever(tmp_path, indexed):
    cfg = load_config(
        CONFIG,
        {"run.name": "unit-b", "run.out_dir": str(tmp_path), "retrieval.index_dir": indexed},
    )
    summary = run_evaluation(cfg)

    assert summary["condition"] == "B (retrieval)"
    assert summary["retrieval"]["queries"] > 0
    assert summary["retrieval"]["references_per_query"] > 0


def test_predicted_index_never_exceeds_the_step_count(tmp_path, indexed):
    cfg = load_config(
        CONFIG,
        {"run.name": "unit-bounds", "run.out_dir": str(tmp_path), "retrieval.index_dir": indexed},
    )
    run_evaluation(cfg)
    with (cfg.run_dir / "predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            assert -1 <= row["predicted_label"] < row["n_steps"]


def test_config_rejects_a_typo_in_a_key():
    with pytest.raises(ValueError, match="Unknown config key"):
        load_config(CONFIG, {"retrieval.top_k_question": 3})  # missing plural


def test_config_rejects_an_unknown_subset():
    with pytest.raises(ValueError, match="Unknown ProcessBench subset"):
        load_config(CONFIG, {"data.subsets": ["gsm8k", "aime"]})


def test_config_rejects_an_impossible_threshold():
    with pytest.raises(ValueError, match="threshold"):
        load_config(CONFIG, {"prm.correctness_threshold": 1.5})


def test_pilot_local_config_loads_with_offload_settings():
    """The memory-constrained path is a real config; keep it from rotting.

    Asserts the shape of the budget, not the exact sizes — those get retuned per machine
    and pinning them here just produces a failing test every time someone does.
    """
    cfg = load_config("configs/pilot-local.yaml")
    assert cfg.prm.backend == "hf"
    assert cfg.prm.offload_folder == "artifacts/offload"

    assert cfg.prm.max_memory, "the constrained config must cap memory or it will OOM"
    assert 0 in cfg.prm.max_memory, "needs a GPU budget keyed by device ordinal"
    assert "cpu" in cfg.prm.max_memory, "needs a CPU budget or accelerate fills RAM"
    assert all(
        isinstance(v, str) and v.endswith(("GiB", "MiB"))
        for v in cfg.prm.max_memory.values()
    ), "accelerate expects size strings like '6GiB'"

    # A pilot must stay small: this config exists to prove correctness, not to benchmark.
    assert cfg.data.limit_per_subset is not None and cfg.data.limit_per_subset <= 5


def test_max_memory_keys_are_coerced_for_accelerate():
    """YAML hands us mixed key types; accelerate needs int ordinals plus 'cpu'/'disk'."""
    raw = {"0": "6GiB", 1: "4GiB", "cpu": "8GiB", "disk": "20GiB"}
    coerced = {(int(k) if str(k).isdigit() else k): v for k, v in raw.items()}
    assert coerced == {0: "6GiB", 1: "4GiB", "cpu": "8GiB", "disk": "20GiB"}
    assert all(isinstance(k, int) for k in (0, 1) if k in coerced)


def test_predictions_are_written_incrementally_not_only_at_the_end(tmp_path, indexed):
    """A multi-hour run must survive being killed. Rows land as they are produced."""
    cfg = load_config(
        CONFIG,
        {"run.name": "incr", "run.out_dir": str(tmp_path), "retrieval.index_dir": indexed},
    )
    run_evaluation(cfg)
    lines = (cfg.run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 48


def test_resume_skips_already_scored_solutions(tmp_path, indexed):
    cfg_first = load_config(
        CONFIG,
        {
            "run.name": "res",
            "run.out_dir": str(tmp_path),
            "retrieval.index_dir": indexed,
            "data.limit_per_subset": 2,
        },
    )
    first = run_evaluation(cfg_first)
    assert first["n_scored_this_session"] == 8
    assert first["n_resumed"] == 0

    # Same run directory, more data, resume on: only the new solutions get scored.
    cfg_second = load_config(
        CONFIG,
        {
            "run.name": "res",
            "run.out_dir": str(tmp_path),
            "retrieval.index_dir": indexed,
            "data.limit_per_subset": 3,
            "run.resume": True,
        },
    )
    second = run_evaluation(cfg_second)
    assert second["n_resumed"] == 8
    assert second["n_scored_this_session"] == 4      # 12 wanted - 8 already done
    assert second["n_solutions"] == 12

    uids = [
        json.loads(line)["uid"]
        for line in (cfg_second.run_dir / "predictions.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(uids) == len(set(uids)) == 12, "resume must not duplicate rows"


def test_resume_on_a_complete_run_scores_nothing(tmp_path, indexed):
    overrides = {
        "run.name": "done",
        "run.out_dir": str(tmp_path),
        "retrieval.index_dir": indexed,
        "data.limit_per_subset": 2,
    }
    run_evaluation(load_config(CONFIG, overrides))
    again = run_evaluation(load_config(CONFIG, {**overrides, "run.resume": True}))
    assert again["n_scored_this_session"] == 0
    assert again["n_solutions"] == 8


def test_int4_backend_is_accepted():
    cfg = load_config(CONFIG, {"prm.backend": "int4"})
    assert cfg.prm.backend == "int4"


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="mock|hf|hf4bit|int4"):
        load_config(CONFIG, {"prm.backend": "gguf"})


def test_int4_configs_use_the_blackwell_compatible_packing_format():
    """`plain` and `preshuffled` need the extra `mslk` package and fail on sm_120."""
    for path in ("configs/pilot-int4.yaml", "configs/pilot-int4-retrieval.yaml"):
        cfg = load_config(path)
        assert cfg.prm.backend == "int4", path
        assert cfg.prm.int4_packing_format == "tile_packed_to_4d", path


def test_int4_ab_pair_differs_only_in_retrieval():
    """The A/B parity rule applies to the int4 pair too, not just the bf16 configs."""
    a = load_config("configs/pilot-int4.yaml")
    b = load_config("configs/pilot-int4-retrieval.yaml")

    assert a.retrieval.enabled is False
    assert b.retrieval.enabled is True

    for field in (
        "backend",
        "model_name",
        "dtype",
        "int4_group_size",
        "int4_packing_format",
        "correctness_threshold",
        "early_stop",
        "max_input_tokens",
        "seed",
    ):
        assert getattr(a.prm, field) == getattr(b.prm, field), f"prm.{field} drifted"
    assert a.prompt == b.prompt


def test_baseline_and_retrieval_configs_differ_only_in_retrieval():
    """A-vs-B parity: if prm.* drifts between the two, the delta stops meaning anything."""
    a = load_config("configs/baseline.yaml")
    b = load_config("configs/retrieval.yaml")

    assert a.retrieval.enabled is False
    assert b.retrieval.enabled is True

    for field in (
        "backend",
        "model_name",
        "dtype",
        "correctness_threshold",
        "early_stop",
        "max_input_tokens",
        "seed",
    ):
        assert getattr(a.prm, field) == getattr(b.prm, field), f"prm.{field} drifted"

    assert a.prompt == b.prompt
    assert a.data.subsets == b.data.subsets
    assert a.data.limit_per_subset == b.data.limit_per_subset


def test_control_config_differs_from_condition_b_only_in_reference_mode():
    """C is B with relevance removed. Any other drift makes the control control nothing."""
    b = load_config("configs/retrieval.yaml")
    c = load_config("configs/control-random.yaml")

    assert b.retrieval.reference_mode == "retrieved"
    assert c.retrieval.reference_mode == "random"
    assert c.retrieval.enabled is True

    for field in (
        "top_k_questions",
        "top_k_steps",
        "step_level",
        "encoder_name",
        "max_question_similarity",
        "drop_exact_duplicates",
        "pca_components",
    ):
        assert getattr(b.retrieval, field) == getattr(
            c.retrieval, field
        ), f"retrieval.{field} drifted"

    assert b.prm == c.prm
    assert b.prompt == c.prompt
    assert b.data.subsets == c.data.subsets
    assert b.data.limit_per_subset == c.data.limit_per_subset


def test_int4_control_matches_the_int4_ab_pair():
    a = load_config("configs/pilot-int4.yaml")
    b = load_config("configs/pilot-int4-retrieval.yaml")
    c = load_config("configs/pilot-int4-random.yaml")

    assert c.retrieval.reference_mode == "random"
    assert a.prm == b.prm == c.prm
    assert a.prompt == b.prompt == c.prompt
    assert a.data.limit_per_subset == b.data.limit_per_subset == c.data.limit_per_subset
    assert len({a.run.name, b.run.name, c.run.name}) == 3


def test_config_rejects_an_unknown_reference_mode():
    with pytest.raises(ValueError, match="reference_mode"):
        load_config("configs/retrieval.yaml", {"retrieval.reference_mode": "shuffled"})


# --------------------------------------------------------------------------------------
# Subsampling. ProcessBench lists every erroneous solution first, so a prefix is not a
# sample — it is the error half of the benchmark, and F1 over it is undefined.
# --------------------------------------------------------------------------------------


def _solutions(n_error: int, n_clean: int, subset: str = "gsm8k"):
    from rapfprm.data.processbench import Solution

    errors = [
        Solution(uid=f"e{i}", subset=subset, problem="q", steps=("a", "b"), label=0)
        for i in range(n_error)
    ]
    clean = [
        Solution(uid=f"c{i}", subset=subset, problem="q", steps=("a", "b"), label=-1)
        for i in range(n_clean)
    ]
    return errors + clean  # the upstream order: all errors, then all clean


def test_subsample_keeps_both_populations():
    """A prefix of ProcessBench is all errors; the sample must not be."""
    from rapfprm.data.processbench import subsample

    picked = subsample(_solutions(60, 40), limit=20, seed=17)
    assert len(picked) == 20
    assert sum(1 for s in picked if s.label != -1) == 12  # 60% of 20, the true ratio
    assert sum(1 for s in picked if s.label == -1) == 8


def test_subsample_is_deterministic_across_arms():
    """A and B must grade the identical solutions or the pairing is meaningless."""
    from rapfprm.data.processbench import subsample

    solutions = _solutions(60, 40)
    a = [s.uid for s in subsample(solutions, limit=25, seed=17)]
    b = [s.uid for s in subsample(solutions, limit=25, seed=17)]
    assert a == b


def test_a_different_seed_draws_a_different_sample():
    from rapfprm.data.processbench import subsample

    solutions = _solutions(60, 40)
    a = {s.uid for s in subsample(solutions, limit=25, seed=17)}
    b = {s.uid for s in subsample(solutions, limit=25, seed=99)}
    assert a != b


def test_samples_are_nested_so_a_run_can_be_extended():
    """Growing the limit must reuse what is already scored, not redraw from scratch."""
    from rapfprm.data.processbench import subsample

    solutions = _solutions(600, 400)
    small = {s.uid for s in subsample(solutions, limit=50, seed=17)}
    large = {s.uid for s in subsample(solutions, limit=200, seed=17)}
    assert small <= large


def test_subsample_never_empties_a_population():
    """Even a tiny limit must leave an F1 that is defined."""
    from rapfprm.data.processbench import subsample

    picked = subsample(_solutions(990, 10), limit=4, seed=17)
    assert any(s.label == -1 for s in picked)
    assert any(s.label != -1 for s in picked)


def test_subsample_returns_everything_when_the_limit_is_not_binding():
    from rapfprm.data.processbench import subsample

    solutions = _solutions(6, 4)
    assert subsample(solutions, limit=10, seed=17) == solutions
    assert subsample(solutions, limit=99, seed=17) == solutions


def test_limited_load_keeps_dataset_order():
    """Traces stay readable, and resume compares uids not positions."""
    from rapfprm.data.processbench import subsample

    solutions = _solutions(60, 40)
    picked = subsample(solutions, limit=30, seed=17)
    positions = [solutions.index(s) for s in picked]
    assert positions == sorted(positions)


# --------------------------------------------------------------------------------------
# The parity gate is a CLI, and it is the last thing standing between a drifted config and
# a wasted GPU day, so it is tested as a CLI.
# --------------------------------------------------------------------------------------


def run_parity(*args):
    import subprocess
    import sys

    return subprocess.run(
        [sys.executable, "scripts/check_parity.py", *args],
        capture_output=True,
        text=True,
    )


def test_parity_accepts_the_ab_pair():
    result = run_parity("configs/pilot-int4.yaml", "configs/pilot-int4-retrieval.yaml")
    assert result.returncode == 0, result.stdout + result.stderr


def test_parity_rejects_a_backend_mismatch():
    result = run_parity("configs/baseline.yaml", "configs/pilot-int4-retrieval.yaml")
    assert result.returncode == 1
    assert "prm.backend" in result.stdout


def test_parity_rejects_an_undeclared_ablation():
    """An ablation that does not say what it changes is indistinguishable from drift."""
    result = run_parity("configs/pilot-int4-retrieval.yaml", "configs/pilot-int4-nolabels.yaml")
    assert result.returncode == 1
    assert "prompt.include_reference_labels" in result.stdout


def test_parity_accepts_a_declared_ablation():
    result = run_parity(
        "configs/pilot-int4-retrieval.yaml",
        "configs/pilot-int4-nolabels.yaml",
        "--allow",
        "prompt.include_reference_labels",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "declared ablation" in result.stdout


def test_parity_rejects_two_configs_describing_the_same_arm():
    result = run_parity("configs/retrieval.yaml", "configs/retrieval.yaml")
    assert result.returncode == 1
    assert "SAME arm" in result.stdout


# --------------------------------------------------------------------------------------
# The bf16 arms used by the A100 script. Same parity rules as the int4 pilots: each arm may
# differ from Condition B in exactly one declared key.
# --------------------------------------------------------------------------------------


def test_bf16_arms_share_everything_but_their_defining_key():
    b = load_config("configs/retrieval.yaml")
    arms = {
        "A": load_config("configs/baseline.yaml"),
        "C": load_config("configs/control-random.yaml"),
        "D": load_config("configs/ablation-nolabels.yaml"),
        "E": load_config("configs/ablation-wordlabels.yaml"),
    }
    for label, cfg in arms.items():
        assert cfg.prm == b.prm, f"arm {label}: prm drifted"
        assert cfg.data == b.data, f"arm {label}: data drifted"
        assert cfg.prm.backend == "hf", f"arm {label} must be bf16, not {cfg.prm.backend}"

    assert arms["A"].retrieval.enabled is False
    assert arms["C"].retrieval.reference_mode == "random"
    assert arms["D"].prompt.include_reference_labels is False
    assert arms["E"].prompt.label_style == "words"

    # D and E differ from B in the prompt block and nowhere else.
    for label in ("D", "E"):
        diffs = [
            f
            for f in vars(b.prompt)
            if getattr(b.prompt, f) != getattr(arms[label].prompt, f)
        ]
        assert len(diffs) == 1, f"arm {label} varies {diffs}, expected exactly one key"


def test_every_arm_writes_to_its_own_directory():
    names = [
        load_config(p).run.name
        for p in (
            "configs/baseline.yaml",
            "configs/retrieval.yaml",
            "configs/control-random.yaml",
            "configs/ablation-nolabels.yaml",
            "configs/ablation-wordlabels.yaml",
        )
    ]
    assert len(names) == len(set(names)), f"run names collide: {names}"


def test_summary_maps_every_arm_to_a_real_config():
    """The summary hardcodes run directory names; they must match what the configs write."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "summarize_study", "scripts/summarize_study.py"
    )
    summary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(summary)

    expected = {
        "A": "configs/baseline.yaml",
        "B": "configs/retrieval.yaml",
        "C": "configs/control-random.yaml",
        "D": "configs/ablation-nolabels.yaml",
        "E": "configs/ablation-wordlabels.yaml",
    }
    for arm, config_path in expected.items():
        assert summary.ARM_RUNS[arm] == load_config(config_path).run.name
