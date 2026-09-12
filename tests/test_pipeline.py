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
