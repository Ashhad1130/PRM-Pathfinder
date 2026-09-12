"""End-to-end pipeline check on bundled fixtures. No GPU, no network, ~5 seconds.

    python scripts/smoke.py

Runs every stage for real — pool load, PCA index, two-stage retrieval, prompt assembly,
scoring loop, ProcessBench metric, A-vs-B comparison — with a hashing encoder and a mock
PRM standing in for the heavy parts.

THE NUMBERS ARE MEANINGLESS. This answers "is the plumbing intact", nothing more.
Run it after every change and before every real experiment.
"""

from __future__ import annotations

import argparse
import json

from _bootstrap import setup_logging  # noqa: F401

from rapfprm.analysis.compare import compare, to_markdown
from rapfprm.config import load_config
from rapfprm.data.pool import load_pool
from rapfprm.eval.runner import run_evaluation
from rapfprm.retrieval.index import build_index

CONFIG = "configs/smoke.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)

    print("=" * 72)
    print("SMOKE TEST — mock backend, fixture data. Not a result.")
    print("=" * 72)

    # 1. index the fixture pool
    cfg_b = load_config(CONFIG, {"run.name": "smoke-b"})
    items = load_pool(cfg_b.data.pool_dir)
    index = build_index(items, cfg_b.retrieval)
    index.save(cfg_b.retrieval.index_dir)
    print(f"\n[1/4] indexed {len(items)} pool items -> {cfg_b.retrieval.index_dir}")

    # 2. Condition A
    cfg_a = load_config(CONFIG, {"run.name": "smoke-a", "retrieval.enabled": False})
    print("\n[2/4] Condition A (baseline)")
    summary_a = run_evaluation(cfg_a)

    # 3. Condition B
    print("\n[3/4] Condition B (retrieval)")
    summary_b = run_evaluation(cfg_b)

    # 4. compare
    print("\n[4/4] comparison")
    result = compare(cfg_a.run_dir, cfg_b.run_dir)
    print()
    print(to_markdown(result))

    print("\nRetrieval stats (B):")
    print(json.dumps(summary_b["retrieval"], indent=2))

    assert summary_a["metrics"]["average_f1"] >= 0.0
    assert summary_b["retrieval"]["queries"] > 0, "retriever was never called"
    print("\nSMOKE TEST PASSED — pipeline is intact.")


if __name__ == "__main__":
    main()
