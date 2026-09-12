"""Measure how much of ProcessBench appears verbatim in the retrieval pool.

    python scripts/check_contamination.py --config configs/retrieval.yaml

Both datasets descend from MATH and GSM8K, so overlap is expected — but it has to be
measured, not assumed, because a retrieved neighbour that IS the eval question hands the
grader a labelled copy of the step it is being asked to judge. Any gain bought that way is
an artefact.

Reports, per subset:
  verbatim   normalised exact question match in the pool
  near       cosine similarity above retrieval.max_question_similarity (needs the index)

Run this before quoting any Condition B number, and put the table in the report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _bootstrap import setup_logging  # noqa: F401

from rapfprm.config import OOD_ORDER, load_config
from rapfprm.data.pool import load_pool, normalise_question
from rapfprm.data.processbench import group_by_subset, load_processbench
from rapfprm.retrieval.encoder import build_encoder
from rapfprm.retrieval.index import PoolIndex


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="runs/contamination.json")
    parser.add_argument("--skip-near", action="store_true", help="verbatim check only")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)

    solutions = load_processbench(cfg.data)
    pool = load_pool(cfg.data.pool_dir)
    pool_questions = {normalise_question(item.question) for item in pool}
    print(f"pool: {len(pool)} items, {len(pool_questions)} unique questions")

    near_by_subset: dict[str, int] = {}
    if not args.skip_near:
        index = PoolIndex.load(cfg.retrieval.index_dir)
        encoder = build_encoder(cfg.retrieval.encoder_name)
        for subset, group in group_by_subset(solutions).items():
            raw = encoder.encode(
                [s.problem for s in group], batch_size=cfg.retrieval.encoder_batch_size
            )
            projected = index.project(raw)
            best = (projected @ index.question_matrix.T).max(axis=1)
            near_by_subset[subset] = int((best >= cfg.retrieval.max_question_similarity).sum())

    rows = []
    for subset, group in group_by_subset(solutions).items():
        verbatim = sum(1 for s in group if normalise_question(s.problem) in pool_questions)
        rows.append(
            {
                "subset": subset,
                "n": len(group),
                "verbatim": verbatim,
                "verbatim_pct": round(100 * verbatim / len(group), 1),
                "near_duplicate": near_by_subset.get(subset),
            }
        )
    rows.sort(key=lambda r: OOD_ORDER.get(r["subset"], 99))

    header = f"{'subset':<16}{'n':>6}{'verbatim':>10}{'%':>8}{'near-dup':>10}"
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        near = "-" if row["near_duplicate"] is None else row["near_duplicate"]
        print(
            f"{row['subset']:<16}{row['n']:>6}{row['verbatim']:>10}"
            f"{row['verbatim_pct']:>7.1f}%{near:>10}"
        )

    total_n = sum(r["n"] for r in rows)
    total_verbatim = sum(r["verbatim"] for r in rows)
    print("-" * len(header))
    print(
        f"{'TOTAL':<16}{total_n:>6}{total_verbatim:>10}"
        f"{100 * total_verbatim / total_n:>7.1f}%"
    )

    result = {
        "pool_items": len(pool),
        "pool_unique_questions": len(pool_questions),
        "max_question_similarity": cfg.retrieval.max_question_similarity,
        "per_subset": rows,
        "total": {
            "n": total_n,
            "verbatim": total_verbatim,
            "verbatim_pct": round(100 * total_verbatim / total_n, 1),
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nWritten to {out_path}")

    if total_verbatim:
        print(
            "\nThe guard in retrieval/retriever.py drops these. Without it, Condition B "
            "would be reading labelled copies of the questions it is grading."
        )


if __name__ == "__main__":
    main()
