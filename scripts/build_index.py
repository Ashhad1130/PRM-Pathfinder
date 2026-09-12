"""Embed the retrieval pool and fit the PCA projection.

    python scripts/build_index.py --config configs/retrieval.yaml

Writes `<retrieval.index_dir>/index.npz` plus metadata. The index records which encoder
built it; the retriever refuses to load an index built with a different one.
"""

from __future__ import annotations

import argparse
import time

from _bootstrap import setup_logging  # noqa: F401

from rapfprm.config import load_config
from rapfprm.data.pool import load_pool
from rapfprm.retrieval.index import build_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    cfg = load_config(args.config)

    items = load_pool(cfg.data.pool_dir)
    print(f"Embedding {len(items)} pool items with {cfg.retrieval.encoder_name} ...")

    started = time.time()
    index = build_index(items, cfg.retrieval)
    path = index.save(cfg.retrieval.index_dir)

    print(f"Built in {time.time() - started:.1f}s -> {path}")
    print(f"  questions : {index.question_matrix.shape}")
    print(f"  steps     : {index.step_matrix.shape}")
    print(f"  PCA dims  : {index.pca_components}")


if __name__ == "__main__":
    main()
