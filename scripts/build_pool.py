"""Download PathFinder-600K and parse it into a flat retrieval pool.

    python scripts/build_pool.py --config configs/retrieval.yaml
"""

from __future__ import annotations

import argparse
import json

from _bootstrap import setup_logging  # noqa: F401  (also fixes sys.path and cwd)

from rapfprm.config import load_config
from rapfprm.data.pool import build_pool, pool_stats, save_pool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-items", type=int, default=None, help="override data.pool_max_items")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    overrides = {"data.pool_max_items": args.max_items} if args.max_items is not None else None
    cfg = load_config(args.config, overrides)

    items = build_pool(cfg.data)
    path = save_pool(items, cfg.data.pool_dir)

    stats = pool_stats(items)
    (path.parent / "pool.stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Wrote {stats['n_items']} pool items -> {path}")
    print(f"  unique questions : {stats['n_unique_questions']}")
    print(f"  error types      : {stats['error_types']}")
    print(f"  sources          : {stats['sources']}")


if __name__ == "__main__":
    main()
