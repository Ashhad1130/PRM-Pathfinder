"""Refuse to start a run whose two conditions differ in more than the arm-defining key.

    python scripts/lightning/_check_parity.py configs/baseline.yaml configs/retrieval.yaml
    python scripts/lightning/_check_parity.py configs/retrieval.yaml configs/control-random.yaml
    python scripts/lightning/_check_parity.py A.yaml B.yaml --backend int4 --limit 400

The comparison only means "retrieval helps" if retrieval is the only thing that changed
(docs/EXPERIMENTS.md). A drifted `prm.max_input_tokens`, a different `limit_per_subset` or
a `top_k_steps` that moved in one arm and not the other would make the delta a config
artefact — and you would not find out until after the GPU hours were spent. Exit code 1
stops the driver before that happens.

Exactly two retrieval keys may differ, because each one defines an arm:

    retrieval.enabled         A (off) vs B/C (on)
    retrieval.reference_mode  B ("retrieved") vs C ("random")

Everything else in `retrieval`, and all of `prm`, `prompt` and the shared `data` keys, must
match. Configs are compared *resolved*, i.e. after defaults are filled in and after the
same command-line overrides the driver will pass to both runs, so a key spelled out in one
file and left to its default in the other is not a difference.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rapfprm.config import load_config  # noqa: E402

#: Blocks that must be identical across the two conditions.
LOCKED_SECTIONS = ("prm", "prompt")
#: Keys inside `data` that must match; the rest (pool_*) only affect the retrieval arms.
LOCKED_DATA_KEYS = (
    "subsets",
    "limit_per_subset",
    "processbench_id",
    "fixtures_dir",
    # Different seeds draw different solutions, so the arms would not be paired at all.
    "sample_seed",
)
#: The only keys allowed to differ, one per arm boundary.
ARM_KEYS = ("enabled", "reference_mode")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_a")
    parser.add_argument("config_b")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--subsets", nargs="+", default=None)
    parser.add_argument(
        "--allow",
        action="append",
        default=[],
        metavar="SECTION.KEY",
        help=(
            "declare one extra key this pair is allowed to vary, e.g. "
            "--allow prompt.include_reference_labels for the unlabelled-reference ablation. "
            "Repeatable. An ablation has to name what it changes; anything it does not name "
            "is still refused."
        ),
    )
    args = parser.parse_args()

    declared = set(args.allow)

    overrides: dict = {}
    if args.backend:
        overrides["prm.backend"] = args.backend
    if args.limit is not None:
        overrides["data.limit_per_subset"] = args.limit
    if args.subsets:
        overrides["data.subsets"] = args.subsets

    a = load_config(args.config_a, overrides or None)
    b = load_config(args.config_b, overrides or None)

    problems: list[str] = []

    ablated: list[str] = []

    for section in LOCKED_SECTIONS:
        sa, sb = asdict(getattr(a, section)), asdict(getattr(b, section))
        for key in sorted(set(sa) | set(sb)):
            if sa.get(key) == sb.get(key):
                continue
            if f"{section}.{key}" in declared:
                ablated.append(f"{section}.{key}: {sa.get(key)!r} -> {sb.get(key)!r}")
                continue
            problems.append(f"{section}.{key}: A={sa.get(key)!r}  B={sb.get(key)!r}")

    da, db = asdict(a.data), asdict(b.data)
    for key in LOCKED_DATA_KEYS:
        if da.get(key) != db.get(key):
            problems.append(f"data.{key}: A={da.get(key)!r}  B={db.get(key)!r}")

    # The retrieval block: everything except the two arm keys has to match. `random_seed`
    # is exempt — it only picks which random references the control draws, and it is
    # meaningless in an arm that does not draw any.
    ra, rb = asdict(a.retrieval), asdict(b.retrieval)
    for key in sorted(set(ra) | set(rb)):
        if key in ARM_KEYS or key == "random_seed":
            continue
        if ra.get(key) != rb.get(key):
            problems.append(f"retrieval.{key}: A={ra.get(key)!r}  B={rb.get(key)!r}")

    arm_differences = [key for key in ARM_KEYS if ra.get(key) != rb.get(key)]
    if not arm_differences and not ablated:
        problems.append(
            "the two configs describe the SAME arm (same retrieval.enabled and "
            "reference_mode) - running both would measure nothing"
        )

    if problems:
        print("  A/B parity check FAILED:")
        for line in problems:
            print(f"    {line}")
        print(
            "\n  The two conditions may differ in retrieval.enabled and/or "
            "retrieval.reference_mode,\n  and in nothing else. Fix the configs, or set the "
            "shared value from the command line\n  (--backend / --limit / --subsets apply "
            "to both conditions)."
        )
        sys.exit(1)

    if arm_differences:
        arms = " and ".join(f"retrieval.{key}" for key in arm_differences)
        print(f"  A/B parity OK - prm/prompt/data/retrieval match; the arms differ in {arms}")
    else:
        print("  A/B parity OK - the pair differs only in the declared ablation below")
    for line in ablated:
        print(f"    declared ablation: {line}")


if __name__ == "__main__":
    main()
