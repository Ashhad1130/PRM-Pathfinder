"""Regenerate the bundled test fixtures.

    python tests/fixtures/generate.py

The fixtures are small, synthetic and deterministic. They exist so the pipeline can be
exercised without downloading ProcessBench or PathFinder-600K; they carry no signal and
must never be used to produce a number that appears in the report.

Shapes mirror the real sources exactly:
  processbench/<subset>.jsonl  -> id, problem, steps[], label, final_answer_correct, generator
  pool/pool.jsonl              -> the parsed PoolItem form of PathFinder-600K rows
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")

TOPICS = [
    ("apples", "baskets"), ("marbles", "jars"), ("coins", "purses"),
    ("stickers", "albums"), ("pencils", "boxes"), ("cards", "decks"),
    ("flowers", "vases"), ("cookies", "tins"), ("stamps", "books"),
    ("shells", "buckets"),
]


def _qid(question: str) -> str:
    return hashlib.sha1(" ".join(question.split()).strip().lower().encode()).hexdigest()[:16]


def make_problem(rng: random.Random, subset: str, i: int) -> dict:
    thing, container = rng.choice(TOPICS)
    per = rng.randint(3, 12)
    n = rng.randint(2, 9)
    extra = rng.randint(1, 20)

    problem = (
        f"There are {n} {container}, each holding {per} {thing}. "
        f"Then {extra} more {thing} are added. How many {thing} are there in total?"
    )

    total = n * per
    steps = [
        f"First, find how many {thing} are in the {container}: {n} x {per} = {total}.",
        f"Next, add the {extra} extra {thing}: {total} + {extra} = {total + extra}.",
        f"So the total number of {thing} is {total + extra}.",
    ]

    # A third of the fixtures carry an injected arithmetic error at a known index.
    label = -1
    if i % 3 == 0:
        bad_index = rng.randint(0, len(steps) - 1)
        wrong = total + extra + rng.choice([-3, -1, 1, 4])
        steps[bad_index] = steps[bad_index].replace(str(total + extra), str(wrong))
        label = bad_index

    return {
        "id": f"{subset}-{i:03d}",
        "problem": problem,
        "steps": steps,
        "label": label,
        "final_answer_correct": label == -1,
        "generator": "fixture-generator",
    }


def make_pool_item(rng: random.Random, i: int) -> dict:
    thing, container = rng.choice(TOPICS)
    per = rng.randint(3, 12)
    n = rng.randint(2, 9)
    total = n * per

    question = (
        f"A shop has {n} {container} of {thing}, with {per} {thing} in each. "
        f"How many {thing} does the shop have?"
    )
    step = f"Multiply the number of {container} by the {thing} in each: {n} x {per} = {total}."

    # Mix of clean steps and each error type, so the reference block is exercised.
    kind = i % 4
    math_ok = kind != 1
    consistency_ok = kind != 2
    if kind == 1:
        step = step.replace(str(total), str(total + rng.randint(1, 5)))
    if kind == 3:
        math_ok = consistency_ok = False

    return {
        "qid": _qid(question),
        "question": question,
        "prev_steps": [f"We are told there are {n} {container}."],
        "step": step,
        "math_ok": math_ok,
        "consistency_ok": consistency_ok,
        "source": "fixture",
    }


def main() -> None:
    rng = random.Random(20260906)

    pb_dir = HERE / "processbench"
    pb_dir.mkdir(parents=True, exist_ok=True)
    for subset in SUBSETS:
        rows = [make_problem(rng, subset, i) for i in range(12)]
        path = pb_dir / f"{subset}.jsonl"
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
        )
        n_err = sum(r["label"] != -1 for r in rows)
        print(f"{path.name}: {len(rows)} solutions ({n_err} with an error)")

    pool_dir = HERE / "pool"
    pool_dir.mkdir(parents=True, exist_ok=True)
    items = [make_pool_item(rng, i) for i in range(60)]
    (pool_dir / "pool.jsonl").write_text(
        "\n".join(json.dumps(it, ensure_ascii=False) for it in items) + "\n", encoding="utf-8"
    )
    print(f"pool.jsonl: {len(items)} items")


if __name__ == "__main__":
    main()
