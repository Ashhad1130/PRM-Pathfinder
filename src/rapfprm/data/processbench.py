"""ProcessBench loading, normalised to one flat record type.

ProcessBench (`Qwen/ProcessBench`) ships four splits — gsm8k, math, olympiadbench,
omnimath — ordered here by out-of-distribution severity. Each row carries a solution
broken into steps and `label`, the index of the FIRST erroneous step (-1 when every
step is correct). That single convention drives the whole evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

from ..config import DataConfig

# Fields we require to be present after loading; a schema drift upstream should fail
# loudly here rather than silently produce nonsense F1 numbers.
REQUIRED_FIELDS = ("problem", "steps", "label")


@dataclass(frozen=True)
class Solution:
    """One ProcessBench item: a problem plus a step-by-step candidate solution."""

    uid: str
    subset: str
    problem: str
    steps: tuple[str, ...]
    #: Index of the first erroneous step; -1 means all steps are correct.
    label: int
    final_answer_correct: bool | None = None
    generator: str | None = None

    @property
    def has_error(self) -> bool:
        return self.label != -1

    def __post_init__(self) -> None:
        if self.label < -1:
            raise ValueError(f"{self.uid}: label must be >= -1, got {self.label}")
        if self.label >= len(self.steps):
            raise ValueError(
                f"{self.uid}: label {self.label} is out of range for {len(self.steps)} steps"
            )
        if not self.steps:
            raise ValueError(f"{self.uid}: solution has no steps")

    def to_dict(self) -> dict:
        return asdict(self)


def _normalise(row: dict, subset: str, position: int) -> Solution:
    missing = [f for f in REQUIRED_FIELDS if f not in row]
    if missing:
        raise KeyError(
            f"ProcessBench row {position} of subset {subset!r} is missing {missing}. "
            f"Available fields: {sorted(row)}. The upstream schema may have changed."
        )
    return Solution(
        uid=str(row.get("id") or f"{subset}-{position}"),
        subset=subset,
        problem=row["problem"],
        steps=tuple(row["steps"]),
        label=int(row["label"]),
        final_answer_correct=row.get("final_answer_correct"),
        generator=row.get("generator"),
    )


def _load_fixture_subset(fixtures_dir: Path, subset: str) -> list[dict]:
    path = fixtures_dir / f"{subset}.jsonl"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_hub_subset(repo_id: str, subset: str) -> list[dict]:
    """Fetch one subset from the Hub.

    ProcessBench publishes each split as a plain ``<subset>.json`` file, so the default
    path just downloads and parses it — no ``datasets``, and therefore no pyarrow. That
    matters: pyarrow ships a native library that a locked-down machine may refuse to load
    (Windows Application Control, for one), and there is no reason for reading a JSON list
    to depend on it.

    Falls back to ``datasets.load_dataset`` for any repo that is not laid out this way.
    """
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id, f"{subset}.json", repo_type="dataset")
        with open(path, encoding="utf-8") as handle:
            rows = json.load(handle)
        if isinstance(rows, list):
            return rows
        raise ValueError(f"{subset}.json holds {type(rows).__name__}, expected a list")
    except Exception as direct_error:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                f"Could not read {repo_id}/{subset}.json directly ({direct_error}), and "
                "`datasets` is unavailable as a fallback. Install datasets, or point "
                "data.fixtures_dir at local JSONL files."
            ) from exc
        return list(load_dataset(repo_id, split=subset))


def load_processbench(cfg: DataConfig) -> list[Solution]:
    """Load the configured subsets, from local fixtures if set, else the Hub."""
    solutions: list[Solution] = []

    for subset in cfg.subsets:
        if cfg.fixtures_dir:
            rows: list[dict] = _load_fixture_subset(Path(cfg.fixtures_dir), subset)
        else:
            rows = _load_hub_subset(cfg.processbench_id, subset)

        if cfg.limit_per_subset is not None:
            rows = rows[: cfg.limit_per_subset]

        solutions.extend(_normalise(row, subset, i) for i, row in enumerate(rows))

    if not solutions:
        raise RuntimeError(
            "Loaded zero ProcessBench examples. Check data.subsets, data.fixtures_dir "
            "and data.limit_per_subset in your config."
        )
    return solutions


def group_by_subset(solutions: list[Solution]) -> dict[str, list[Solution]]:
    grouped: dict[str, list[Solution]] = {}
    for solution in solutions:
        grouped.setdefault(solution.subset, []).append(solution)
    return grouped


def iter_prefixes(solution: Solution) -> Iterator[tuple[int, tuple[str, ...], str]]:
    """Yield (index, previous_steps, current_step) for every step in order.

    PathFinder-PRM scores one step at a time, conditioned on all preceding steps.
    """
    for index, step in enumerate(solution.steps):
        yield index, solution.steps[:index], step
