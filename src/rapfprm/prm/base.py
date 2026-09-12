"""The scoring contract shared by every PRM backend."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Protocol, Sequence

from ..config import PRMConfig
from ..retrieval.retriever import Reference


@dataclass(frozen=True)
class StepVerdict:
    """PathFinder-PRM's hierarchical judgement for a single step."""

    #: Stage 1 — False means the model emitted <-> for "Math reasoning".
    math_ok: bool
    #: Stage 1 — False means the model emitted <-> for "Consistency".
    consistency_ok: bool
    #: Stage 2 — P(<+>) for "Correctness". None when stage 1 already found an error,
    #: because the model card short-circuits and never runs the second pass.
    correctness_prob: float | None
    #: Number of references injected into the prompt (0 in Condition A).
    n_references: int = 0

    @property
    def stage1_error(self) -> bool:
        return not (self.math_ok and self.consistency_ok)

    def is_error(self, threshold: float) -> bool:
        """Final verdict: erroneous if stage 1 flagged it, or stage 2 fell below threshold."""
        if self.stage1_error:
            return True
        if self.correctness_prob is None:
            return False
        return self.correctness_prob < threshold

    @property
    def error_type(self) -> str:
        if not self.math_ok and not self.consistency_ok:
            return "math and consistency error"
        if not self.math_ok:
            return "math error"
        if not self.consistency_ok:
            return "consistency error"
        return "no error"

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["error_type"] = self.error_type
        return payload


class PRMBackend(Protocol):
    """Scores one step of one solution."""

    name: str

    def score_step(
        self,
        question: str,
        prev_steps: Sequence[str],
        step: str,
        references: list[Reference] | None,
    ) -> StepVerdict: ...


def build_backend(cfg: PRMConfig, prompt_cfg):
    """Factory dispatching on `prm.backend`."""
    if cfg.backend == "mock":
        from .mock import MockPRM

        return MockPRM(cfg, prompt_cfg)
    if cfg.backend in {"hf", "hf4bit", "int4"}:
        from .pathfinder import PathFinderPRM

        return PathFinderPRM(cfg, prompt_cfg, load_in_4bit=cfg.backend == "hf4bit")
    raise ValueError(f"Unknown backend {cfg.backend!r}")
