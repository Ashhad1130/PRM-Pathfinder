"""Deterministic stand-in PRM.

Exists so the full pipeline — pool, index, retrieval, prompt assembly, scoring loop,
metrics, comparison — can be exercised in seconds with no GPU and no downloads. It builds
the same prompts as the real backend (so prompt bugs still surface) and then ignores them,
deriving a reproducible pseudo-verdict from a hash.

**Numbers produced by this backend are meaningless.** The runner stamps `backend: mock`
into every run summary so a mock run can never be mistaken for a real result.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

from ..config import PRMConfig, PromptConfig
from ..retrieval.retriever import Reference
from .base import StepVerdict
from .prompts import build_messages


def _unit_hash(*parts: str) -> float:
    """Stable pseudo-random float in [0, 1) derived from the inputs."""
    digest = hashlib.md5("\x1f".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") / float(1 << 64)


class MockPRM:
    """Hash-driven verdicts with the real backend's interface."""

    #: How strongly retrieved references perturb the verdict. 0.0 makes Condition B
    #: bit-identical to Condition A, which is the invariant the smoke test asserts.
    sensitivity: float = 0.35

    def __init__(self, cfg: PRMConfig, prompt_cfg: PromptConfig) -> None:
        self.cfg = cfg
        self.prompt_cfg = prompt_cfg
        self.name = "mock"
        self.dropped_references = 0

    def score_step(
        self,
        question: str,
        prev_steps: Sequence[str],
        step: str,
        references: list[Reference] | None = None,
    ) -> StepVerdict:
        references = references or []

        # Build the prompts for real: prompt-assembly bugs must fail here too.
        messages = build_messages(question, tuple(prev_steps), step, references, self.prompt_cfg)
        n_references = messages[0]["content"].count("[Reference ")

        salt = f"{self.cfg.seed}"
        base = _unit_hash(salt, question, step)

        # References nudge the verdict so downstream comparison code sees two different runs.
        nudge = self.sensitivity * n_references * (_unit_hash(salt, "ref", question, step) - 0.5)
        score = min(max(base + nudge, 0.0), 1.0)

        if score < 0.12:
            return StepVerdict(False, True, None, n_references)
        if score < 0.22:
            return StepVerdict(True, False, None, n_references)
        return StepVerdict(True, True, correctness_prob=score, n_references=n_references)
