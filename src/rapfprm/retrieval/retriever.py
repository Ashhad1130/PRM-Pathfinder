"""Two-stage retrieval with a contamination guard.

Stage 1 (question level) attacks *dataset* shift: find pool problems that look like the
one being graded. Stage 2 (step level) attacks *reasoning-style* shift: among those
problems, find the steps that most resemble the step under judgement.

`reference_mode: random` turns the same machinery into the control arm (Condition C): the
guard, the reference count and the rendering are unchanged, only the ranking is replaced by
a uniform draw. Condition B beating C is evidence for retrieval; B matching C means the
gain was prompt length.

The contamination guard is not optional. ProcessBench and PathFinder-600K both descend
from MATH/GSM8K, so a near-identical pool question can hand the grader a labelled copy of
the very step it is being asked to judge. That would inflate Condition B for a reason that
has nothing to do with the mechanism under test. Every filtered neighbour is counted and
reported so the guard's effect is visible in the run summary.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from ..config import RetrievalConfig
from ..data.pool import PoolItem, normalise_question
from .encoder import Encoder, build_encoder
from .index import PoolIndex


@dataclass(frozen=True)
class Reference:
    """A retrieved exemplar, ready to be rendered into the prompt."""

    item: PoolItem
    question_similarity: float
    step_similarity: float


@dataclass
class RetrievalStats:
    """Counters aggregated across a run, written into the run summary."""

    queries: int = 0
    references_returned: int = 0
    filtered_exact_duplicate: int = 0
    filtered_similarity: int = 0
    empty_results: int = 0
    similarity_sum: float = 0.0
    step_similarity_sum: float = 0.0
    #: "retrieved" or "random" — recorded so a summary.json can never be mistaken for the
    #: other arm's.
    mode: str = "retrieved"

    def as_dict(self) -> dict:
        mean_similarity = (
            self.similarity_sum / self.references_returned if self.references_returned else 0.0
        )
        mean_step_similarity = (
            self.step_similarity_sum / self.references_returned if self.references_returned else 0.0
        )
        return {
            "mode": self.mode,
            "queries": self.queries,
            "references_returned": self.references_returned,
            "references_per_query": (
                self.references_returned / self.queries if self.queries else 0.0
            ),
            "filtered_exact_duplicate": self.filtered_exact_duplicate,
            "filtered_similarity": self.filtered_similarity,
            "empty_results": self.empty_results,
            "mean_question_similarity": mean_similarity,
            #: The control arm's evidence that it really is a control: in "random" mode
            #: this should sit near chance, far below the "retrieved" arm's value.
            "mean_step_similarity": mean_step_similarity,
        }


@dataclass
class Retriever:
    """Ranks pool items against a (question, step) query."""

    items: list[PoolItem]
    index: PoolIndex
    cfg: RetrievalConfig
    encoder: Encoder | None = None
    stats: RetrievalStats = field(default_factory=RetrievalStats)

    # Pool item rows grouped by question row, built once.
    _rows_by_question: dict[int, list[int]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.encoder is None:
            self.encoder = build_encoder(self.cfg.encoder_name)
        if len(self.items) != self.index.step_matrix.shape[0]:
            raise ValueError(
                f"Pool/index mismatch: {len(self.items)} items but "
                f"{self.index.step_matrix.shape[0]} indexed steps. Rebuild the index."
            )
        for row, question_row in enumerate(self.index.item_question_row):
            self._rows_by_question.setdefault(int(question_row), []).append(row)

    # ---- internals --------------------------------------------------------------------

    def _embed(self, texts: list[str]) -> np.ndarray:
        raw = self.encoder.encode(texts, batch_size=self.cfg.encoder_batch_size)
        return self.index.project(raw)

    def _query_rng(self, question: str, step: str) -> np.random.Generator:
        """Deterministic per-query RNG for the random-reference control.

        Seeded from the query text, not from a counter, so the references drawn for a
        solution do not depend on how many solutions came before it. Without that, a run
        resumed at solution 900 would draw different references from an uninterrupted one
        and the control arm would not be reproducible.
        """
        digest = hashlib.blake2b(
            f"{self.cfg.random_seed}\x00{question}\x00{step}".encode("utf-8"), digest_size=8
        ).digest()
        return np.random.default_rng(int.from_bytes(digest, "big"))

    def _allowed_question_rows(
        self, similarities: np.ndarray, query_norm: str, order: np.ndarray
    ) -> tuple[np.ndarray, int, int]:
        """Walk question rows in `order`, dropping contaminated ones.

        Both arms share this loop on purpose: the control must be filtered by exactly the
        same guard as the treatment, or the two arms differ in two ways instead of one.
        Returns (rows, n_exact, n_sim).
        """
        exact_dropped = similarity_dropped = 0
        kept: list[int] = []
        for row in order:
            if len(kept) >= self.cfg.top_k_questions:
                break
            if self.cfg.drop_exact_duplicates and self.index.question_norm_text[row] == query_norm:
                exact_dropped += 1
                continue
            if similarities[row] >= self.cfg.max_question_similarity:
                similarity_dropped += 1
                continue
            kept.append(int(row))

        return np.array(kept, dtype=np.int32), exact_dropped, similarity_dropped

    # ---- public API -------------------------------------------------------------------

    def retrieve(self, question: str, step: str) -> list[Reference]:
        """Return up to `top_k_steps` references for one (question, step) query."""
        self.stats.queries += 1
        self.stats.mode = self.cfg.reference_mode
        random_mode = self.cfg.reference_mode == "random"
        rng = self._query_rng(question, step) if random_mode else None

        query = self._embed([question, step])
        question_vector, step_vector = query[0], query[1]

        # The similarities are computed in both arms: the control still needs them for the
        # contamination guard, and reporting them is what proves it drew unrelated text.
        question_similarities = self.index.question_matrix @ question_vector
        order = (
            rng.permutation(self.index.question_matrix.shape[0])
            if random_mode
            else np.argsort(-question_similarities)
        )
        question_rows, exact_dropped, similarity_dropped = self._allowed_question_rows(
            question_similarities, normalise_question(question), order
        )
        self.stats.filtered_exact_duplicate += exact_dropped
        self.stats.filtered_similarity += similarity_dropped

        if question_rows.size == 0:
            self.stats.empty_results += 1
            return []

        candidate_rows = [row for q in question_rows for row in self._rows_by_question.get(int(q), [])]
        if not candidate_rows:
            self.stats.empty_results += 1
            return []

        candidates = np.array(candidate_rows, dtype=np.int32)
        if random_mode:
            # Control arm: same count, same guard, no ranking. Prompt length is therefore
            # matched to the treatment arm while relevance is removed — the only way to
            # tell "retrieval helps" apart from "extra context helps".
            k = min(self.cfg.top_k_steps, candidates.size)
            top = candidates[rng.choice(candidates.size, size=k, replace=False)]
        elif self.cfg.step_level:
            scores = self.index.step_matrix[candidates] @ step_vector
            top = candidates[np.argsort(-scores)[: self.cfg.top_k_steps]]
        else:
            # Stage-2 ablation: rank purely by how similar the *question* was.
            scores = question_similarities[self.index.item_question_row[candidates]]
            top = candidates[np.argsort(-scores)[: self.cfg.top_k_steps]]

        references: list[Reference] = []
        for row in top:
            item = self.items[int(row)]
            question_similarity = float(
                question_similarities[int(self.index.item_question_row[int(row)])]
            )
            step_similarity = float(self.index.step_matrix[int(row)] @ step_vector)
            references.append(
                Reference(
                    item=item,
                    question_similarity=question_similarity,
                    step_similarity=step_similarity,
                )
            )
            self.stats.similarity_sum += question_similarity
            self.stats.step_similarity_sum += step_similarity

        self.stats.references_returned += len(references)
        if not references:
            self.stats.empty_results += 1
        return references


def load_retriever(cfg: RetrievalConfig, pool_dir: str) -> Retriever:
    """Convenience loader used by the evaluation runner."""
    from ..data.pool import load_pool

    items = load_pool(pool_dir)
    index = PoolIndex.load(cfg.index_dir)
    if index.encoder_name != cfg.encoder_name:
        raise ValueError(
            f"Index was built with encoder {index.encoder_name!r} but the config asks for "
            f"{cfg.encoder_name!r}. Rebuild the index or fix the config — mixing encoders "
            "silently destroys retrieval quality."
        )
    return Retriever(items=items, index=index, cfg=cfg)
