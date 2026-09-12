"""Embedding index: SBERT -> PCA -> cosine similarity.

This is RetrievalPRM's retrieval recipe. PCA is not just compression: reducing the
embedding to a few hundred components denoises the similarity estimate, which matters when
the pool is large and the neighbourhoods are dense.

The index is a dense matrix, not FAISS. At seminar scale (tens of thousands of items,
a few thousand queries) a single normalised mat-mul is faster than building an ANN
structure, and it is exact — no recall/latency trade-off to defend in the report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import RetrievalConfig
from ..data.pool import PoolItem, normalise_question
from .encoder import Encoder, build_encoder, l2_normalise


@dataclass
class PoolIndex:
    """Question-level and step-level embedding matrices over the retrieval pool."""

    #: (n_questions, d) unit-norm question embeddings.
    question_matrix: np.ndarray
    #: Ordered unique question ids, aligned with `question_matrix` rows.
    question_ids: list[str]
    #: (n_items, d) unit-norm step embeddings.
    step_matrix: np.ndarray
    #: Pool item index -> row in `step_matrix` is the identity; this maps item -> question row.
    item_question_row: np.ndarray
    #: Normalised question text per question row, for exact-duplicate detection.
    question_norm_text: list[str]

    encoder_name: str = ""
    pca_components: int | None = None

    # Fitted transform, needed to project queries into the same space.
    _pca_mean: np.ndarray | None = None
    _pca_components_matrix: np.ndarray | None = None

    # ---- query projection -------------------------------------------------------------

    def project(self, raw: np.ndarray) -> np.ndarray:
        """Apply the fitted PCA (if any) and L2-normalise, matching the stored matrices."""
        if self._pca_components_matrix is not None and self._pca_mean is not None:
            raw = (raw - self._pca_mean) @ self._pca_components_matrix.T
        return l2_normalise(np.asarray(raw, dtype=np.float32))

    # ---- persistence ------------------------------------------------------------------

    def save(self, index_dir: str | Path) -> Path:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        arrays = {
            "question_matrix": self.question_matrix,
            "step_matrix": self.step_matrix,
            "item_question_row": self.item_question_row,
        }
        if self._pca_mean is not None:
            arrays["pca_mean"] = self._pca_mean
        if self._pca_components_matrix is not None:
            arrays["pca_components_matrix"] = self._pca_components_matrix
        np.savez_compressed(index_dir / "index.npz", **arrays)

        (index_dir / "index.meta.json").write_text(
            json.dumps(
                {
                    "question_ids": self.question_ids,
                    "question_norm_text": self.question_norm_text,
                    "encoder_name": self.encoder_name,
                    "pca_components": self.pca_components,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return index_dir / "index.npz"

    @classmethod
    def load(cls, index_dir: str | Path) -> "PoolIndex":
        index_dir = Path(index_dir)
        npz_path = index_dir / "index.npz"
        if not npz_path.exists():
            raise FileNotFoundError(
                f"No index at {npz_path}. Run: python scripts/build_index.py --config <cfg>"
            )
        arrays = np.load(npz_path)
        meta = json.loads((index_dir / "index.meta.json").read_text(encoding="utf-8"))
        return cls(
            question_matrix=arrays["question_matrix"],
            question_ids=meta["question_ids"],
            step_matrix=arrays["step_matrix"],
            item_question_row=arrays["item_question_row"],
            question_norm_text=meta["question_norm_text"],
            encoder_name=meta["encoder_name"],
            pca_components=meta["pca_components"],
            _pca_mean=arrays["pca_mean"] if "pca_mean" in arrays else None,
            _pca_components_matrix=(
                arrays["pca_components_matrix"] if "pca_components_matrix" in arrays else None
            ),
        )


def _fit_pca(matrix: np.ndarray, n_components: int, whiten: bool):
    """Fit PCA by SVD and return (mean, components), or (None, None) if data is too small.

    Implemented directly in numpy rather than via ``sklearn.decomposition.PCA``. It is the
    same computation — centre, take the right singular vectors — and it drops a heavy
    dependency whose import chain has proven fragile (scikit-learn does an unguarded
    ``import pyarrow`` in ``utils/fixes.py``, so a broken pyarrow takes sklearn down with
    it, and with it the whole retrieval index).

    Signs of singular vectors are arbitrary, so they are fixed deterministically: the
    largest-magnitude entry of each component is forced positive. Without that the index
    would not be reproducible across BLAS builds.
    """
    n_components = min(n_components, matrix.shape[0], matrix.shape[1])
    if n_components < 2:
        return None, None

    matrix = np.asarray(matrix, dtype=np.float64)
    mean = matrix.mean(axis=0)
    centred = matrix - mean

    # full_matrices=False gives Vt with shape (min(n, d), d) — all we need.
    _, singular_values, vt = np.linalg.svd(centred, full_matrices=False)
    components = vt[:n_components]

    # Deterministic sign convention.
    dominant = np.argmax(np.abs(components), axis=1)
    signs = np.sign(components[np.arange(components.shape[0]), dominant])
    signs[signs == 0] = 1.0
    components = components * signs[:, np.newaxis]

    if whiten:
        n_samples = matrix.shape[0]
        explained_std = singular_values[:n_components] / np.sqrt(max(n_samples - 1, 1))
        components = components / np.maximum(explained_std, 1e-12)[:, np.newaxis]

    return mean.astype(np.float32), components.astype(np.float32)


def build_index(
    items: list[PoolItem],
    cfg: RetrievalConfig,
    encoder: Encoder | None = None,
) -> PoolIndex:
    """Embed the pool at both granularities and fit the shared PCA projection."""
    encoder = encoder or build_encoder(cfg.encoder_name)

    # Unique questions, order-stable.
    question_ids: list[str] = []
    question_texts: list[str] = []
    row_of_qid: dict[str, int] = {}
    for item in items:
        if item.qid not in row_of_qid:
            row_of_qid[item.qid] = len(question_ids)
            question_ids.append(item.qid)
            question_texts.append(item.question)

    item_question_row = np.array([row_of_qid[item.qid] for item in items], dtype=np.int32)

    raw_questions = encoder.encode(question_texts, batch_size=cfg.encoder_batch_size)
    raw_steps = encoder.encode([item.step for item in items], batch_size=cfg.encoder_batch_size)

    pca_mean = pca_components = None
    if cfg.pca_components:
        # Fit on both granularities jointly so questions and steps share one space.
        pca_mean, pca_components = _fit_pca(
            np.vstack([raw_questions, raw_steps]), cfg.pca_components, cfg.pca_whiten
        )

    def transform(matrix: np.ndarray) -> np.ndarray:
        if pca_components is not None and pca_mean is not None:
            matrix = (matrix - pca_mean) @ pca_components.T
        return l2_normalise(np.asarray(matrix, dtype=np.float32))

    return PoolIndex(
        question_matrix=transform(raw_questions),
        question_ids=question_ids,
        step_matrix=transform(raw_steps),
        item_question_row=item_question_row,
        question_norm_text=[normalise_question(t) for t in question_texts],
        encoder_name=encoder.name,
        pca_components=cfg.pca_components,
        _pca_mean=pca_mean,
        _pca_components_matrix=pca_components,
    )
