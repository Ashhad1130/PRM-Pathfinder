"""Sentence encoders.

`SbertEncoder` is the real one — Sentence-BERT, exactly RetrievalPRM's choice.
`HashingEncoder` is a dependency-free stand-in with the same interface so the pipeline,
the unit tests and the smoke run work on a machine without `sentence-transformers`.
It is deterministic but semantically meaningless: never report numbers produced with it.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol, Sequence

import numpy as np


class Encoder(Protocol):
    """Anything that turns text into a float32 matrix of shape (n, dim)."""

    name: str
    dim: int

    def encode(self, texts: Sequence[str], batch_size: int = 128) -> np.ndarray: ...


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Guard against zero vectors (empty strings) producing NaNs.
    np.maximum(norms, 1e-12, out=norms)
    return matrix / norms


class SbertEncoder:
    """Sentence-BERT wrapper (the reference implementation)."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - environment problem
            # Distinguish "not installed" from "installed but its import chain is broken".
            # sentence-transformers pulls in `datasets` -> pyarrow, whose native library a
            # locked-down machine may refuse to load; saying "not installed" would send
            # the reader off to reinstall something that is already there.
            missing = isinstance(exc, ModuleNotFoundError) and exc.name in {
                "sentence_transformers",
                None,
            }
            hint = (
                "sentence-transformers is not installed; `pip install sentence-transformers`."
                if missing
                else f"sentence-transformers is installed but failed to import ({exc}). "
                "This is usually a broken native dependency further down the chain "
                "(commonly pyarrow, via `datasets`)."
            )
            raise RuntimeError(
                f"{hint} Alternatives: set retrieval.encoder_name to 'hashing' (test-only, "
                "results are meaningless), or use the transformers-only fallback, which "
                "build_encoder() selects automatically."
            ) from exc

        self._model = SentenceTransformer(model_name, device=device)
        self.name = model_name
        # Renamed in sentence-transformers 6; keep working on older installs too.
        get_dim = getattr(
            self._model, "get_embedding_dimension", None
        ) or self._model.get_sentence_embedding_dimension
        self.dim = int(get_dim())

    def encode(self, texts: Sequence[str], batch_size: int = 128) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 2048,
            normalize_embeddings=False,
        )
        return np.asarray(vectors, dtype=np.float32)


class HashingEncoder:
    """Deterministic bag-of-token hashing. Test scaffolding only."""

    def __init__(self, dim: int = 256, seed: int = 0) -> None:
        self.name = "hashing"
        self.dim = dim
        self._seed = seed

    def _bucket(self, token: str) -> int:
        digest = hashlib.md5(f"{self._seed}:{token}".encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "little") % self.dim

    def encode(self, texts: Sequence[str], batch_size: int = 128) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                matrix[row, self._bucket(token)] += 1.0
        return matrix


class MeanPoolingEncoder:
    """Sentence embeddings using plain `transformers`: mean pooling + L2 normalisation.

    This is exactly what Sentence-BERT does for the mean-pooling family that includes
    `all-MiniLM-L6-v2` (Transformer -> Pooling(mean) -> Normalize), so on those models it
    reproduces `SbertEncoder` numerically. It exists because `sentence_transformers`
    imports `datasets`, which imports pyarrow, which a locked-down machine may refuse to
    load — none of which a mean pool over hidden states actually needs.

    It does NOT reproduce models whose pipeline has extra modules (a trained dense layer,
    CLS pooling, ...). `build_encoder` warns when it falls back for that reason.
    """

    def __init__(self, model_name: str, device: str | None = None) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).eval()

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self.dim = int(self._model.config.hidden_size)

    def encode(self, texts: Sequence[str], batch_size: int = 128) -> np.ndarray:
        torch = self._torch
        outputs: list[np.ndarray] = []

        for start in range(0, len(texts), batch_size):
            batch = [t if t else " " for t in texts[start : start + batch_size]]
            encoded = self._tokenizer(
                batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(self._device)

            with torch.no_grad():
                hidden = self._model(**encoded).last_hidden_state

            # Mean over real tokens only — padding must not drag the vector toward zero.
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            outputs.append((summed / counts).float().cpu().numpy())

        return np.vstack(outputs).astype(np.float32)


def build_encoder(name: str, device: str | None = None) -> Encoder:
    """Factory. The literal name 'hashing' selects the offline stand-in.

    Prefers sentence-transformers (correct for every model type) and falls back to the
    transformers-only mean-pooling encoder when that import chain is broken.
    """
    import logging

    if name == "hashing":
        return HashingEncoder()

    try:
        return SbertEncoder(name, device=device)
    except RuntimeError as exc:
        logging.getLogger(__name__).warning(
            "Falling back to the transformers-only mean-pooling encoder. %s\n"
            "This reproduces Sentence-BERT exactly for mean-pooling models such as "
            "all-MiniLM-L6-v2. If you switch retrieval.encoder_name to a model with a "
            "trained dense head or CLS pooling, fix the sentence-transformers install "
            "instead — the fallback would silently embed it differently.",
            exc,
        )
        return MeanPoolingEncoder(name, device=device)
