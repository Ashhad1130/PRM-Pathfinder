"""Retrieval mechanics, above all the contamination guard."""

import numpy as np
import pytest

from rapfprm.config import RetrievalConfig
from rapfprm.data.pool import PoolItem, parse_row
from rapfprm.retrieval.encoder import HashingEncoder, l2_normalise
from rapfprm.retrieval.index import build_index
from rapfprm.retrieval.retriever import Retriever


def make_item(question, step, qid=None):
    return PoolItem(
        qid=qid or question[:12],
        question=question,
        prev_steps=(),
        step=step,
        math_ok=True,
        consistency_ok=True,
        source="test",
    )


@pytest.fixture
def pool():
    return [
        make_item("How many apples in 3 baskets of 4?", "3 x 4 = 12."),
        make_item("How many pears in 5 crates of 6?", "5 x 6 = 30."),
        make_item("What is the derivative of x squared?", "The derivative is 2x."),
        make_item("Find the integral of 2x.", "The integral is x squared."),
    ]


@pytest.fixture
def retriever(pool):
    cfg = RetrievalConfig(
        encoder_name="hashing",
        pca_components=8,
        top_k_questions=3,
        top_k_steps=2,
        max_question_similarity=0.95,
    )
    encoder = HashingEncoder(dim=64)
    index = build_index(pool, cfg, encoder=encoder)
    return Retriever(items=pool, index=index, cfg=cfg, encoder=encoder)


def test_returns_requested_number_of_references(retriever):
    refs = retriever.retrieve("How many plums in 2 boxes of 7?", "2 x 7 = 14.")
    assert len(refs) == 2
    assert all(0.0 <= r.question_similarity <= 1.0001 for r in refs)


def test_exact_duplicate_question_is_filtered(retriever, pool):
    """The eval question appearing verbatim in the pool would leak the answer key."""
    query = pool[0].question
    refs = retriever.retrieve(query, "3 x 4 = 12.")
    assert all(r.item.question != query for r in refs)
    assert retriever.stats.filtered_exact_duplicate >= 1


def test_similarity_threshold_filters_near_duplicates(pool):
    cfg = RetrievalConfig(
        encoder_name="hashing",
        pca_components=None,
        top_k_questions=4,
        top_k_steps=2,
        max_question_similarity=0.10,  # aggressive: nearly everything is "too similar"
        drop_exact_duplicates=False,
    )
    encoder = HashingEncoder(dim=64)
    index = build_index(pool, cfg, encoder=encoder)
    retriever = Retriever(items=pool, index=index, cfg=cfg, encoder=encoder)

    retriever.retrieve("How many apples in 3 baskets of 4?", "3 x 4 = 12.")
    assert retriever.stats.filtered_similarity > 0


def test_stats_track_every_query(retriever):
    retriever.retrieve("A question", "A step")
    retriever.retrieve("Another question", "Another step")
    assert retriever.stats.as_dict()["queries"] == 2


def test_index_rejects_a_mismatched_pool(retriever, pool):
    with pytest.raises(ValueError, match="Pool/index mismatch"):
        Retriever(items=pool[:2], index=retriever.index, cfg=retriever.cfg)


def test_embeddings_are_unit_norm(retriever):
    norms = np.linalg.norm(retriever.index.question_matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_l2_normalise_survives_a_zero_vector():
    out = l2_normalise(np.zeros((1, 4), dtype=np.float32))
    assert np.isfinite(out).all()


# --- PCA ------------------------------------------------------------------------------


def test_pca_matches_sklearn_when_it_is_importable():
    """Our numpy PCA must be the same computation sklearn does, not an approximation."""
    sklearn_decomposition = pytest.importorskip(
        "sklearn.decomposition",
        reason="sklearn unavailable in this environment",
        exc_type=ImportError,  # a blocked/broken native lib raises ImportError, not ModuleNotFound
    )
    from rapfprm.retrieval.index import _fit_pca

    rng = np.random.default_rng(0)
    data = rng.normal(size=(64, 12))

    mean, components = _fit_pca(data, n_components=5, whiten=False)

    reference = sklearn_decomposition.PCA(n_components=5, random_state=0).fit(data)
    assert np.allclose(mean, reference.mean_, atol=1e-5)
    # Singular-vector signs are arbitrary; compare the subspace via |cosine|.
    for ours, theirs in zip(components, reference.components_):
        assert abs(float(np.dot(ours, theirs))) == pytest.approx(1.0, abs=1e-4)


def test_pca_projection_preserves_relative_distances():
    """A sanity check that survives without sklearn: PCA keeps the dominant structure."""
    from rapfprm.retrieval.index import _fit_pca

    rng = np.random.default_rng(1)
    # Two well-separated clusters in 20-D.
    cluster_a = rng.normal(loc=-4.0, scale=0.2, size=(40, 20))
    cluster_b = rng.normal(loc=+4.0, scale=0.2, size=(40, 20))
    data = np.vstack([cluster_a, cluster_b])

    mean, components = _fit_pca(data, n_components=3, whiten=False)
    projected = (data - mean) @ components.T

    within = np.linalg.norm(projected[:40].mean(0) - projected[0])
    between = np.linalg.norm(projected[:40].mean(0) - projected[40:].mean(0))
    assert between > 5 * within


def test_pca_is_deterministic_across_calls():
    from rapfprm.retrieval.index import _fit_pca

    rng = np.random.default_rng(2)
    data = rng.normal(size=(50, 10))
    first = _fit_pca(data, n_components=4, whiten=False)
    second = _fit_pca(data, n_components=4, whiten=False)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_pca_declines_when_there_is_nothing_to_fit():
    from rapfprm.retrieval.index import _fit_pca

    mean, components = _fit_pca(np.zeros((1, 8)), n_components=4, whiten=False)
    assert mean is None and components is None


# --- PathFinder-600K parsing ----------------------------------------------------------


def test_parse_row_recovers_question_steps_and_labels():
    row = {
        "inputs": [
            {"role": "user", "content": "You are a Math Teacher. ...\n\n Question: What is 2+2?"},
            {
                "role": "assistant",
                "content": (
                    "First we read it.\n\nCurrent Step: We add to get 5."
                    " Math reasoning: <extra>, Consistency: <extra>"
                ),
            },
        ],
        "labels": [
            {"role": "user", "content": "ignored"},
            {
                "role": "assistant",
                "content": (
                    "First we read it.\n\nCurrent Step: We add to get 5."
                    " Math reasoning: <->, Consistency: <+>"
                ),
            },
        ],
        "source": "prm800k",
    }
    item = parse_row(row)
    assert item is not None
    assert item.question == "What is 2+2?"
    assert item.prev_steps == ("First we read it.",)
    assert item.step == "We add to get 5."
    assert item.math_ok is False
    assert item.consistency_ok is True
    assert item.error_type == "math error"


def test_parse_row_handles_an_optimality_row_with_no_previous_steps():
    """Real shape from PathFinder-600K: opens with the marker, ends with Correctness.

    These are ~half the pool. An end-anchored regex or a "\\n\\nCurrent Step: " marker
    silently drops every one of them.
    """
    assistant = (
        "Current Step: The price of the pants is $340 / 2 = $170."
        " Math reasoning: <->, Consistency: <->, Correctness: <extra>"
    )
    row = {
        "inputs": [
            {"role": "user", "content": "prefix\n\n Question: Mark bought a shirt for $340."},
            {"role": "assistant", "content": assistant},
        ],
        "labels": [
            {"role": "user", "content": "prefix\n\n Question: Mark bought a shirt for $340."},
            {
                "role": "assistant",
                "content": assistant.replace("Correctness: <extra>", "Correctness: <->"),
            },
        ],
        "source": "rlhflow_mistral",
    }
    item = parse_row(row)
    assert item is not None
    assert item.prev_steps == ()
    assert item.step == "The price of the pants is $340 / 2 = $170."
    assert (item.math_ok, item.consistency_ok, item.correctness_ok) == (False, False, False)
    assert item.error_type == "math and consistency error"


def test_error_detection_rows_carry_no_correctness_label():
    row = {
        "inputs": [
            {"role": "user", "content": "prefix\n\n Question: Q?"},
            {
                "role": "assistant",
                "content": "Current Step: S. Math reasoning: <extra>, Consistency: <extra>",
            },
        ],
        "labels": [
            {
                "role": "assistant",
                "content": "Current Step: S. Math reasoning: <+>, Consistency: <+>",
            }
        ],
    }
    item = parse_row(row)
    assert item is not None
    assert item.correctness_ok is None
    assert item.error_type == "no error"


def test_parse_row_skips_rows_whose_labels_are_still_masked():
    """Rows whose labels are still masked carry no Math/Consistency gold labels."""
    row = {
        "inputs": [
            {"role": "user", "content": "prefix\n\n Question: Q?"},
            {
                "role": "assistant",
                "content": "Current Step: S. Math reasoning: <extra>, Consistency: <extra>",
            },
        ],
        "labels": [
            {
                "role": "assistant",
                "content": "Current Step: S. Math reasoning: <extra>, Consistency: <extra>",
            }
        ],
    }
    assert parse_row(row) is None


def test_parse_row_rejects_malformed_input():
    assert parse_row({"inputs": [], "labels": []}) is None
    assert parse_row({}) is None
