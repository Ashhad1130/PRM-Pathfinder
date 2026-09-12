"""Retrieval pool built from `declare-lab/PathFinder-600K`.

Schema of the upstream dataset (verified 2026-09-06 against the dataset card):

    inputs : list[{role, content}]   # user turn = PROMPT_PREFIX + "\\n\\n Question: " + q
                                     # assistant turn = prev steps + "\\n\\nCurrent Step: "
                                     #   + step + " Math reasoning: <extra>, Consistency: <extra>"
    labels : list[{role, content}]   # identical, with <extra> replaced by <+> / <->
    source : str                     # "prm800k" | "rlhflow_mistral"

We parse that back into flat, retrievable units: one record per *step*, carrying its
question, its preceding context and its gold Math / Consistency labels. Those gold labels
are what make a retrieved neighbour useful as a reference — the model sees not just a
similar step but how that step was judged.

The upstream viewer is known to fail (train is arrow, test is json); that affects the web
preview only, not `load_dataset`. If loading does fail, `build_pool` says exactly what to try.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from ..config import DataConfig

POS = "<+>"
NEG = "<->"

# The assistant turn separates prior context from the step under judgement with this
# marker. On the first step of a solution there is no prior context, so the turn opens
# with the marker and has no leading newlines — hence a `find`, not a `\n\n`-anchored split.
CURRENT_STEP_MARKER = "Current Step: "

# Trailing judgement scaffold. PathFinder-600K mixes two row shapes:
#   error detection : "... Math reasoning: <extra>, Consistency: <extra>"
#   step optimality : "... Math reasoning: <->, Consistency: <->, Correctness: <extra>"
# The optimality rows hand the model the error labels as *input* and mask only
# Correctness. Either way the label turn carries every field filled in, so an optional
# Correctness group is all it takes to read both. Rejecting the optimality shape would
# throw away well over half the pool.
JUDGEMENT_RE = re.compile(
    r"\s*Math reasoning:\s*(?P<math><\+>|<->|<extra>)\s*,\s*"
    r"Consistency:\s*(?P<consistency><\+>|<->|<extra>)"
    r"(?:\s*,\s*Correctness:\s*(?P<correctness><\+>|<->|<extra>))?\s*$"
)
# The user turn is "<prefix>\n\n Question: <question>"; the prefix is fixed boilerplate.
QUESTION_RE = re.compile(r"Question:\s*(?P<question>.*)\Z", re.DOTALL)


@dataclass(frozen=True)
class PoolItem:
    """One retrievable step-level exemplar."""

    qid: str
    question: str
    prev_steps: tuple[str, ...]
    step: str
    #: True when the gold label is <+> (no math error). False when <->.
    math_ok: bool
    #: True when the gold label is <+> (no consistency error). False when <->.
    consistency_ok: bool
    #: Stage-2 gold label, present only on optimality-stage rows. None when the row is an
    #: error-detection example, which never carries a Correctness label.
    correctness_ok: bool | None = None
    source: str | None = None

    @property
    def error_type(self) -> str:
        """Human-readable error type, used when rendering the reference block."""
        if not self.math_ok and not self.consistency_ok:
            return "math and consistency error"
        if not self.math_ok:
            return "math error"
        if not self.consistency_ok:
            return "consistency error"
        return "no error"

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["prev_steps"] = list(self.prev_steps)
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "PoolItem":
        payload = dict(payload)
        payload["prev_steps"] = tuple(payload.get("prev_steps") or ())
        return cls(**payload)


def _qid(question: str) -> str:
    return hashlib.sha1(normalise_question(question).encode("utf-8")).hexdigest()[:16]


def normalise_question(text: str) -> str:
    """Whitespace/case-normalised form, used for hashing and exact-duplicate detection."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _extract_question(user_content: str) -> str | None:
    match = QUESTION_RE.search(user_content)
    return match.group("question").strip() if match else None


def _split_assistant_turn(content: str) -> tuple[tuple[str, ...], str] | None:
    """Return (previous_steps, current_step_without_judgement_scaffold)."""
    marker_at = content.find(CURRENT_STEP_MARKER)
    if marker_at == -1:
        return None
    prefix = content[:marker_at]
    current = content[marker_at + len(CURRENT_STEP_MARKER) :]
    prev_steps = tuple(s.strip() for s in prefix.split("\n\n") if s.strip())
    current = JUDGEMENT_RE.sub("", current).strip()
    return prev_steps, current


def _read_labels(label_content: str) -> tuple[bool, bool, bool | None] | None:
    """Read the gold judgement. Returns (math_ok, consistency_ok, correctness_ok)."""
    match = JUDGEMENT_RE.search(label_content)
    if not match:
        return None

    math_token, consistency_token = match.group("math"), match.group("consistency")
    if math_token not in (POS, NEG) or consistency_token not in (POS, NEG):
        # Still masked: this row carries no usable Math/Consistency gold labels.
        return None

    correctness_token = match.group("correctness")
    correctness_ok = (
        correctness_token == POS if correctness_token in (POS, NEG) else None
    )
    return math_token == POS, consistency_token == POS, correctness_ok


def parse_row(row: dict) -> PoolItem | None:
    """Convert one PathFinder-600K row into a PoolItem, or None if it is not usable."""
    inputs, labels = row.get("inputs"), row.get("labels")
    if not inputs or not labels:
        return None

    user_turns = [t for t in inputs if t.get("role") == "user"]
    assistant_turns = [t for t in inputs if t.get("role") == "assistant"]
    label_turns = [t for t in labels if t.get("role") == "assistant"]
    if not user_turns or not assistant_turns or not label_turns:
        return None

    question = _extract_question(user_turns[0].get("content", ""))
    if not question:
        return None

    split = _split_assistant_turn(assistant_turns[-1].get("content", ""))
    if split is None:
        return None
    prev_steps, step = split
    if not step:
        return None

    judged = _read_labels(label_turns[-1].get("content", ""))
    if judged is None:
        return None
    math_ok, consistency_ok, correctness_ok = judged

    return PoolItem(
        qid=_qid(question),
        question=question,
        prev_steps=prev_steps,
        step=step,
        math_ok=math_ok,
        consistency_ok=consistency_ok,
        correctness_ok=correctness_ok,
        source=row.get("source"),
    )


def build_pool(cfg: DataConfig) -> list[PoolItem]:
    """Download PathFinder-600K and parse it into pool items."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment problem, not logic
        raise RuntimeError("`datasets` is required to build the pool: pip install datasets") from exc

    try:
        dataset = load_dataset(cfg.pool_id, split=cfg.pool_split)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load {cfg.pool_id} split {cfg.pool_split!r}: {exc}\n"
            "This dataset mixes arrow (train) and json (test) formats, which breaks the web "
            "viewer and can confuse split inference. Try data.pool_split: 'train', or load "
            "explicitly with load_dataset(..., data_files=...)."
        ) from exc

    items: list[PoolItem] = []
    for row in dataset:
        item = parse_row(row)
        if item is not None:
            items.append(item)
        if cfg.pool_max_items is not None and len(items) >= cfg.pool_max_items:
            break

    if not items:
        raise RuntimeError(
            f"Parsed 0 usable items from {cfg.pool_id}. The upstream schema likely changed; "
            "check src/rapfprm/data/pool.py against a sample row."
        )
    return items


def save_pool(items: list[PoolItem], pool_dir: str | Path) -> Path:
    pool_dir = Path(pool_dir)
    pool_dir.mkdir(parents=True, exist_ok=True)
    path = pool_dir / "pool.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
    return path


def load_pool(pool_dir: str | Path) -> list[PoolItem]:
    path = Path(pool_dir) / "pool.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"No retrieval pool at {path}. Run: python scripts/build_pool.py --config <cfg>"
        )
    with path.open(encoding="utf-8") as handle:
        return [PoolItem.from_dict(json.loads(line)) for line in handle if line.strip()]


def pool_stats(items: list[PoolItem]) -> dict:
    """Summary written into the run directory so the pool is documented alongside results."""
    questions = {item.qid for item in items}
    error_types: dict[str, int] = {}
    sources: dict[str, int] = {}
    for item in items:
        error_types[item.error_type] = error_types.get(item.error_type, 0) + 1
        key = item.source or "unknown"
        sources[key] = sources.get(key, 0) + 1
    return {
        "n_items": len(items),
        "n_unique_questions": len(questions),
        "error_types": error_types,
        "sources": sources,
    }
