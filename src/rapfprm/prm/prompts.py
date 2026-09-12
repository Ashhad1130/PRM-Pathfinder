"""Prompt construction — the only thing that differs between Condition A and Condition B.

Condition A reproduces the PathFinder-PRM-7B model card exactly:

    user      : PROMPT_PREFIX + "\\n\\n Question: " + question
    assistant : "\\n\\n".join(prev_steps) + "\\n\\nCurrent Step: " + step
                + " Math reasoning: <extra>, Consistency: <extra>"

Condition B inserts a reference block into the **user** turn only. The assistant turn —
which carries the `<extra>` mask positions the model actually scores — is byte-identical
across conditions. That is what keeps the comparison honest: same weights, same masks,
same decoding rule, more context.

Safety invariant enforced here: reference text may never contain `<extra>`. A stray mask
token inside a reference would add a scoring position and silently corrupt every verdict.
`sanitise_reference` strips them and `count_mask_tokens` lets the backend assert the
expected count before it trusts a forward pass.
"""

from __future__ import annotations

from ..config import PromptConfig
from ..retrieval.retriever import Reference

MASK = "<extra>"
POS = "<+>"
NEG = "<->"

CURRENT_STEP_MARKER = "\n\nCurrent Step: "
JUDGEMENT_SCAFFOLD = f" Math reasoning: {MASK}, Consistency: {MASK}"
CORRECTNESS_SCAFFOLD = f", Correctness: {MASK}"

REFERENCE_HEADER = (
    "\n\nBelow are similar solution steps from other problems that have already been "
    "graded by a teacher. Use them as reference for what kinds of mistakes look like.\n"
)


def sanitise_reference(text: str, max_chars: int) -> str:
    """Strip mask tokens from retrieved text and bound its length."""
    cleaned = text.replace(MASK, " ").strip()
    if max_chars and len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip() + " ..."
    return cleaned


def count_mask_tokens(text: str) -> int:
    return text.count(MASK)


def render_reference_block(references: list[Reference], cfg: PromptConfig) -> str:
    """Render retrieved exemplars as reference text for the user turn."""
    if not references:
        return ""

    blocks: list[str] = [REFERENCE_HEADER]
    for position, reference in enumerate(references, start=1):
        item = reference.item
        lines = [
            f"[Reference {position}]",
            f"Question: {sanitise_reference(item.question, cfg.max_reference_chars)}",
            f"Step: {sanitise_reference(item.step, cfg.max_reference_chars)}",
        ]
        if cfg.include_reference_labels:
            math_token = POS if item.math_ok else NEG
            consistency_token = POS if item.consistency_ok else NEG
            lines.append(
                f"Teacher's judgement: Math reasoning: {math_token}, "
                f"Consistency: {consistency_token} ({item.error_type})"
            )
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def build_user_turn(question: str, references: list[Reference], cfg: PromptConfig) -> str:
    """PROMPT_PREFIX [+ references] + the question, in the model card's exact shape."""
    return cfg.prompt_prefix + render_reference_block(references, cfg) + "\n\n Question: " + question


def build_assistant_turn(prev_steps: tuple[str, ...] | list[str], step: str) -> str:
    """The step under judgement plus the two mask positions. Never varies by condition."""
    return "\n\n".join(prev_steps) + CURRENT_STEP_MARKER + step + JUDGEMENT_SCAFFOLD


def build_messages(
    question: str,
    prev_steps: tuple[str, ...] | list[str],
    step: str,
    references: list[Reference] | None,
    cfg: PromptConfig,
) -> list[dict[str, str]]:
    """Full two-turn chat input for stage 1."""
    messages = [
        {"role": "user", "content": build_user_turn(question, references or [], cfg)},
        {"role": "assistant", "content": build_assistant_turn(prev_steps, step)},
    ]

    stray = count_mask_tokens(messages[0]["content"])
    if stray:
        raise AssertionError(
            f"{stray} stray {MASK} token(s) leaked into the user turn. This would add "
            "phantom scoring positions and corrupt every verdict."
        )
    expected = count_mask_tokens(messages[1]["content"])
    if expected != 2:
        raise AssertionError(
            f"Assistant turn must contain exactly 2 {MASK} tokens (math, consistency); "
            f"found {expected}."
        )
    return messages


def advance_to_correctness(messages: list[dict[str, str]], stage1_tokens: list[str]) -> list[dict]:
    """Build the stage-2 input: fill in stage-1 verdicts, then ask for Correctness.

    Mirrors the model card: replace each `<extra>` with the token that was predicted for
    it, in order, then append the correctness scaffold.
    """
    advanced = [dict(turn) for turn in messages]
    content = advanced[-1]["content"]
    for token in stage1_tokens:
        content = content.replace(MASK, token, 1)
    content += CORRECTNESS_SCAFFOLD
    advanced[-1]["content"] = content

    if count_mask_tokens(content) != 1:
        raise AssertionError(
            "Stage-2 assistant turn must contain exactly 1 mask token (correctness); found "
            f"{count_mask_tokens(content)}."
        )
    return advanced
