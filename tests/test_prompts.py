"""Prompt invariants — the ones that keep Condition A vs B a fair comparison."""

import pytest

from rapfprm.config import PromptConfig
from rapfprm.data.pool import PoolItem
from rapfprm.prm.prompts import (
    advance_to_correctness,
    build_messages,
    render_reference_block,
    sanitise_reference,
)
from rapfprm.retrieval.retriever import Reference

CFG = PromptConfig()


def make_reference(step="We compute 2 + 2 = 5.", math_ok=False, consistency_ok=True):
    return Reference(
        item=PoolItem(
            qid="abc123",
            question="What is 2 + 2?",
            prev_steps=("Read the problem.",),
            step=step,
            math_ok=math_ok,
            consistency_ok=consistency_ok,
            source="test",
        ),
        question_similarity=0.5,
        step_similarity=0.4,
    )


def test_assistant_turn_is_identical_with_and_without_references():
    """The scored half of the prompt must not vary by condition."""
    without = build_messages("Q?", ("s1",), "s2", [], CFG)
    with_refs = build_messages("Q?", ("s1",), "s2", [make_reference()], CFG)
    assert without[1]["content"] == with_refs[1]["content"]
    assert without[0]["content"] != with_refs[0]["content"]


def test_exactly_two_masks_in_stage_one():
    messages = build_messages("Q?", (), "only step", [make_reference()], CFG)
    assert messages[1]["content"].count("<extra>") == 2
    assert "<extra>" not in messages[0]["content"]


def test_mask_token_in_retrieved_text_is_stripped():
    """A stray <extra> in a reference would add a phantom scoring position."""
    poisoned = make_reference(step="Evil step <extra> with a mask.")
    block = render_reference_block([poisoned], CFG)
    assert "<extra>" not in block

    messages = build_messages("Q?", (), "step", [poisoned], CFG)
    assert messages[0]["content"].count("<extra>") == 0


def test_build_messages_rejects_a_question_carrying_a_mask():
    with pytest.raises(AssertionError, match="stray"):
        build_messages("What is <extra> here?", (), "step", [], CFG)


def test_reference_labels_use_the_models_own_tokens():
    block = render_reference_block([make_reference(math_ok=False)], CFG)
    assert "Math reasoning: <->" in block
    assert "Consistency: <+>" in block
    assert "math error" in block


def test_reference_labels_can_be_omitted():
    cfg = PromptConfig(include_reference_labels=False)
    block = render_reference_block([make_reference()], cfg)
    assert "Teacher's judgement" not in block


def test_empty_reference_list_renders_nothing():
    assert render_reference_block([], CFG) == ""


def test_reference_text_is_truncated_to_budget():
    cfg = PromptConfig(max_reference_chars=20)
    long_step = "x" * 500
    assert len(sanitise_reference(long_step, cfg.max_reference_chars)) < 40


def test_stage_two_fills_verdicts_and_appends_correctness():
    messages = build_messages("Q?", (), "step", [], CFG)
    stage2 = advance_to_correctness(messages, ["<+>", "<+>"])
    content = stage2[-1]["content"]
    assert content.endswith(" Math reasoning: <+>, Consistency: <+>, Correctness: <extra>")
    assert content.count("<extra>") == 1


def test_previous_steps_are_joined_with_blank_lines():
    messages = build_messages("Q?", ("one", "two"), "three", [], CFG)
    assert messages[1]["content"].startswith("one\n\ntwo\n\nCurrent Step: three")
