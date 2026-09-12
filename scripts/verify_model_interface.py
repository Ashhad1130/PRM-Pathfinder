"""Guard the frozen PathFinder-PRM contract.

Two levels:

  python scripts/verify_model_interface.py
      Offline. Rebuilds the model card's worked example through our prompt builders and
      asserts the strings match the card BYTE FOR BYTE, and that adding references leaves
      the assistant turn untouched. Catches ~every drift that would silently invalidate
      Condition A, in under a second, with no download.

  python scripts/verify_model_interface.py --load-model
      Additionally loads the 7B checkpoint and scores the card's example. The card's step
      is correct, so a healthy adapter returns a correctness probability, not an error.

Run the offline check in CI and before every experiment; run the model check whenever the
checkpoint, transformers version or backend changes.
"""

from __future__ import annotations

import argparse
import sys

from _bootstrap import setup_logging  # noqa: F401

from rapfprm.config import PRMConfig, PromptConfig
from rapfprm.prm.prompts import advance_to_correctness, build_messages

# ---------------------------------------------------------------------------------------
# Copied verbatim from the declare-lab/PathFinder-PRM-7B model card (fetched 2026-09-06).
# Do not reformat: the point of this file is byte-level fidelity.
# ---------------------------------------------------------------------------------------

PROMPT_PREFIX = "You are a Math Teacher. Given a question and a student's solution, evaluate the mathemetical correctness, logic consistency of the current step and whether it will lead to the correct final solution"

QUESTION = "Sue lives in a fun neighborhood.  One weekend, the neighbors decided to play a prank on Sue.  On Friday morning, the neighbors placed 18 pink plastic flamingos out on Sue's front yard.  On Saturday morning, the neighbors took back one third of the flamingos, painted them white, and put these newly painted white flamingos back out on Sue's front yard.  Then, on Sunday morning, they added another 18 pink plastic flamingos to the collection. At noon on Sunday, how many more pink plastic flamingos were out than white plastic flamingos?"

PREV_STEPS = [
    "To find out how many more pink plastic flamingos were out than white plastic flamingos at noon on Sunday, we can break down the problem into steps. First, on Friday, the neighbors start with 18 pink plastic flamingos.",
    "On Saturday, they take back one third of the flamingos. Since there were 18 flamingos, (1/3 \\times 18 = 6) flamingos are taken back. So, they have (18 - 6 = 12) flamingos left in their possession. Then, they paint these 6 flamingos white and put them back out on Sue's front yard. Now, Sue has the original 12 pink flamingos plus the 6 new white ones. Thus, by the end of Saturday, Sue has (12 + 6 = 18) pink flamingos and 6 white flamingos.",
    "On Sunday, the neighbors add another 18 pink plastic flamingos to Sue's front yard. By the end of Sunday morning, Sue has (18 + 18 = 36) pink flamingos and still 6 white flamingos.",
]

CURR_STEP = "To find the difference, subtract the number of white flamingos from the number of pink flamingos: (36 - 6 = 30). Therefore, at noon on Sunday, there were 30 more pink plastic flamingos out than white plastic flamingos. The answer is (\\boxed{30})."

# The exact two turns the card constructs.
EXPECTED_USER = PROMPT_PREFIX + "\n\n Question: " + QUESTION
EXPECTED_ASSISTANT = (
    "\n\n".join(PREV_STEPS)
    + "\n\nCurrent Step: "
    + CURR_STEP
    + " Math reasoning: <extra>, Consistency: <extra>"
)


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if not ok and detail:
        print(f"         {detail}")
    return ok


def offline_checks() -> bool:
    prompt_cfg = PromptConfig()
    passed = True

    print("Offline prompt-parity checks")

    passed &= check(
        "PROMPT_PREFIX matches the model card",
        prompt_cfg.prompt_prefix == PROMPT_PREFIX,
        f"config has: {prompt_cfg.prompt_prefix!r}",
    )

    baseline = build_messages(QUESTION, tuple(PREV_STEPS), CURR_STEP, [], prompt_cfg)

    passed &= check(
        "Condition A user turn is byte-identical to the card",
        baseline[0]["content"] == EXPECTED_USER,
    )
    passed &= check(
        "Condition A assistant turn is byte-identical to the card",
        baseline[1]["content"] == EXPECTED_ASSISTANT,
    )

    # The invariant that makes A vs B a fair test: references touch the user turn only.
    from rapfprm.data.pool import PoolItem
    from rapfprm.retrieval.retriever import Reference

    fake_reference = Reference(
        item=PoolItem(
            qid="deadbeef",
            question="A different problem about apples.",
            prev_steps=("Some earlier step.",),
            step="We add 2 and 3 to get 6.",
            math_ok=False,
            consistency_ok=True,
            source="unit-test",
        ),
        question_similarity=0.42,
        step_similarity=0.37,
    )
    augmented = build_messages(
        QUESTION, tuple(PREV_STEPS), CURR_STEP, [fake_reference], prompt_cfg
    )

    passed &= check(
        "Condition B leaves the assistant turn untouched",
        augmented[1]["content"] == baseline[1]["content"],
    )
    passed &= check(
        "Condition B does change the user turn",
        augmented[0]["content"] != baseline[0]["content"],
    )
    passed &= check(
        "Condition B user turn still ends with the question",
        augmented[0]["content"].endswith("\n\n Question: " + QUESTION),
    )
    passed &= check(
        "no <extra> leaks into the user turn",
        "<extra>" not in augmented[0]["content"],
    )
    passed &= check(
        "assistant turn holds exactly 2 mask tokens",
        augmented[1]["content"].count("<extra>") == 2,
    )

    stage2 = advance_to_correctness(baseline, ["<+>", "<+>"])
    passed &= check(
        "stage 2 fills verdicts and appends the correctness scaffold",
        stage2[-1]["content"].endswith(
            " Math reasoning: <+>, Consistency: <+>, Correctness: <extra>"
        ),
        f"got: ...{stage2[-1]['content'][-70:]!r}",
    )
    passed &= check(
        "stage 2 holds exactly 1 mask token",
        stage2[-1]["content"].count("<extra>") == 1,
    )

    return bool(passed)


def model_check(backend: str) -> bool:
    print(f"\nLive model check (backend={backend}) — this downloads ~15 GB on first run")

    cfg = PRMConfig(backend=backend)
    from rapfprm.prm.base import build_backend

    prm = build_backend(cfg, PromptConfig())
    verdict = prm.score_step(QUESTION, tuple(PREV_STEPS), CURR_STEP, [])

    print(f"  math_ok          : {verdict.math_ok}")
    print(f"  consistency_ok   : {verdict.consistency_ok}")
    print(f"  correctness_prob : {verdict.correctness_prob}")
    print(f"  error_type       : {verdict.error_type}")

    # The card's step is a correct final step; a healthy adapter should not flag it.
    return check(
        "card's correct step is not flagged as an error",
        not verdict.stage1_error,
        "Stage 1 flagged a step the model card treats as correct — the adapter, the "
        "tokenizer's special tokens, or the chat template is wrong.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-model", action="store_true", help="also run a real forward pass")
    parser.add_argument("--backend", default="hf", choices=["hf", "hf4bit"])
    args = parser.parse_args()

    setup_logging()
    ok = offline_checks()
    if args.load_model:
        ok = model_check(args.backend) and ok

    print("\n" + ("ALL CHECKS PASSED" if ok else "CHECKS FAILED — do not trust any results"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
