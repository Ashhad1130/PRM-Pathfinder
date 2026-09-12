"""The real scoring path, exercised without a GPU.

`MockPRM` bypasses `PathFinderPRM` entirely, so the adapter's actual logic — shifted mask,
stage-1 token comparison, the short circuit, the stage-2 softmax, reference dropping —
had no automated coverage. Loading the real 7B for every test is not an option, so the
tokenizer and model are stubbed with the smallest things that behave correctly.

These tests are what stop a refactor of `pathfinder.py` from silently changing verdicts.
"""

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from rapfprm.config import PRMConfig, PromptConfig
from rapfprm.prm.pathfinder import PathFinderPRM
from rapfprm.prm.prompts import MASK, NEG, POS

POS_ID, NEG_ID, MASK_ID = 101, 102, 103
VOCAB = 200


class StubBatch(dict):
    """Mimics transformers' BatchEncoding: a mapping that also supports `.to(device)`."""

    def to(self, device):
        return self


class StubTokenizer:
    """Maps the three special tokens to fixed ids; everything else hashes to filler."""

    def encode(self, text):
        return {POS: [POS_ID], NEG: [NEG_ID], MASK: [MASK_ID]}[text]

    def decode(self, ids, skip_special_tokens=False):
        return {POS_ID: POS, NEG_ID: NEG, MASK_ID: MASK}[int(ids[0])]

    def apply_chat_template(self, messages, tokenize=True, return_dict=True, return_tensors="pt"):
        ids = []
        for turn in messages:
            for chunk in turn["content"].split(MASK):
                ids.extend(7 for _ in chunk.split())
                ids.append(MASK_ID)
            ids.pop()  # split() leaves one MASK too many
        return StubBatch(
            input_ids=torch.tensor([ids]),
            attention_mask=torch.ones(1, len(ids), dtype=torch.long),
        )


class StubOutput:
    def __init__(self, logits):
        self.logits = logits


class StubModel:
    """Returns scripted verdicts at the positions preceding each mask token."""

    def __init__(self, verdicts):
        #: (pos_logit, neg_logit) pairs consumed in order across ALL forward passes —
        #: stage 1 takes two, stage 2 takes the next one.
        self.verdicts = list(verdicts)
        self.cursor = 0
        self.device = torch.device("cpu")
        self.lm_head = object()
        self.calls = 0

    def __call__(self, input_ids=None, attention_mask=None, **kwargs):
        self.calls += 1
        seq = input_ids.shape[-1]
        logits = torch.zeros(1, seq, VOCAB)
        mask_positions = (input_ids[0] == MASK_ID).nonzero().flatten().tolist()
        for position in mask_positions:
            pos_logit, neg_logit = self.verdicts[self.cursor]
            self.cursor += 1
            logits[0, position - 1, POS_ID] = pos_logit
            logits[0, position - 1, NEG_ID] = neg_logit
        return StubOutput(logits)

    def eval(self):
        return self


def make_prm(verdicts, **cfg_kwargs):
    """Build a PathFinderPRM with the stubs, skipping __init__'s model download."""
    prm = PathFinderPRM.__new__(PathFinderPRM)
    prm.cfg = PRMConfig(**cfg_kwargs)
    prm.prompt_cfg = PromptConfig()
    prm.name = "stub"
    prm._torch = torch
    prm.dropped_references = 0
    prm.tokenizer = StubTokenizer()
    prm.model = StubModel(verdicts)
    prm.pos_token_id, prm.neg_token_id, prm.mask_token_id = POS_ID, NEG_ID, MASK_ID
    prm.allowed_token_ids = torch.tensor([POS_ID, NEG_ID])
    return prm


def test_clean_step_runs_both_stages_and_returns_a_probability():
    # stage 1: math <+>, consistency <+>; stage 2: correctness strongly <+>
    prm = make_prm([(5.0, 0.0), (5.0, 0.0), (4.0, 0.0)])
    verdict = prm.score_step("Q?", ("prev step",), "current step", [])

    assert verdict.math_ok is True
    assert verdict.consistency_ok is True
    assert verdict.correctness_prob is not None
    assert verdict.correctness_prob > 0.9
    assert prm.model.calls == 2, "a clean step must run stage 1 and stage 2"


def test_math_error_short_circuits_before_stage_two():
    """The model card returns -1 here and never runs the second pass."""
    prm = make_prm([(0.0, 5.0), (5.0, 0.0)])
    verdict = prm.score_step("Q?", (), "bad step", [])

    assert verdict.math_ok is False
    assert verdict.consistency_ok is True
    assert verdict.correctness_prob is None
    assert verdict.error_type == "math error"
    assert prm.model.calls == 1, "stage 2 must not run once stage 1 finds an error"


def test_consistency_error_is_reported_separately_from_math():
    """Keeping the two apart is the project's dependent variable."""
    prm = make_prm([(5.0, 0.0), (0.0, 5.0)])
    verdict = prm.score_step("Q?", (), "step", [])

    assert verdict.math_ok is True
    assert verdict.consistency_ok is False
    assert verdict.error_type == "consistency error"


def test_both_errors_are_reported_together():
    prm = make_prm([(0.0, 5.0), (0.0, 5.0)])
    verdict = prm.score_step("Q?", (), "step", [])
    assert verdict.error_type == "math and consistency error"


def test_low_correctness_probability_counts_as_an_error_at_threshold():
    prm = make_prm([(5.0, 0.0), (5.0, 0.0), (0.0, 4.0)])
    verdict = prm.score_step("Q?", (), "step", [])

    assert verdict.stage1_error is False
    assert verdict.correctness_prob < 0.5
    assert verdict.is_error(0.5) is True


def test_fit_to_budget_returns_the_encoding_it_measured():
    """The encoding is reused for the forward pass; encoding twice per step is wasteful."""
    prm = make_prm([(5.0, 0.0), (5.0, 0.0), (4.0, 0.0)])
    messages, encoded = prm._fit_to_budget("Q?", ("a",), "b", [])

    assert isinstance(messages, list) and len(messages) == 2
    assert "input_ids" in encoded
    # The returned encoding must correspond to the returned messages.
    assert encoded["input_ids"].shape[-1] == prm._encode(messages)["input_ids"].shape[-1]


def test_references_are_dropped_when_the_prompt_exceeds_the_budget():
    from rapfprm.data.pool import PoolItem
    from rapfprm.retrieval.retriever import Reference

    refs = [
        Reference(
            item=PoolItem(
                qid=f"q{i}",
                question="a fairly wordy reference question " * 5,
                prev_steps=(),
                step="a fairly wordy reference step " * 5,
                math_ok=True,
                consistency_ok=True,
            ),
            question_similarity=0.5,
            step_similarity=0.5,
        )
        for i in range(3)
    ]

    prm = make_prm([(5.0, 0.0), (5.0, 0.0), (4.0, 0.0)], max_input_tokens=40)
    messages, _ = prm._fit_to_budget("Q?", (), "step", refs)

    assert prm.dropped_references > 0
    assert messages[0]["content"].count("[Reference ") < len(refs)


def test_solution_content_is_never_dropped_to_fit():
    """Only references are droppable; the step under judgement must survive intact."""
    prm = make_prm([(5.0, 0.0), (5.0, 0.0), (4.0, 0.0)], max_input_tokens=5)
    messages, _ = prm._fit_to_budget("Q?", ("earlier",), "the step being judged", [])

    assert "the step being judged" in messages[1]["content"]
    assert messages[1]["content"].count(MASK) == 2


def test_a_stray_mask_position_is_caught_rather_than_scored():
    """A phantom mask would silently corrupt every verdict, so it must raise."""
    prm = make_prm([(5.0, 0.0), (5.0, 0.0), (5.0, 0.0)])
    with pytest.raises(AssertionError, match="stray"):
        prm.score_step(f"Question with a {MASK} in it", (), "step", [])
