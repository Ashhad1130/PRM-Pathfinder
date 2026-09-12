"""PathFinder-PRM-7B adapter.

This mirrors the official model card (`declare-lab/PathFinder-PRM-7B`) — prompts, special
tokens and scoring procedure are reproduced byte for byte. Treat them as a FROZEN CONTRACT
(see docs/MODEL_INTERFACE.md): if they drift, Condition A stops reproducing the published
baseline and the comparison loses its anchor. `scripts/verify_model_interface.py` re-checks
them against the card's worked example.

There is exactly ONE intentional departure, marked in full at the load site below: the card
writes `AutoModel`, which on current transformers loads a headless base model with no
`.logits`, making its own algorithm impossible. We load `AutoModelForCausalLM`.

How the model scores a step:

1. Build a two-turn chat where the assistant turn ends with
   ``... Math reasoning: <extra>, Consistency: <extra>``.
2. One forward pass. At each position *preceding* an ``<extra>`` token (hence the shifted
   mask), compare the logits of ``<+>`` and ``<->``; the larger one is the prediction.
3. If either prediction is ``<->`` the step is erroneous and scoring stops — the model
   card returns -1 here and never runs stage 2.
4. Otherwise fill the predictions in, append ``, Correctness: <extra>``, run a second
   forward pass, and softmax ``<+>`` against ``<->`` to get the correctness probability.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from ..config import PRMConfig, PromptConfig
from ..retrieval.retriever import Reference
from .base import StepVerdict
from .prompts import POS, NEG, MASK, advance_to_correctness, build_messages

logger = logging.getLogger(__name__)


def _resolve_attn_implementation(requested: str) -> str:
    """flash_attention_2 when it is genuinely available, else sdpa."""
    if requested != "auto":
        return requested
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        # Expected on Windows and on any machine without the compiled kernel.
        return "sdpa"


def _dtype_kwarg(dtype_name: str) -> dict:
    """`torch_dtype` was renamed to `dtype` in transformers v5; support both.

    `from_pretrained` swallows unknown keys into **kwargs rather than raising, so passing
    the wrong one would silently load in the default dtype — hence the explicit check.
    """
    import torch
    import transformers

    dtype = getattr(torch, dtype_name)
    try:
        major = int(transformers.__version__.split(".")[0])
    except (AttributeError, ValueError):
        major = 4
    return {"dtype" if major >= 5 else "torch_dtype": dtype}


class PathFinderPRM:
    """Frozen PathFinder-PRM-7B, scored one step at a time."""

    def __init__(
        self,
        cfg: PRMConfig,
        prompt_cfg: PromptConfig,
        load_in_4bit: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.prompt_cfg = prompt_cfg
        precision_tag = {"hf4bit": " (nf4)", "int4": " (int4-torchao)"}.get(cfg.backend, "")
        self.name = f"{cfg.model_name}{precision_tag}"
        self._torch = torch
        self.dropped_references = 0

        torch.manual_seed(cfg.seed)

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name, trust_remote_code=cfg.trust_remote_code
        )

        model_kwargs = {
            "device_map": cfg.device_map,
            "trust_remote_code": cfg.trust_remote_code,
            "attn_implementation": _resolve_attn_implementation(cfg.attn_implementation),
            **_dtype_kwarg(cfg.dtype),
        }

        if cfg.max_memory:
            # YAML gives string keys; accelerate wants int ordinals for GPUs and "cpu"/"disk".
            model_kwargs["max_memory"] = {
                (int(k) if str(k).isdigit() else k): v for k, v in cfg.max_memory.items()
            }
        if cfg.offload_folder:
            Path(cfg.offload_folder).mkdir(parents=True, exist_ok=True)
            model_kwargs["offload_folder"] = cfg.offload_folder
            logger.warning(
                "Disk offload is enabled (%s). Layers stream from disk on every forward "
                "pass — expect seconds per step. Use it to validate correctness on a few "
                "solutions, never for a full benchmark run.",
                cfg.offload_folder,
            )

        if cfg.backend == "int4":
            # int4 weight-only via torchao. The point is speed, not memory frugality for
            # its own sake: at ~5.5 GB the whole model sits in 8 GB of VRAM, so nothing
            # offloads to disk — and disk offload is a ~150x penalty per forward pass.
            #
            # `lm_head` and the embeddings are deliberately left in bf16
            # (`include_input_output_embeddings=False`, the default). That is not a
            # rounding decision: PathFinder's verdict IS a comparison of the <+> and <->
            # logits produced by lm_head, so quantising that head would inject noise
            # directly into the quantity being measured.
            try:
                from transformers import TorchAoConfig
                from torchao.quantization import Int4WeightOnlyConfig
            except ImportError as exc:
                raise RuntimeError(
                    "backend 'int4' needs torchao: pip install torchao"
                ) from exc

            model_kwargs["quantization_config"] = TorchAoConfig(
                quant_type=Int4WeightOnlyConfig(
                    group_size=cfg.int4_group_size,
                    int4_packing_format=cfg.int4_packing_format,
                )
            )
            # Quantised weights are built on the GPU; an accelerate memory cap or an
            # offload folder would fight that and push layers back to disk.
            model_kwargs.pop("max_memory", None)
            model_kwargs.pop("offload_folder", None)
            logger.warning(
                "Loading int4 (torchao, group_size=%d, packing=%s). Verdicts come from a "
                "<+> vs <-> logit comparison and quantisation perturbs those logits, so "
                "absolute F1 will not match the published 69.5. Deltas are valid only "
                "between runs at the SAME precision — never mix int4 and bf16 numbers.",
                cfg.int4_group_size,
                cfg.int4_packing_format,
            )

        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("4-bit loading needs a transformers build with bitsandbytes") from exc
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=getattr(torch, cfg.dtype),
            )
            logger.warning(
                "Loading in 4-bit NF4. Verdicts come from a <+> vs <-> logit comparison, and "
                "quantisation perturbs exactly those logits — absolute F1 will not match the "
                "published 69.5. Only A-vs-B deltas at the same precision are meaningful."
            )

        # DELIBERATE DEVIATION FROM THE MODEL CARD, which writes `AutoModel`.
        #
        # The checkpoint declares `architectures: ["Qwen2ForCausalLM"]`, ships
        # `lm_head.weight`, and carries no custom modeling code. On current transformers
        # `AutoModel` therefore resolves to the *base* `Qwen2Model`: it silently drops the
        # LM head and returns `BaseModelOutputWithPast`, which has no `.logits` at all —
        # `AttributeError: 'BaseModelOutputWithPast' object has no attribute 'logits'`.
        #
        # The card's algorithm is defined entirely in terms of vocabulary logits at the
        # mask positions, so the head is not optional; `AutoModelForCausalLM` is the
        # faithful reading of what the card *does*, not a departure from it.
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs).eval()

        if not hasattr(self.model, "lm_head"):
            raise RuntimeError(
                f"{cfg.model_name} loaded without an LM head, so there are no token logits "
                "to read <+> against <->. Scoring is impossible with this class."
            )

        # Special tokens, exactly as the model card obtains them.
        self.pos_token_id = self.tokenizer.encode(POS)[0]
        self.neg_token_id = self.tokenizer.encode(NEG)[0]
        self.mask_token_id = self.tokenizer.encode(MASK)[0]
        if len({self.pos_token_id, self.neg_token_id, self.mask_token_id}) != 3:
            raise RuntimeError(
                "The tokenizer collapsed <+>, <-> and <extra> onto overlapping ids. The "
                "checkpoint's special tokens are missing — scoring would be meaningless."
            )
        self.allowed_token_ids = torch.tensor(
            [self.pos_token_id, self.neg_token_id], device=self.model.device
        )

    # ---- tokenisation helpers -----------------------------------------------------------

    def _encode(self, messages: list[dict[str, str]]):
        return self.tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)

    def _shifted_mask(self, input_ids):
        """Positions whose logits predict the following <extra> token."""
        torch = self._torch
        token_masks = input_ids == self.mask_token_id
        return torch.cat(
            [
                token_masks[:, 1:],
                torch.zeros(token_masks.size(0), 1, dtype=torch.bool, device=input_ids.device),
            ],
            dim=1,
        )

    def _fit_to_budget(
        self,
        question: str,
        prev_steps: Sequence[str],
        step: str,
        references: list[Reference],
    ) -> tuple[list[dict[str, str]], dict]:
        """Drop references (never solution content) until the prompt fits the budget.

        Returns the messages *and* the encoding that was used to measure them, so the
        caller can feed the model directly. Encoding an 841-token prompt is not free and
        this used to happen twice per step — measure, throw away, then encode again.
        """
        working = list(references)
        while True:
            messages = build_messages(question, tuple(prev_steps), step, working, self.prompt_cfg)
            encoded = self._encode(messages)
            length = encoded["input_ids"].shape[-1]

            if length <= self.cfg.max_input_tokens or not working:
                if length > self.cfg.max_input_tokens:
                    logger.warning(
                        "Prompt is %d tokens, over the %d budget, with no references left to "
                        "drop. Proceeding; the solution itself is oversized.",
                        length,
                        self.cfg.max_input_tokens,
                    )
                return messages, encoded

            working.pop()
            self.dropped_references += 1

    # ---- scoring ------------------------------------------------------------------------

    def score_step(
        self,
        question: str,
        prev_steps: Sequence[str],
        step: str,
        references: list[Reference] | None = None,
    ) -> StepVerdict:
        torch = self._torch
        import torch.nn.functional as F

        references = references or []
        # Reuse the encoding produced by the budget check instead of redoing it.
        messages, encoded = self._fit_to_budget(question, prev_steps, step, references)
        n_references = messages[0]["content"].count("[Reference ")

        # --- stage 1: math reasoning + consistency -------------------------------------
        shifted_mask = self._shifted_mask(encoded["input_ids"])

        n_positions = int(shifted_mask.sum().item())
        if n_positions != 2:
            raise AssertionError(
                f"Stage 1 expected exactly 2 mask positions, found {n_positions}. A mask "
                "token leaked into the prompt or the chat template changed."
            )

        with torch.no_grad():
            outputs = self.model(**encoded)

        masked_logits = outputs.logits[shifted_mask][:, self.allowed_token_ids]
        predicted = self.allowed_token_ids[masked_logits.argmax(dim=-1)]
        decoded = [
            self.tokenizer.decode([int(token_id)], skip_special_tokens=False)
            for token_id in predicted
        ]
        math_ok = decoded[0] == POS
        consistency_ok = decoded[1] == POS

        if NEG in decoded:
            # Model card short-circuits here; stage 2 never runs.
            return StepVerdict(
                math_ok=math_ok,
                consistency_ok=consistency_ok,
                correctness_prob=None,
                n_references=n_references,
            )

        # --- stage 2: correctness -------------------------------------------------------
        stage2_messages = advance_to_correctness(messages, decoded)
        encoded2 = self._encode(stage2_messages)
        shifted_mask2 = self._shifted_mask(encoded2["input_ids"])

        n_positions2 = int(shifted_mask2.sum().item())
        if n_positions2 != 1:
            raise AssertionError(
                f"Stage 2 expected exactly 1 mask position, found {n_positions2}."
            )

        with torch.no_grad():
            outputs2 = self.model(**encoded2)

        restricted = outputs2.logits[shifted_mask2][:, [self.pos_token_id, self.neg_token_id]]
        probabilities = F.softmax(restricted, dim=-1)

        return StepVerdict(
            math_ok=True,
            consistency_ok=True,
            correctness_prob=float(probabilities[0][0].cpu().item()),
            n_references=n_references,
        )
