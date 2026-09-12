# The frozen PathFinder-PRM contract

Everything in this file is a **contract, not a design choice**. Condition A only means
something if it reproduces the published baseline, and it only reproduces the published
baseline if we talk to the model exactly the way its authors do.

Source: the `declare-lab/PathFinder-PRM-7B` model card, fetched 2026-09-06. The worked
example from that card is embedded verbatim in `scripts/verify_model_interface.py`.

## The three special tokens

| Token | Meaning | Obtained as |
| --- | --- | --- |
| `<+>` | positive / no error | `tokenizer.encode("<+>")[0]` |
| `<->` | negative / error | `tokenizer.encode("<->")[0]` |
| `<extra>` | a position to be predicted | `tokenizer.encode("<extra>")[0]` |

The adapter asserts these resolve to three distinct ids. If a tokenizer ever collapses
them, every verdict silently becomes noise — so that check fails loudly instead.

## The two turns

```python
user      = PROMPT_PREFIX + "\n\n Question: " + question
assistant = "\n\n".join(prev_steps) + "\n\nCurrent Step: " + curr_step \
            + " Math reasoning: <extra>, Consistency: <extra>"
```

Note the details that look like typos and are not:

- `"\n\n Question: "` has a **space after the second newline**.
- `PROMPT_PREFIX` contains the misspelling **"mathemetical"**. It is in the released
  prefix, so it is in ours. The model was tuned with it.
- The judgement scaffold begins with a **single leading space**, not a newline.

`scripts/verify_model_interface.py` compares our generated strings against the card's
byte for byte. Run it after touching anything in `src/rapfprm/prm/prompts.py`.

## The scoring procedure

1. **Shifted mask.** The logits that predict an `<extra>` token sit at the position
   *before* it, so the mask is shifted left by one and padded with `False`:

   ```python
   token_masks  = input_ids == mask_token_id
   shifted_mask = cat([token_masks[:, 1:], zeros(batch, 1, dtype=bool)], dim=1)
   ```

2. **Stage 1 — one forward pass.** At the two masked positions, restrict the logits to
   `[pos_token_id, neg_token_id]` and take the argmax. Position 0 is *Math reasoning*,
   position 1 is *Consistency*.

3. **Short circuit.** If either is `<->`, the step is erroneous and stage 2 never runs.
   The model card returns `-1` here. We return a `StepVerdict` that keeps *which* of the
   two failed — that distinction is the error-typing signal this project is about, and
   throwing it away would discard our main dependent variable.

4. **Stage 2 — a second forward pass.** Replace each `<extra>` with the token predicted
   for it, append `", Correctness: <extra>"`, re-encode, and softmax `<+>` against `<->`
   at the single remaining masked position. `P(<+>)` is the step's reward.

5. **Final verdict.** A step is erroneous iff stage 1 flagged it **or**
   `P(<+>) < prm.correctness_threshold` (default 0.5).

## Invariants the code enforces

| Invariant | Where | Why it matters |
| --- | --- | --- |
| exactly 2 mask positions in stage 1 | `pathfinder.py`, `prompts.py` | a stray mask adds a phantom scoring position |
| exactly 1 mask position in stage 2 | `pathfinder.py`, `prompts.py` | same |
| no `<extra>` in the user turn | `prompts.build_messages` | retrieved text is untrusted and could contain one |
| assistant turn identical across A and B | `verify_model_interface.py`, `test_prompts.py` | otherwise the delta measures prompt drift, not retrieval |
| `<+>`, `<->`, `<extra>` are distinct ids | `PathFinderPRM.__init__` | a broken tokenizer would produce plausible-looking noise |

## Where retrieval is allowed to touch the prompt

**The user turn only.** References are inserted between `PROMPT_PREFIX` and
`"\n\n Question: "`. The assistant turn — which holds every masked position the model
actually scores — is byte-identical in both conditions. That single restriction is what
makes Condition B a test of retrieval rather than a test of prompt engineering.

## The one place we deliberately deviate from the card

The card writes:

```python
model = AutoModel.from_pretrained(model_name, ...)
...
outputs.logits[shifted_mask]
```

**That does not run on current transformers**, and the failure is instructive:

```
AttributeError: 'BaseModelOutputWithPast' object has no attribute 'logits'
```

Checked against the checkpoint itself:

| Fact | Value |
| --- | --- |
| `config.json` → `architectures` | `["Qwen2ForCausalLM"]` |
| `auto_map` / custom modeling code | none |
| `lm_head.weight` in the shard index | present |

So `AutoModel` resolves to the base `Qwen2Model`, which **silently discards the LM head**
and returns hidden states — no vocabulary logits at all. Since the card's entire algorithm
is "compare the `<+>` and `<->` logits at the mask positions", the head is not optional.

We therefore load with **`AutoModelForCausalLM`** and assert the model has an `lm_head`.
This is the faithful implementation of what the card *does*; the class name in the snippet
is simply wrong for a checkpoint with a head. Everything else — prompt strings, mask
shifting, the two-stage procedure, the short circuit — is unchanged.

If you ever see `BaseModelOutputWithPast` in a traceback here, someone has reverted this.

## Environment notes
- **`torch_dtype` → `dtype`.** transformers v5 renamed the argument. `_dtype_kwarg()`
  picks the right one for the installed version. (This machine has transformers 5.14.1.)
- **flash-attention.** The card passes `attn_implementation="flash_attention_2"`, which is
  not installable on Windows. `attn_implementation: auto` uses flash-attn when it imports
  and falls back to `sdpa` otherwise. This changes speed, not numerics.
- **A broken pyarrow stops the model loading at all.** `transformers.generation.
  candidate_generator` does `from sklearn.metrics import roc_curve`, and scikit-learn does
  an `import pyarrow` guarded only by `except ModuleNotFoundError`. A pyarrow that is
  *installed but unloadable* raises plain `ImportError`, escapes that guard, and takes
  sklearn — and therefore transformers — down with it. `src/rapfprm/compat.py` presents
  such a pyarrow as absent so the guards work; it is a no-op on healthy machines. Symptom
  to recognise: `ImportError: DLL load failed while importing lib` deep in a traceback
  that mentions `sklearn/utils/fixes.py`.
- **Memory.** `prm.max_memory` and `prm.offload_folder` map onto accelerate's budget and
  disk-offload directory. Needed on any machine that cannot hold ~15 GB of weights; see
  `configs/pilot-local.yaml`. Offloaded layers stream per forward pass, so this is for
  correctness pilots, not benchmark runs.

## If the check fails

Do not "fix" the comparison by adjusting the metric. Work through, in order:

1. Did the model card change? Re-fetch it and diff against the constants in
   `scripts/verify_model_interface.py`.
2. Did the tokenizer's chat template change? Print
   `tokenizer.apply_chat_template(messages, tokenize=False)` and inspect it.
3. Did a transformers upgrade move the logits or rename an argument?

Only once the offline check passes again is Condition A worth running.
