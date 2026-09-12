# Experiment protocol

## The comparison

| | Condition A | Condition B |
| --- | --- | --- |
| Model | PathFinder-PRM-7B, frozen | PathFinder-PRM-7B, frozen (same weights) |
| Prompt | model card, unmodified | + retrieved references in the **user turn only** |
| Assistant turn | `... Math reasoning: <extra>, Consistency: <extra>` | byte-identical |
| Threshold | 0.5 | 0.5 |
| Data | ProcessBench, 4 subsets | identical solutions, identical order |

Exactly one thing varies. Keep it that way: if you change `prm.*` in one config, change it
in the other, or the delta stops measuring retrieval.

## Metric

ProcessBench asks for the index of the **first** erroneous step, or −1 for a clean
solution. Reported per subset:

- `error_acc` — exact-index accuracy over solutions that contain an error
- `correct_acc` — accuracy over clean solutions (i.e. correctly predicting −1)
- `F1` — the **harmonic mean** of those two

The harmonic mean is what makes the benchmark adversarial: a grader that flags every step
scores `error_acc = 1.0`, `correct_acc = 0.0`, F1 = 0. Published anchors, average F1 across
the four subsets: RetrievalPRM-7B **65.8**, PathFinder-PRM-7B **69.5**.

## Controls

### 1. Contamination guard (mandatory)

**This is not hypothetical. Measured on this pool (50K items, 15,316 unique questions):**

| Subset | n | verbatim in pool | near-duplicate (cos ≥ 0.95) |
| --- | ---: | ---: | ---: |
| gsm8k | 400 | 0 (0.0%) | 0 |
| **math** | 1000 | **596 (59.6%)** | **602** |
| olympiadbench | 1000 | 0 (0.0%) | 4 |
| omnimath | 1000 | 0 (0.0%) | 22 |
| **total** | 3400 | **596 (17.5%)** | 628 |

Reproduce with `python scripts/check_contamination.py --config configs/retrieval.yaml`.

Nearly **60% of the MATH subset appears verbatim** in PathFinder-600K, and the other three
subsets are essentially clean. That concentration is the dangerous part: MATH is the
"moderate OOD" cell of the expected-evidence table, so an unguarded run would show a large
gain exactly where the hypothesis predicts a modest one — manufacturing a trend out of
memorisation. In a full unguarded run the guard's counters show this would have fired on
1,149 of 6,568 retrieval queries (17.5%).

Two consequences for the report:

1. Keep the guard on. Always. It is on by default.
2. Treat the **MATH row with suspicion even with the guard on**, and say so. The guard
   removes exact and near-exact question matches, not paraphrases or shared sub-problems.
   The cleanest supporting evidence for the hypothesis comes from OlympiadBench and
   OmniMATH, which are uncontaminated and also the most out-of-distribution.

ProcessBench and PathFinder-600K both descend from MATH and GSM8K. Without a guard, the
"retrieved reference" can be a labelled copy of the very step under judgement — Condition B
would win for a reason that has nothing to do with the mechanism.

`retrieval.max_question_similarity: 0.95` drops any pool question at or above that cosine
similarity to the eval question; `drop_exact_duplicates: true` also drops normalised exact
matches. Every filtered neighbour is counted in `summary.json`:

```json
"retrieval": { "filtered_exact_duplicate": 41, "filtered_similarity": 388, ... }
```

**Report those counts.** If they are near zero on the MATH subset, be suspicious — that is
where overlap is most likely, and a zero suggests the guard is not firing.

Sensitivity check worth running: re-run Condition B at `max_question_similarity: 0.85` and
`0.99`. If the gain vanishes at 0.85 but is large at 0.99, the gain was contamination.

### 2. Baseline fidelity

Condition A at bf16 should land near the published 69.5 average F1. It is the anchor for
everything else. If it does not reproduce, the adapter is wrong — fix that before drawing
any conclusion from a delta.

### 3. Retrieval ablations

Each isolates one part of the mechanism. Run on a subset if compute is tight.

| Ablation | Config change | Question it answers |
| --- | --- | --- |
| No step-level stage | `retrieval.step_level: false` | Does stage 2 (reasoning-style shift) matter, or is question-level retrieval enough? |
| Unlabelled references | `prompt.include_reference_labels: false` | Is the gain from the *examples* or from the *labels*? |
| More references | `retrieval.top_k_steps: 4` | Does more context help or just add noise? |
| Random references | see note below | Is it retrieval, or just having any extra text? |

The random-reference control is the strongest of the four and worth the compute: replace
the ranking with a random draw from the pool. If random references help as much as similar
ones, the effect is prompt length, not retrieval.

## Quantisation

On an 8 GB GPU use **`prm.backend: int4`** (torchao). It is not a compromise for its own
sake — it is what makes the run finish:

Measured on real ProcessBench data, 24 solutions per condition across all four subsets:

| | bf16 + disk offload | int4 |
| --- | ---: | ---: |
| weights in VRAM | ~8 GB of 15.2 GB | all ~6.3 GB |
| Condition A | ~100 s/solution | **5.4 s/solution** |
| Condition B | — | **28.3 s/solution** |
| full A+B run | ~190 h | **~32 h** |

The 190 h figure was never about compute; it was disk streaming. Removing the offload is
worth ~6× end-to-end and costs only precision on the *weights* — while `lm_head`, which
produces the `<+>`/`<->` logits the verdict reads, stays in bf16.

### Where Condition B's time goes

Profiled at 20 steps: retrieval **71 ms/step (1.2%)**, model forward **5,771 ms/step
(98.8%)**, mean prompt 841 tokens (max 1,225). Two consequences:

- Optimising the retriever is pointless. Optimising *prompt length* is not.
- int4 is weight-only: it dequantises on every forward, which costs compute. Our workload
  is pure prefill (one pass over the prompt, no generation), so int4 buys memory, not
  arithmetic. It is still the right choice here only because it eliminates disk offload.

Cheap levers on prompt length, in order of value per unit of risk:

| Lever | Effect | Cost to the science |
| --- | --- | --- |
| `--limit 400` per subset | 32 h → ~15 h | none — still powered for 6–13 pt effects |
| `prompt.max_reference_chars: 300` | shorter prompts | changes what B sees; ablate, don't assume |
| `retrieval.top_k_steps: 1` | ~1 fewer reference block | this is a research variable, so report it |

Prefer cutting the sample over cutting the method: `--limit` leaves the comparison intact,
whereas shrinking references changes the treatment being tested.

Rules for reporting quantised results:

- **Absolute F1 will not match the published 69.5.** Never present it as if it does.
- **Deltas remain meaningful** only when A and B use the identical backend and precision.
  An int4 number and a bf16 number are not comparable — not even loosely.
- State the backend beside every number in the report.

**Recommended plan.** Run the full A/B at int4 on the laptop (~6 h), which answers the
research question, since the question is about a *delta*. If a 16 GB+ GPU is reachable,
additionally run Condition A alone at bf16 to show your baseline reproduces the paper's
69.5 — that anchors the int4 numbers without needing the whole study rerun.

Sanity check worth 20 minutes: run `--limit 25` at both int4 and bf16 (offloaded) and
compare Condition A's per-step verdicts. If quantisation is flipping a large fraction of
verdicts, raise `int4_group_size` accuracy by lowering it to 32, or fall back to bf16.

## Suggested run order

```bash
python scripts/smoke.py                                        # plumbing
python scripts/verify_model_interface.py                       # prompt parity
python scripts/verify_model_interface.py --load-model          # real forward pass
python scripts/run_eval.py --config configs/baseline.yaml  --limit 25 --name pilot-a
python scripts/run_eval.py --config configs/retrieval.yaml --limit 25 --name pilot-b
python scripts/compare_runs.py --a runs/pilot-a --b runs/pilot-b
# only then, the full runs
```

## Reading the result honestly

The project's prediction is a *pattern*, not a number: Δ F1 should grow with OOD severity
(gsm8k → math → olympiadbench → omnimath). `compare_runs.py` reports the slope, the
Spearman correlation over severity ranks, and whether the deltas are monotone — and
refuses to call a flat delta vector a trend.

Four outcomes, all reportable:

| Outcome | What to write |
| --- | --- |
| Δ grows with difficulty, CIs exclude 0 | The mechanism transfers. The headline result. |
| Δ positive but flat across subsets | Retrieval helps, but *not* by fixing distribution shift — a different story from RetrievalPRM's. Say so. |
| Δ ≈ 0 | Retrieval's benefit does not transfer from binary grading to error typing. A clean negative result; the pitch already commits to reporting it. |
| Δ negative | References distract the classifier. Check prompt length effects and the random-reference control before concluding. |

Two guards against fooling yourself:

- A per-subset CI that includes 0 means "no detectable difference", not "a small gain".
- McNemar with fewer than ~25 discordant solutions is not trustworthy; `comparison.json`
  carries a `reliable` flag for exactly this reason.
