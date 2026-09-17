# Experiment protocol

Measured outcomes live in [RESULTS.md](RESULTS.md). This file is the protocol: what
gets run, what is held fixed, and what has to be reported alongside a number.

## The comparison

| | Condition A | Condition B | Condition C (control) |
| --- | --- | --- | --- |
| Model | PathFinder-PRM-7B, frozen | same weights | same weights |
| Prompt | model card, unmodified | + **retrieved** references, user turn only | + **randomly drawn** references, user turn only |
| References per step | 0 | `top_k_steps` | the same `top_k_steps` |
| Assistant turn | `... Math reasoning: <extra>, Consistency: <extra>` | byte-identical | byte-identical |
| Threshold | 0.5 | 0.5 | 0.5 |
| Data | ProcessBench, 4 subsets | identical solutions, identical order | identical solutions, identical order |
| Config | `baseline.yaml` | `retrieval.yaml` | `control-random.yaml` |

Exactly one thing varies between any two of them: `retrieval.enabled` for A vs B,
`retrieval.reference_mode` for B vs C. Keep it that way — if you change `prm.*` in one
config, change it in all of them, or the delta stops measuring retrieval. The Lightning
driver enforces this before it starts
(`scripts/lightning/_check_parity.py`); `tests/test_pipeline.py` enforces it in CI.

An ablation varies a key outside the retrieval block, so it has to *declare* that key:

```bash
python scripts/lightning/_check_parity.py configs/pilot-int4-retrieval.yaml \
    configs/pilot-int4-nolabels.yaml --allow prompt.include_reference_labels
```

Anything the ablation does not name is still refused. That keeps a declared ablation and an
accidental config drift from looking identical at the command line.

Condition C is what makes B interpretable: B's prompts are longer than A's, so without a
length-matched control, "retrieval helps" and "more text helps" are the same number. See
[Condition C](#4-condition-c--the-random-reference-control).

## Metric

ProcessBench asks for the index of the **first** erroneous step, or −1 for a clean
solution. Reported per subset:

- `error_acc` — exact-index accuracy over solutions that contain an error
- `correct_acc` — accuracy over clean solutions (i.e. correctly predicting −1)
- `F1` — the **harmonic mean** of those two

The harmonic mean is what makes the benchmark adversarial: a grader that flags every step
scores `error_acc = 1.0`, `correct_acc = 0.0`, F1 = 0. Published anchors, average F1 across
the four subsets: RetrievalPRM-7B **65.8**, PathFinder-PRM-7B **69.5**.

### Report the pooled contrast, not just the four subset rows

Per-subset McNemar tests run out of *discordant* solutions long before they run out of
solutions. At 50 per subset a contrast typically produces 10-20 disagreements, under the ~25
the normal approximation needs, so every per-subset p-value on a pilot is decoration.

`compare_runs.py` therefore also reports an `overall` block: a bootstrap interval for the
macro-average delta (resampled within subsets, since the average is a mean of four F1s) and
a McNemar test pooled over every paired solution. On the 200-solution run those pooled tests
carry 42-61 discordant solutions and clear the reliability bar, which is what allows the
relevance contrast to be reported as a measured null rather than as insufficient data.

### Sampling: `--limit` draws, it does not slice

ProcessBench lists **every erroneous solution before the first clean one**. A prefix is
therefore not a sample of the subset, it is the error half of it:

| | gsm8k | math | olympiadbench | omnimath |
| --- | ---: | ---: | ---: | ---: |
| full subset (error / clean) | 207 / 193 | 594 / 406 | 661 / 339 | 759 / 241 |
| first 100 rows | 100 / 0 | 100 / 0 | 100 / 0 | 100 / 0 |
| first 400 rows | 207 / 193 | 400 / 0 | 400 / 0 | 400 / 0 |

With no clean solutions, `correct_acc` is undefined and so is F1 — the metric says so
rather than scoring the missing population as zero, which is how this was caught after a
92-minute run produced four undefined subsets.

`data/processbench.py::subsample` therefore takes a **stratified** sample: both populations
are drawn at the subset's own rate, so `--limit 100` gives 52/48 on gsm8k and 76/24 on
omnimath, matching the real balance. Two properties make it safe for this experiment:

- **Seeded** (`data.sample_seed`, default 17) — every arm loading the same subset, limit and
  seed gets the identical solutions, which is what makes A, B and C paired. The parity
  checker treats `sample_seed` as a locked key for that reason.
- **Nested** — the limit-50 sample sits inside the limit-100 sample, so a run can be grown
  (or reported at a smaller size) and `--resume` reuses every solution already scored.

## Controls

### 1. Contamination guard (mandatory)

**This is not hypothetical. Measured on this pool (50K items, 15,316 unique questions):**

| Subset | n | verbatim in pool | near-duplicate (cos ≥ 0.95) |
| --- | ---: | ---: | ---: |
| gsm8k | 400 | 0 (0.0%) | 0 |
| **math** | 1000 | **596 (59.6%)** | **605** |
| olympiadbench | 1000 | 0 (0.0%) | 4 |
| omnimath | 1000 | 0 (0.0%) | 24 |
| **total** | 3400 | **596 (17.5%)** | 633 |

Every verbatim match is also a near-duplicate (cosine 1.0), so 633 is the union — the
number of eval solutions `--exclude-contaminated` removes. The near-dup column shifts by a
few items between encoder runs because those items sit on the 0.95 threshold itself; the
verbatim column is exact.

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

#### The guard filters neighbours, not eval items — so also report the filtered table

There is a second-order effect the guard cannot remove, and it has to be named in the
report. The guard drops a contaminated question's *neighbours*; the question itself is
still graded. So on MATH, where 59.6% of questions have a verbatim pool copy, Condition B
meets those items with its rank-2+ references while a clean OlympiadBench item gets its
true top-1. The two subsets are not receiving the same treatment, and the headline table
cannot show that.

The fix is to recompute the comparison with the flagged eval solutions removed from **both**
conditions. `check_contamination.py` records their uids, and `compare_runs.py` consumes them:

```bash
python scripts/check_contamination.py --config configs/retrieval.yaml   # writes flagged_uids
python scripts/compare_runs.py --a runs/baseline --b runs/retrieval \
    --exclude-contaminated runs/contamination.json
```

That writes a second set of artefacts alongside the first, prefixed `uncontaminated-`, with
the exclusion and the surviving `n` stated in the table header so the two can never be
confused. Report both: the full table is the benchmark as published, the filtered one is
the benchmark without the asymmetry. Where they disagree, the filtered table is the one
that supports a claim about the mechanism.

The Lightning driver passes the flag automatically whenever `runs/contamination.json`
exists, for every contrast including the control ones.

### 2. Baseline fidelity

Condition A at bf16 should land near the published 69.5 average F1. It is the anchor for
everything else. If it does not reproduce, the adapter is wrong — fix that before drawing
any conclusion from a delta.

### 3. Retrieval ablations

Each isolates one part of the mechanism. Run on a subset if compute is tight.

| Ablation | Config change | Question it answers |
| --- | --- | --- |
| No step-level stage | `retrieval.step_level: false` | Does stage 2 (reasoning-style shift) matter, or is question-level retrieval enough? |
| Unlabelled references | `configs/pilot-int4-nolabels.yaml` | Is the effect from the *examples* or from the `<+>`/`<->` verdict tokens the labels render? **Run; see RESULTS.md.** |
| More references | `retrieval.top_k_steps: 4` | Does more context help or just add noise? |
| Random references | `retrieval.reference_mode: random` | Is it retrieval, or just having any extra text? **Not an optional ablation — see below.** |

### 4. Condition C — the random-reference control

Condition B's prompts are ~841 tokens against Condition A's ~350. So a gain in B has two
possible causes, and the headline table cannot tell them apart:

1. the references are **relevant** — the mechanism the project claims;
2. the prompt is simply **longer** — more context, any context.

Condition C separates them. `configs/control-random.yaml` is Condition B with
`retrieval.reference_mode: random`: references are drawn uniformly from the pool instead of
ranked, and *everything else is identical* — the same number of references, the same
rendering, the same contamination guard, the same prompt budget. Only relevance is gone.

```bash
python scripts/run_eval.py --config configs/control-random.yaml     # Condition C
python scripts/compare_runs.py --a runs/baseline       --b runs/control-random  # A -> C
python scripts/compare_runs.py --a runs/control-random --b runs/retrieval       # C -> B
```

| Contrast | Isolates | Reading |
| --- | --- | --- |
| A → B | everything retrieval adds | the headline delta, and on its own uninterpretable |
| A → C | extra context alone | a gain here is about prompt length, not retrieval |
| **C → B** | **relevance, prompt length held fixed** | **this is the number the hypothesis actually rests on** |

If C → B is flat, the honest headline is that the gain was context, not retrieval — no
matter how good A → B looks. Report all three contrasts, always, and in that order.

**Check the control really was one.** `summary.json` records
`retrieval.mean_step_similarity` for both arms; C's must sit far below B's. If they are
close, the pool is small or homogeneous enough that a random draw is a similar draw, and
the control is not controlling anything. `scripts/smoke.py` asserts both halves of this
(same reference count, lower similarity) on fixture data before you spend GPU hours.

Cost: Condition C is a retrieval-length run, so budget it at roughly the same as B. On the
Lightning driver it is a default stage; `--no-control` skips it, and then the report has to
say that the headline gain has no control behind it.

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
python scripts/run_eval.py --config configs/baseline.yaml      --limit 25 --name pilot-a
python scripts/run_eval.py --config configs/retrieval.yaml     --limit 25 --name pilot-b
python scripts/run_eval.py --config configs/control-random.yaml --limit 25 --name pilot-c
python scripts/compare_runs.py --a runs/pilot-a --b runs/pilot-b
python scripts/compare_runs.py --a runs/pilot-c --b runs/pilot-b   # the relevance contrast
# only then, the full runs
```

On a Linux GPU box the whole sequence, all three arms included, is one command —
`bash scripts/lightning/run_experiment.sh` (see `docs/LIGHTNING.md`). It refuses to start
if the arms' configs differ in anything but `retrieval.enabled` / `reference_mode`.

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

Whichever row you land in, the claim is only as strong as the **C → B** contrast behind it
and the `uncontaminated-` table beside it. A positive A → B with a flat C → B is a result
about prompt length; a positive A → B that disappears once contaminated items are dropped
is a result about memorisation.

Two guards against fooling yourself:

- A per-subset CI that includes 0 means "no detectable difference", not "a small gain".
- McNemar with fewer than ~25 discordant solutions is not trustworthy; `comparison.json`
  carries a `reliable` flag for exactly this reason.
