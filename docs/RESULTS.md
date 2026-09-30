# Results

RTX 5070 Laptop (8 GB), PathFinder-PRM-7B quantised to int4 with torchao. Five conditions,
50 solutions per ProcessBench subset, 200 solutions per arm, the same 200 solutions in every
arm. The written-up version, with discussion, is `report/main.pdf`.

**Retrieval does not help PathFinder-PRM, and most of what goes wrong is the form of the
references' labels.** Retrieved references with gold labels cost 17.3 F1 points. Writing the
same labels as words instead of the model's own `<+>` / `<->` verdict tokens recovers 12.9 of
them, and is indistinguishable from showing no label at all. Relevance helps a little, mainly
on the two hardest subsets, but never enough to offset the loss.

| | F1 | against A |
| --- | ---: | ---: |
| **A** no references | **67.3** | |
| **E** references, labels as words | 62.9 | −4.4 |
| **D** references, labels stripped | 60.9 | −6.5 |
| **B** references, labels as `<+>` / `<->` | 50.0 | −17.3 |
| **C** random references, labels as `<+>` / `<->` | 44.3 | −23.0 |

---

## A correction first

The first run of B scored 46.3 and did **not** use Sentence-BERT. A Windows Application
Control policy blocked a pyarrow DLL, `sentence-transformers` failed to import, and the code
fell back to `MeanPoolingEncoder`. That fallback skipped Sentence-BERT's final Normalize
step: its vectors had the right direction but a norm of ~5.8, and the index's PCA (fitted on
unit vectors) projected them to the wrong place. Retrieval returned poor neighbours, in one
checked case the ones ranked 760th and 3,333rd.

It was caught because E, run later with the real encoder, did not receive the same references
as B. The fallback is fixed (`src/rapfprm/retrieval/encoder.py`, now within 4e-8 of
Sentence-BERT) with a regression test, and B was re-run. C also ran under the fallback, but
it draws references at random; recomputing them with the corrected encoder reproduces all 732
of its steps exactly. D and E used the real encoder throughout. B, D and E now receive
identical references on every step they share.

The degraded run is kept in `results/int4-b-fallback-encoder/`. Every B number below is from
the corrected run.

## The arms

| | References (2 per step) | Labels | Config |
| --- | --- | --- | --- |
| **A** | none | — | `pilot-int4.yaml` |
| **B** | retrieved, ranked | `<+>` / `<->` | `pilot-int4-retrieval.yaml` |
| **C** | random from the pool | `<+>` / `<->` | `pilot-int4-random.yaml` |
| **D** | retrieved, ranked | none | `pilot-int4-nolabels.yaml` |
| **E** | retrieved, ranked | *correct* / *incorrect* | `pilot-int4-wordlabels.yaml` |

Everything else is held fixed: same weights, same int4 quantisation, same 0.5 threshold,
same 4096-token budget, same contamination guard, same solutions in the same order. The
parity checker refuses to start a pair that differs anywhere else, and an ablation has to
declare the key it varies.

## Per subset

| Subset | A | E | D | B | C |
| --- | ---: | ---: | ---: | ---: | ---: |
| gsm8k | 70.0 | 69.0 | 65.7 | 55.1 | 54.1 |
| math | 72.7 | 66.5 | 61.6 | 50.3 | 54.3 |
| olympiadbench | 69.8 | 63.7 | 63.7 | 49.0 | 34.5 |
| omnimath | 56.8 | 52.5 | 52.5 | 45.7 | 34.2 |
| **average** | **67.3** | **62.9** | **60.9** | **50.0** | **44.3** |

Condition A reproduces the published baseline: 67.3 at int4 on 200 solutions against the
paper's 69.5 at bf16 on 3,400, with omnimath weakest as expected.

## Contrasts

Paired bootstrap interval on the macro-average delta; McNemar pooled over all 200 solutions
(discordant as second-arm-only / first-arm-only). "Clean" drops the 27 pool-contaminated
solutions from both arms (n = 173).

| Contrast | Δ F1 | 95% CI | discordant | p | Δ clean | clean CI |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| A→B | −17.3 | [−25.2, −10.1] | 8/43 | < 0.0001 | −18.8 | [−27.9, −9.9] |
| A→C | −23.0 | [−31.2, −15.4] | 7/54 | < 0.0001 | −21.1 | [−31.0, −13.1] |
| **C→B** relevance | +5.8 | [−1.1, +12.8] | 29/17 | 0.10 | +2.3 | [−6.2, +11.2] |
| A→D | −6.5 | [−12.8, −1.5] | 5/16 † | 0.03 | −5.7 | [−13.0, +0.3] |
| **D→B** labels | −10.8 | [−17.8, −4.7] | 8/32 | 0.0003 | −13.1 | [−21.9, −5.3] |
| A→E | −4.4 | [−10.0, +0.6] | 4/12 † | 0.08 | −1.7 | [−7.0, +3.7] |
| D→E | +2.0 | [−2.2, +6.4] | 9/6 † | 0.61 | +4.0 | [−1.9, +10.6] |
| **E→B** tokens | −12.9 | [−19.0, −7.7] | 6/33 | < 0.0001 | −17.0 | [−25.1, −10.7] |

† fewer than 25 discordant solutions; the McNemar p is not reliable.

**Relevance (C→B).** Small and not significant on average. Per subset it is +1.0 on gsm8k,
−4.0 on math, **+14.4** on olympiadbench ([+1.2, +29.0]) and **+11.6** on omnimath
([+0.0, +27.1]): the hardest, uncontaminated subsets, which is the shape RetrievalPRM reported.
Two per-subset intervals that barely clear zero are suggestive, not evidence.

**Labels (D→B, E→B, D→E).** The judgement costs nothing measurable when written in words
(D→E +2.0). Written as the model's own verdict tokens it costs 11 to 13 points. The residual
cost of reference text itself (A→D, A→E) is a few points on thin evidence.

**Difficulty.** A→B is −14.9, −22.4, −20.8, −11.1 from gsm8k to omnimath: a loss on every
subset with no trend in either direction.

## Where it goes wrong: stage 2, not the error typing

Over the 200 shared solutions (127 with an error, 73 clean):

| | A | D | E | B | C |
| --- | ---: | ---: | ---: | ---: | ---: |
| false alarms (clean flagged) | 16.4% | 26.0% | 23.3% | 34.2% | 39.7% |
| misses | 10.2% | 11.0% | 13.4% | 9.4% | 7.9% |
| exact index, of those flagged | 65.8% | 62.8% | 65.5% | 46.1% | 38.5% |
| flagged too early | 12.3% | 15.9% | 13.6% | 38.3% | 49.6% |
| mean index offset | +0.32 | +0.12 | +0.24 | −0.63 | −1.03 |
| mean steps scored | 4.92 | 4.62 | 4.79 | 3.95 | 3.66 |

Step level, over the 629 steps scored in every arm (548 of them should not be flagged),
bootstrapping whole solutions:

| A→ | stage-1 false flags | stage-2 false flags |
| --- | --- | --- |
| D | +0.4 [−0.5, +1.3] | +1.8 [+0.7, +3.2] |
| E | +0.0 [−0.9, +0.9] | +0.7 [+0.0, +1.7] |
| B | +0.5 [−0.6, +1.8] | **+7.7 [+5.4, +10.5]** |
| C | +0.4 [−0.3, +1.1] | **+11.1 [+8.4, +14.5]** |

The math / consistency typing does not measurably move in any arm. The verdict tokens shift
the stage-2 `P(<+>)` distribution down (mean 0.92 in A, 0.79 in B; 8.5% of steps below 0.5
against 0.3%), which flags correct steps early. Of the solutions B flags, 57 of 140 are
flagged by the correctness score alone (A: 6 of 126, E: 11 of 127). This is not majority-label
copying: 60% of pool steps are labelled `<+>`, yet the shift is towards `<->`.

Within B, binning solutions by mean reference similarity (0.39 / 0.48 / 0.56) gives losses of
20.9 / 19.7 / 11.9 points against A; the best-matched bin loses least but is also where A is
weakest, so it has less to lose.

## What this does not show

- **Nothing here transfers to bf16.** Every number is int4. Deltas between arms are valid;
  absolute F1 is not comparable to the paper's 69.5.
- **Error-type accuracy is unmeasured.** ProcessBench annotates the first error's index, not
  its type, so stage-1 statements are about how often it flags, not whether it types correctly.
- **Per-subset statistics are thin.** Intervals are ±15 to ±20 points. Quote the pooled
  contrasts; per-subset rows, including the relevance pattern, show shape.
- **D and E were designed after the first results**, so the label findings are exploratory.
- **One retrieval configuration.** Two references per step, MiniLM, PCA to 128, a 50K pool,
  guard at 0.95. The sensitivity checks (guard 0.85 / 0.99, `top_k_steps: 4`) were not run.

## Cost

| Arm | Wall time | Per solution |
| --- | ---: | ---: |
| A | 55 min | 16.6 s |
| C | 2 h 38 min | 47.4 s |
| B | 3 h 01 min | ~54 s |
| D | 4 h 59 min | 89.7 s |
| E | ~5 h 35 min | ~100 s |

B and E were interrupted (memory pressure, one CUDA out-of-memory on a long omnimath
solution) and resumed; resuming skips finished solutions and does not change them. On 8 GB of
VRAM the longest omnimath solutions spill into shared system memory and slow to ~24 s per
step. A full-benchmark run of all five arms would take roughly 290 hours on this machine.

## Reproducing

See the README. After the five runs are in `results/`, `python report/collect_numbers.py`
regenerates every contrast, the verdict profile, both figures and the step-level numbers.
