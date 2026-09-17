# Results

Run on 17 September 2026, RTX 5070 Laptop (8 GB), PathFinder-PRM-7B quantised to int4 with
torchao. Four conditions, 50 solutions per ProcessBench subset, 200 solutions per arm, the
same 200 solutions in every arm.

**Retrieval does not help PathFinder-PRM's error typing. Putting reference examples in the
prompt at all costs about 21 F1 points, and whether those references are relevant makes no
measurable difference.**

The second half of that sentence is the part that needed a control, and it is the part that
would have been missed without one. Condition B on its own looks like evidence that
retrieval is actively harmful. It isn't. Random references hurt just as much.

---

## The arms

| | Retrieval | References | Config | Run |
| --- | --- | --- | --- | --- |
| **A** baseline | off | none | `pilot-int4.yaml` | `runs/int4-a` |
| **B** retrieval | on | 2 per step, ranked by similarity | `pilot-int4-retrieval.yaml` | `runs/int4-b` |
| **C** control | on | 2 per step, drawn uniformly at random | `pilot-int4-random.yaml` | `runs/int4-c` |
| **D** ablation | on | 2 per step, ranked, gold labels stripped | `pilot-int4-nolabels.yaml` | `runs/int4-d` |

Everything else is held fixed: same weights, same int4 quantisation, same 0.5 threshold,
same 4096-token budget, same contamination guard, same solutions in the same order. The
parity checker refuses to start a pair that differs anywhere else, and an ablation has to
declare the key it varies (`--allow prompt.include_reference_labels`).

The sample is stratified, seeded and drawn — not sliced. ProcessBench lists every erroneous
solution before the first clean one, so a prefix would have given 50 error cases and 0 clean
ones per subset, and F1 over that is undefined. See
[EXPERIMENTS.md](EXPERIMENTS.md#sampling---limit-draws-it-does-not-slice).

## Condition A reproduces the published baseline

| Subset | n | error / clean | error_acc | correct_acc | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| gsm8k | 50 | 26 / 24 | 53.8 | 100.0 | 70.0 |
| math | 50 | 30 / 20 | 66.7 | 80.0 | 72.7 |
| olympiadbench | 50 | 33 / 17 | 60.6 | 82.4 | 69.8 |
| omnimath | 50 | 38 / 12 | 55.3 | 58.3 | 56.8 |
| **average** | | | | | **67.3** |

The paper reports 69.5 at bf16 over the full 3,400 solutions. We get 67.3 at int4 over 200.
Two points of that gap are unsurprising: quantisation perturbs the `<+>` / `<->` logits the
verdict is read from, and a 50-solution subset has a standard error of several points on its
own. The ordering across subsets also comes out as expected, with omnimath weakest.

This is the anchor. Without it there would be no reason to believe anything downstream.

## Adding references costs about 21 points

![Condition A, C and B by subset](figures/arms.png)

| Subset | A | B | Δ | 95% CI on Δ | McNemar p | discordant |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| gsm8k | 70.0 | 54.1 | −15.9 | [−35.1, +4.0] | 0.039 | 15 |
| math | 72.7 | 51.6 | −21.1 | [−38.0, −6.2] | 0.034 | 18 |
| olympiadbench | 69.8 | 49.3 | −20.5 | [−35.3, −8.9] | 0.009 | 12 |
| omnimath | 56.8 | 30.2 | −26.6 | [−42.6, −12.3] | 0.001 | 14 |
| **average** | **67.3** | **46.3** | **−21.0** | | | |

Three of four per-subset intervals exclude zero. Pooled over all 200 paired solutions the
average delta is −21.0 with a 95% interval of [−29.5, −13.0], and the paired test is
unambiguous: 59 solutions where exactly one arm was right, 51 of them broken by B against 8
fixed, p < 0.0001. That pooled test clears the 25-discordant bar the project set for itself;
the per-subset ones (12 to 18) do not, which is why they are quoted here as description
rather than evidence.

Both halves of the metric fall in every subset — the model finds fewer of the real errors
*and* flags more clean solutions — so this is not a threshold sitting in the wrong place.

The retrieval itself worked. Two references per query, mean question similarity 0.618, mean
step similarity 0.502, 77 exact duplicates and 6 near-duplicates dropped by the guard, no
empty results, and no reference discarded for length. Whatever went wrong, it was not the
retriever.

## The control: relevance is not what matters

| Subset | A | C (random) | B (retrieved) | A→C | C→B |
| --- | ---: | ---: | ---: | ---: | ---: |
| gsm8k | 70.0 | 54.1 | 54.1 | −15.9 | +0.0 |
| math | 72.7 | 54.3 | 51.6 | −18.4 | −2.7 |
| olympiadbench | 69.8 | 34.5 | 49.3 | −35.3 | +14.8 |
| omnimath | 56.8 | 34.2 | 30.2 | −22.6 | −4.0 |
| **average** | **67.3** | **44.3** | **46.3** | **−23.0** | **+2.0** |

Random references cost 23 points. Ranked ones give back 2.0 on average, and no subset's
interval excludes zero (p = 0.68, 1.00, 0.10, 0.80). The +14.8 on olympiadbench is the only
large value in the C→B column and it sits inside a [−0.0, +29.7] interval with 13 discordant
solutions, which at this sample size is what noise looks like.

Pooled, the relevance contrast is as flat as it looks: average delta +2.0, interval
[−5.5, +9.1], and 48 discordant solutions split almost evenly — 26 that retrieval fixed
against 22 that it broke, p = 0.67. This is the one place where pooling does more than
tidy up the presentation. With 48 discordant pairs the test is adequately powered by the
project's own standard, so "no detectable effect" here means the effect was measured and
came out near zero, not that the sample was too small to see it.

That the control really was a control is visible in its own counters: C drew the same 2.00
references per query as B, with mean step similarity 0.002 against B's 0.502. The two arms
differ in relevance and in nothing else that we can measure.

## How the verdicts change: the model pulls the trigger sooner

F1 falling by 21 points could mean the grader stopped finding errors, started inventing
them, or kept finding them in the wrong place. It is the second and third, and the traces
say so plainly. Over the 200 shared solutions (127 with an error, 73 clean):

| | A | C (random) | B (retrieved) |
| --- | ---: | ---: | ---: |
| false alarms — clean solutions flagged | 16.4% | 39.7% | 41.1% |
| misses — errors waved through | 10.2% | 7.9% | 7.1% |
| exact index, of those flagged | 65.8% | 38.5% | 42.4% |
| flagged too early | 12.3% | 49.6% | 41.5% |
| flagged too late | 21.9% | 12.0% | 16.1% |
| mean index offset (predicted − gold) | +0.32 | −1.03 | −0.90 |
| mean steps scored per solution | 4.92 | 3.66 | 3.67 |

References make the model quicker to condemn a step. Its mean blame lands almost a full step
earlier than the truth, against a third of a step late for the baseline, and with `early_stop`
on it halts after 3.7 steps instead of 4.9. Everything else follows from that. False alarms
on clean solutions go up by a factor of two and a half. Exact-index accuracy collapses
because the model blames a step before the one that actually breaks. The one number that
improves is the miss rate, from 10.2% to 7.1%, which is what a lower effective threshold
should do.

So this is not the model becoming worse at mathematics. It is the same model with its
decision point moved, and the harmonic mean of ProcessBench punishes that hard: a grader
that flags everything scores zero.

Here too C and B are the same arm to within noise — 39.7% against 41.1% false alarms,
−1.03 against −0.90 index offset. Whatever moves the decision point, it is not the content
of the references.

### Within Condition B, better retrieval does not mean less damage

The control answers this between arms. The traces answer it again inside the treatment arm,
for free: each step records the similarity of the references it was given, so B's solutions
can be split by how good their retrieval actually was and compared against A on those same
solutions.

| bin | n | mean reference similarity | A | B | Δ |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 67 | 0.401 | 65.7% | 46.3% | −19.4 |
| 2 | 66 | 0.527 | 66.7% | 40.9% | −25.8 |
| 3 | 67 | 0.637 | 71.6% | 52.2% | −19.4 |

(Solution-level accuracy, exact first-error index.) The solutions with the best retrieval
lose as much as the solutions with the worst. Retrieval quality varies by more than half a
standard deviation across these bins and the damage does not follow it.

Reproduce with:

```bash
python scripts/error_analysis.py --runs A=runs/int4-a C=runs/int4-c B=runs/int4-b     --similarity-bins runs/int4-b runs/int4-a
```

## With contaminated items removed, relevance buys nothing at all

59.6% of the MATH subset appears verbatim in PathFinder-600K. The retrieval guard filters
those questions' *neighbours*, but the questions themselves are still graded, so MATH is
being treated differently from the clean subsets. Dropping all 633 flagged solutions from
both arms leaves 173 of the 200 (math falls to 24):

| Contrast | full sample | contaminated items removed |
| --- | ---: | ---: |
| A→B | −21.0 | −21.3 |
| A→C | −23.0 | −21.1 |
| **C→B** | **+2.0** | **−0.2** |

The +2.0 was coming from the contaminated half of MATH. On clean data the relevance effect
is −0.2 points with an interval of [−10.2, +9.1] and a pooled p of 0.88 over 42 discordant
solutions. The headline does not move: adding references still costs about 21 points either
way.

## The difficulty trend runs backwards

RetrievalPRM's case rested on gains that grew with problem difficulty, and the project was
written to look for the same signature. The signature is there, with the sign flipped:
−15.9, −21.1, −20.5, −26.6 across gsm8k → math → olympiadbench → omnimath, slope −3.1 F1 per
severity rank, Spearman −0.80.

The obvious reading is that the harder the problem, the more the model has to lose from a
distracted context. The honest reading is that four points and a rank correlation are not
much to build on, and the analysis code declines to call it a trend for that reason.

## What this does not show

- **Nothing here transfers to bf16.** Every number is int4. A-vs-B deltas are valid because
  both arms ran at identical precision; absolute F1 is not comparable to the paper's 69.5,
  even though A lands close to it.
- **Per-subset statistics are thin.** 50 solutions per subset gives intervals ±15 to ±20
  points wide, and each per-subset McNemar has only 11 to 20 discordant solutions. The
  pooled contrasts are the ones to quote; the per-subset rows show shape, not significance.
- **A null is not proof of absence.** The relevance contrast is bounded, not zeroed: the
  pooled interval runs [−10.2, +9.1] on clean data. That rules out retrieval buying the
  6-13 points the source papers report. It does not rule out two or three points, which
  would need roughly ten times this sample.
- **One retrieval configuration.** Two references per step, MiniLM embeddings, PCA to 128
  dimensions, a 50K-item pool, guard at cosine 0.95. A different k or a different pool might
  behave differently, though the size of the context penalty makes that look unpromising.

## Cost

| Arm | Solutions | Wall time | Per solution |
| --- | ---: | ---: | ---: |
| A | 200 | 55 min | 16.6 s |
| B | 200 | 3 h 07 min | 56.0 s |
| C | 200 | 2 h 38 min | 47.4 s |

No arm was resumed, so these are honest throughput figures. B and C cost three times what A
does for the same solutions, entirely because their prompts are longer; profiling elsewhere
puts the retrieval machinery itself at about 1% of step time. On 8 GB of VRAM the longest
omnimath solutions in B pushed allocation into shared system memory and slowed to roughly
24 s per step, which is worth knowing before anyone plans a full-benchmark run on a laptop.

## Reproducing

```bash
bash scripts/lightning/run_experiment.sh \
    --config-a configs/pilot-int4.yaml \
    --config-b configs/pilot-int4-retrieval.yaml \
    --config-c configs/pilot-int4-random.yaml \
    --limit 50 --stages a,b,c,compare

python scripts/run_eval.py --config configs/pilot-int4-nolabels.yaml --limit 50   # arm D
python scripts/plot_arms.py --runs "A=runs/int4-a" "C=runs/int4-c" "B=runs/int4-b" \
    --out runs/comparison/arms.png
```

Numbers, intervals and tests live in `runs/comparison/`, `runs/comparison-a-vs-control/` and
`runs/comparison-control-vs-b/`, each with an `uncontaminated-` twin.
