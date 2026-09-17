# Results

Run on 17 September 2026, RTX 5070 Laptop (8 GB), PathFinder-PRM-7B quantised to int4 with
torchao. Four conditions, 50 solutions per ProcessBench subset, 200 solutions per arm, the
same 200 solutions in every arm.

**Retrieval does not help PathFinder-PRM's error typing, and most of what goes wrong is not
the retrieval at all.** Adding retrieved references costs 21 F1 points, but two thirds of
that comes from rendering each reference's gold judgement as the model's own `<+>` / `<->`
tokens inside the prompt. Strip the labels and the cost falls to about 6 points. Replace the
retrieved references with random ones and nothing changes at all.

| | F1 | against A |
| --- | ---: | ---: |
| **A** no references | **67.3** | |
| **D** references, labels stripped | 60.9 | −6.5 |
| **B** references with gold labels | 46.3 | −21.0 |
| **C** random references with gold labels | 44.3 | −23.0 |

Two comparisons carry the finding. **D → B is −14.6** (95% CI [−22.0, −7.1], pooled McNemar
p < 0.0001 over 50 discordant solutions): the same references in the same positions, labels
on versus off. **C → B is +2.0** ([−5.5, +9.1], p = 0.67), and −0.2 once pool-contaminated
eval items are dropped from both arms: relevance, with prompt length held fixed, does nothing
measurable.

---

## The arms

| | Retrieval | References | Config | Run |
| --- | --- | --- | --- | --- |
| **A** baseline | off | none | `pilot-int4.yaml` | `runs/int4-a` |
| **B** retrieval | on | 2 per step, ranked, with gold labels | `pilot-int4-retrieval.yaml` | `runs/int4-b` |
| **C** control | on | 2 per step, random, with gold labels | `pilot-int4-random.yaml` | `runs/int4-c` |
| **D** ablation | on | 2 per step, ranked, labels stripped | `pilot-int4-nolabels.yaml` | `runs/int4-d` |

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
own. The ordering across subsets comes out as expected, with omnimath weakest.

This is the anchor. Without it there would be no reason to believe anything downstream.

## The decomposition

![Conditions A, D, B and C by subset](figures/arms.png)

| Subset | A | D (no labels) | B (labels) | C (random) | A→D | D→B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gsm8k | 70.0 | 65.7 | 54.1 | 54.1 | −4.3 | −11.6 |
| math | 72.7 | 61.6 | 51.6 | 54.3 | −11.1 | −10.0 |
| olympiadbench | 69.8 | 63.7 | 49.3 | 34.5 | −6.2 | −14.4 |
| omnimath | 56.8 | 52.5 | 30.2 | 34.2 | −4.3 | −22.3 |
| **average** | **67.3** | **60.9** | **46.3** | **44.3** | **−6.5** | **−14.6** |

Read across a row and the 21-point drop pulls apart into two very different effects.

**Reference text costs about six points, and the evidence for even that is thin.** A→D is
−6.5 with an interval of [−12.8, −1.5] and 21 discordant solutions, under the 25 the paired
test needs. On contamination-cleaned data it is −5.7 with an interval of [−13.0, +0.3],
which straddles zero. Call it a small cost, not a demonstrated one.

**The gold labels cost fourteen, and that one is solid.** D→B is −14.6, interval
[−22.0, −7.1], 50 discordant solutions, p < 0.0001, and it holds at −15.6 after contamination
cleaning. The two arms differ in exactly one thing: whether each reference ends with a line
like

```
Teacher's judgement: Math reasoning: <->, Consistency: <-> (math and consistency error)
```

`<+>` and `<->` are not decoration. They are the two tokens whose logits the scoring rule
compares at the mask positions to produce the verdict. Condition B puts four of them into the
user turn on every query. The prompt builder already guards against `<extra>` leaking into
the user turn, since a stray mask token would add a phantom scoring position; the verdict
tokens were never treated as equally dangerous, and on this evidence they are.

## The control: relevance is not what matters

| Subset | A | C (random) | B (retrieved) | A→C | C→B |
| --- | ---: | ---: | ---: | ---: | ---: |
| gsm8k | 70.0 | 54.1 | 54.1 | −15.9 | +0.0 |
| math | 72.7 | 54.3 | 51.6 | −18.4 | −2.7 |
| olympiadbench | 69.8 | 34.5 | 49.3 | −35.3 | +14.8 |
| omnimath | 56.8 | 34.2 | 30.2 | −22.6 | −4.0 |
| **average** | **67.3** | **44.3** | **46.3** | **−23.0** | **+2.0** |

Random references cost 23 points, ranked ones 21. Pooled, the relevance contrast is +2.0 with
an interval of [−5.5, +9.1] and 48 discordant solutions splitting almost evenly — 26 that
retrieval fixed against 22 it broke, p = 0.67. With 48 discordant pairs the test is
adequately powered by the project's own standard, so this is an effect that was measured and
came out near zero, not one the sample was too small to see.

That the control really was a control is visible in its own counters: C drew the same 2.00
references per query as B, with mean step similarity 0.002 against B's 0.502.

The +14.8 on olympiadbench is the only large value in the C→B column, sits inside a
[−0.0, +29.7] interval, and does not survive the contamination-cleaned recomputation.

## With contaminated items removed, relevance buys nothing at all

59.6% of the MATH subset appears verbatim in PathFinder-600K. The retrieval guard filters
those questions' *neighbours*, but the questions themselves are still graded, so MATH is
treated differently from the clean subsets. Dropping all 633 flagged solutions from both arms
leaves 173 of the 200 (math falls to 24):

| Contrast | full sample | contaminated items removed |
| --- | ---: | ---: |
| A→B | −21.0 | −21.3 |
| A→C | −23.0 | −21.1 |
| A→D | −6.5 | −5.7 |
| D→B | −14.6 | −15.6 |
| **C→B** | **+2.0** | **−0.2** |

The +2.0 was coming from the contaminated half of MATH. On clean data the relevance effect is
−0.2 points, interval [−10.2, +9.1], pooled p = 0.88 over 42 discordant solutions.

## How the verdicts change: the model pulls the trigger sooner

F1 falling could mean the grader stopped finding errors, started inventing them, or kept
finding them in the wrong place. Over the 200 shared solutions (127 with an error, 73 clean):

| | A | D (no labels) | B (labels) | C (random) |
| --- | ---: | ---: | ---: | ---: |
| false alarms — clean solutions flagged | 16.4% | 26.0% | 41.1% | 39.7% |
| misses — errors waved through | 10.2% | 11.0% | 7.1% | 7.9% |
| exact index, of those flagged | 65.8% | 62.8% | 42.4% | 38.5% |
| flagged too early | 12.3% | 15.9% | 41.5% | 49.6% |
| flagged too late | 21.9% | 21.2% | 16.1% | 12.0% |
| mean index offset (predicted − gold) | +0.32 | +0.12 | −0.90 | −1.03 |
| mean steps scored per solution | 4.92 | 4.62 | 3.67 | 3.66 |

The labelled arms condemn steps about a step earlier than the truth, against a third of a step
*late* for the baseline, and with `early_stop` on they halt after 3.7 steps instead of 4.9.
Everything else follows: false alarms on clean solutions rise two and a half times,
exact-index accuracy collapses because the blame lands before the real break, and the only
figure that improves is the miss rate, which is what a lower effective threshold does.

Condition D barely moves any of this. Its index offset is +0.12 against A's +0.32, and it
scores 4.6 steps per solution against A's 4.9. The decision point does not shift because the
prompt got longer, or because other people's solutions are sitting in it. It shifts when the
model is shown its own verdict tokens.

The obvious hypothesis is that `<->` tokens in the reference block bias the comparison at the
mask positions directly. This run does not test that. Rendering the same judgements as plain
words, or restricting references to error-free examples, would separate "the token" from "the
judgement it encodes", and both are cheap to run.

### Within Condition B, better retrieval does not mean less damage

| bin | n | mean reference similarity | A | B | Δ |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 67 | 0.401 | 65.7% | 46.3% | −19.4 |
| 2 | 66 | 0.527 | 66.7% | 40.9% | −25.8 |
| 3 | 67 | 0.637 | 71.6% | 52.2% | −19.4 |

(Solution-level accuracy, exact first-error index.) The solutions with the best retrieval lose
as much as the solutions with the worst. Retrieval quality varies by more than half a standard
deviation across these bins and the damage does not follow it.

## The difficulty trend runs backwards

RetrievalPRM's case rested on gains that grew with problem difficulty, and this project was
written to look for the same signature. The signature is there with the sign flipped: A→B is
−15.9, −21.1, −20.5, −26.6 across gsm8k → math → olympiadbench → omnimath, slope −3.1 F1 per
severity rank, Spearman −0.80. The label effect follows the same shape (−11.6, −10.0, −14.4,
−22.3), which fits a grader with less slack to lose on harder problems.

Four points and a rank correlation are not much to build on, and the analysis code declines to
call it a trend for that reason.

## What this does not show

- **Nothing here transfers to bf16.** Every number is int4. Deltas between arms are valid
  because all arms ran at identical precision; absolute F1 is not comparable to the paper's
  69.5, even though A lands close to it.
- **The token-level mechanism is a hypothesis.** The labels cost 14.6 points. That is not the
  same as showing the `<+>` / `<->` tokens specifically are the cause, rather than the extra
  line of text or the judgement it expresses.
- **Per-subset statistics are thin.** 50 solutions per subset gives intervals ±15 to ±20
  points wide and 11 to 21 discordant solutions per test. The pooled contrasts are the ones to
  quote; per-subset rows show shape, not significance.
- **A null is not proof of absence.** The relevance contrast is bounded, not zeroed: the
  pooled interval runs [−10.2, +9.1] on clean data. That rules out retrieval buying the 6-13
  points the source papers report. It does not rule out two or three.
- **One retrieval configuration.** Two references per step, MiniLM embeddings, PCA to 128
  dimensions, a 50K-item pool, guard at cosine 0.95. The sensitivity checks the protocol
  recommends (guard at 0.85 and 0.99, `top_k_steps: 4`) have not been run.

## Cost

| Arm | Solutions | Wall time | Per solution |
| --- | ---: | ---: | ---: |
| A | 200 | 55 min | 16.6 s |
| C | 200 | 2 h 38 min | 47.4 s |
| B | 200 | 3 h 07 min | 56.0 s |
| D | 200 | 4 h 59 min | 89.7 s |

No arm was resumed, so these are honest throughput figures. D is the slowest even though its
prompts are shorter than B's, because it rejects fewer steps and `early_stop` therefore saves
it less work. On 8 GB of VRAM the longest omnimath solutions push allocation into shared
system memory and slow to roughly 24 s per step, which is worth knowing before planning a
full-benchmark run on a laptop.

## Reproducing

```bash
bash scripts/lightning/run_experiment.sh \
    --config-a configs/pilot-int4.yaml \
    --config-b configs/pilot-int4-retrieval.yaml \
    --config-c configs/pilot-int4-random.yaml \
    --limit 50 --stages a,b,c,compare

python scripts/run_eval.py --config configs/pilot-int4-nolabels.yaml --limit 50
python scripts/compare_runs.py --a runs/int4-a --b runs/int4-d \
    --out runs/comparison-a-vs-nolabels --exclude-contaminated runs/contamination.json
python scripts/compare_runs.py --a runs/int4-d --b runs/int4-b \
    --out runs/comparison-nolabels-vs-b --exclude-contaminated runs/contamination.json

python scripts/plot_arms.py --out runs/comparison/arms.png \
    --runs "A: no refs=runs/int4-a" "D: refs, no labels=runs/int4-d" \
           "B: refs + labels=runs/int4-b" "C: random refs + labels=runs/int4-c"
python scripts/error_analysis.py \
    --runs A=runs/int4-a D=runs/int4-d B=runs/int4-b C=runs/int4-c \
    --similarity-bins runs/int4-b runs/int4-a
```

Numbers, intervals and tests live in `runs/comparison*/`, each with an `uncontaminated-` twin.
