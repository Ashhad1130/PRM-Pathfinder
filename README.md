# Retrieval-Augmented PathFinder-PRM

**From Similar Examples to Error Types: Extending PathFinder-PRM with Retrieval-Augmented Verification**

Seminar project · Institute for Computational Linguistics, Heidelberg University
Pratik Goyal & Ashhad Raza Quadri · Supervisor: Lei Tang

---

## The question

Two papers improve step-level math graders in unrelated ways. **RetrievalPRM**
(Zhu et al. 2025, arXiv:2502.14361) shows the grader similar solved problems before it
judges a step — an open-book exam. **PathFinder-PRM** (Pala et al. 2025, arXiv:2505.19706)
replaces the binary correct/wrong verdict with a hierarchical one: first *what kind* of
error, then how good the step is. Nobody had combined them.

> Does inserting retrieval in front of PathFinder-PRM's error-typing stage make it better at
> identifying errors, especially on out-of-distribution problems — and does any gain survive
> a length-matched control?

No training, no new data. Inference only, on the released 7B checkpoint, with the model's
prompt contract held byte-identical across conditions.

## The answer

**No. Retrieval buys nothing here, and most of the damage it appears to cause comes from
something else entirely.**

| | references shown | average F1 | vs A |
| --- | --- | ---: | ---: |
| **A** baseline | none | **67.3** | |
| **D** ablation | retrieved, labels stripped | 60.9 | −6.5 |
| **B** treatment | retrieved, labels as `<+>` / `<->` | 46.3 | −21.0 |
| **C** control | **random**, labels as `<+>` / `<->` | 44.3 | −23.0 |

Three readings, in the order they matter:

**Relevance does nothing.** Random references cost as much as retrieved ones. C→B is +2.0
with a 95% interval of [−5.5, +9.1], and −0.2 once pool-contaminated eval items are dropped
from both arms. The pooled test has 48 discordant solutions splitting 26/22, so this is a
measured null rather than a sample too small to see an effect.

**Most of the loss is the labels, not the examples.** Each reference in Condition B ends
with `Teacher's judgement: Math reasoning: <->, Consistency: <->`. Those two tokens are what
the scoring rule compares at the mask positions to produce a verdict, and Condition B puts
four of them in the user turn on every query. Removing that one line recovers 14.6 points
(D→B, 95% CI [−22.0, −7.1], p < 0.0001).

**The failure has a shape.** References move the grader's decision point about a step
earlier: false alarms on clean solutions rise from 16.4% to 41.1%, exact-index accuracy
collapses because the blame lands before the real break, and the only figure that improves
is the miss rate. Condition D barely moves any of it.

![Conditions A, D, B and C by subset](docs/figures/arms.png)

Condition A lands at 67.3 against the paper's published 69.5, which is the anchor that makes
the rest worth reading. Measured at int4 on 50 solutions per subset; deltas between arms are
valid because all arms ran at identical precision, absolute values are not comparable to
bf16.

**Full numbers, intervals, significance tests and limitations: [`docs/RESULTS.md`](docs/RESULTS.md).**
The artefacts behind them: [`results/`](results/).

## How the comparison is kept honest

The finding rests on four arms differing by one thing each, so the machinery that enforces
that is the important part of this repository.

| Guard | What it prevents |
| --- | --- |
| `scripts/check_parity.py` | Two arms differing anywhere but their one defining key. An ablation must *declare* the key it varies. |
| Stratified sampling | ProcessBench lists every erroneous solution first, so `--limit` draws a seeded, nested sample instead of slicing an all-error prefix. |
| Contamination guard + exclusion | 59.6% of MATH appears verbatim in the retrieval pool. The guard filters neighbours; the exclusion recomputes every table without the affected eval items. |
| Undefined ≠ zero | An absent population makes F1 undefined, and the metric says so instead of scoring it 0. This is what caught the sampling bug. |
| Pooled contrasts | Per-subset McNemar runs out of discordant solutions at this scale; the pooled test does not. |
| Frozen prompt contract | `<extra>` may never reach the user turn, the assistant turn must carry exactly two mask positions, and `scripts/verify_model_interface.py` re-checks both against the model card. |

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows; source .venv/bin/activate elsewhere
pip install -r requirements.txt
pip install -e .
```

**Check the plumbing** — no GPU, no downloads, about five seconds:

```bash
python scripts/smoke.py
pytest
```

`smoke.py` runs every stage against a mock PRM and bundled fixtures, and asserts that the
control arm really is one: same reference count as the treatment, far lower similarity. Its
numbers are meaningless by construction; what it proves is that the pipeline is intact.

**Check the model contract** before trusting any real number:

```bash
python scripts/verify_model_interface.py
```

## Reproducing the study

```bash
python scripts/build_pool.py          --config configs/retrieval.yaml
python scripts/build_index.py         --config configs/retrieval.yaml
python scripts/check_contamination.py --config configs/retrieval.yaml   # read this first

python scripts/run_eval.py --config configs/pilot-int4.yaml           --limit 50  # A
python scripts/run_eval.py --config configs/pilot-int4-retrieval.yaml --limit 50  # B
python scripts/run_eval.py --config configs/pilot-int4-random.yaml    --limit 50  # C
python scripts/run_eval.py --config configs/pilot-int4-nolabels.yaml  --limit 50  # D

python scripts/compare_runs.py --a runs/int4-a --b runs/int4-b \
    --out runs/comparison --exclude-contaminated runs/contamination.json
python scripts/compare_runs.py --a runs/int4-c --b runs/int4-b --out runs/comparison-c-vs-b
python scripts/compare_runs.py --a runs/int4-d --b runs/int4-b --out runs/comparison-d-vs-b

python scripts/plot_arms.py --out runs/comparison/arms.png \
    --runs "A=runs/int4-a" "D=runs/int4-d" "B=runs/int4-b" "C=runs/int4-c"
python scripts/error_analysis.py --runs A=runs/int4-a D=runs/int4-d B=runs/int4-b C=runs/int4-c
```

Runs are resumable: every solution is flushed to `predictions.jsonl` as it finishes, and
`--resume` skips what is already there, so an interrupted run costs at most one solution.

The `configs/pilot-int4-*.yaml` set is what produced the reported numbers on an 8 GB laptop.
On a 24 GB or larger GPU use `configs/baseline.yaml`, `retrieval.yaml`,
`control-random.yaml`, `ablation-nolabels.yaml` and `ablation-wordlabels.yaml` instead —
same design at bf16, where absolute F1 becomes comparable to the published 69.5. Compare
arms only within one precision; never an int4 number against a bf16 one.

## Hardware

Measured on an RTX 5070 Laptop, 8 GB VRAM. A 7B in bf16 needs ~15.2 GB of weights, so the
reported run uses int4 weight-only quantisation via torchao.

| Backend | `prm.backend` | Needs | Notes |
| --- | --- | --- | --- |
| int4 (torchao) | `int4` | ~6.3 GB VRAM | what produced these results; `lm_head` stays in bf16 |
| bf16 | `hf` | 20 GB+ VRAM | for numbers comparable to the paper |
| mock | `mock` | nothing | development only, numbers meaningless |
| fp16 + offload | `hf` + `max_memory` | any VRAM + disk | ~12 s per forward pass; correctness pilots only |

`int4_packing_format: tile_packed_to_4d` is the kernel that works on Blackwell (sm_120).
`lm_head` and the embeddings stay in bf16 deliberately: the verdict *is* a comparison of the
`<+>` and `<->` logits from that head, so quantising it would inject noise straight into the
measured quantity.

Cost of the reported run, 200 solutions per arm: A 55 min, C 2 h 38 m, B 3 h 07 m,
D 4 h 59 m. The reference-carrying arms are slower because their prompts are two to three
times longer; retrieval itself is about 1% of step time.

## Layout

```
configs/      one YAML per arm; int4 pilots and bf16 equivalents
src/rapfprm/
  data/       ProcessBench loading and stratified sampling, retrieval-pool construction
  retrieval/  SBERT encoder, PCA + cosine index, two-stage retriever, random control
  prm/        PRM backends and prompt builders (the frozen contract lives here)
  eval/       the scoring runner and the official ProcessBench metric
  analysis/   contrasts, pooled tests, verdict profiling, figures
scripts/      command-line entry points
tests/        136 tests, including an end-to-end run on fixtures
results/      the committed artefacts behind docs/RESULTS.md
docs/         protocol, results, model-interface notes
```

## Documentation

- [`docs/RESULTS.md`](docs/RESULTS.md) — what was found, with intervals and limitations
- [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) — the protocol: arms, controls, sampling, what must be reported
- [`docs/MODEL_INTERFACE.md`](docs/MODEL_INTERFACE.md) — the frozen PathFinder-PRM contract and why it must not drift
- [`results/README.md`](results/README.md) — what each committed artefact is

## Open questions

- The token-level mechanism is a hypothesis. The labels cost 14.6 points; whether the
  `<+>` / `<->` tokens specifically are the cause, rather than the extra line or the
  judgement it carries, needs the word-rendered arm (`ablation-wordlabels.yaml`, unrun).
- No bf16 arm. Every number here is int4.
- 50 solutions per subset. The pooled contrasts are adequately powered; the per-subset rows
  show shape, not significance.
- The protocol's sensitivity checks — guard at 0.85 and 0.99, `top_k_steps: 4` — have not
  been run.

## References

- Zhu et al. (2025). *Retrieval-Augmented Process Reward Model*. arXiv:2502.14361
- Pala et al. (2025). *Error Typing for Smarter Rewards*. arXiv:2505.19706
- Zheng et al. (2024). *ProcessBench*. `Qwen/ProcessBench`
- Model: `declare-lab/PathFinder-PRM-7B` · Pool: `declare-lab/PathFinder-600K`
