# Retrieval-Augmented PathFinder-PRM

**From Similar Examples to Error Types: Extending PathFinder-PRM with Retrieval-Augmented Verification**

Seminar project · Institute for Computational Linguistics, Heidelberg University
Pratik Goyal & Ashhad Raza Quadri · Supervisor: Lei Tang

---

## The one-paragraph version

Two papers improve step-level math graders (Process Reward Models) in unrelated ways.
**RetrievalPRM** (Zhu et al. 2025, arXiv:2502.14361) retrieves similar solved problems and
shows them to the grader before it judges a step — an open-book exam. **PathFinder-PRM**
(Pala et al. 2025, arXiv:2505.19706) replaces the binary correct/wrong verdict with a
hierarchical one: first *what kind* of error (math vs. consistency), then how good the step
is. Nobody has combined them. This project inserts RetrievalPRM's retrieval step in front of
PathFinder-PRM's **frozen, unmodified** error-classification stage and measures whether the
error-typing gets better — especially on out-of-distribution problems.

**No training. No new data collection.** Inference only, on an already-released 7B checkpoint.

---

## Research question

> If we add retrieval into PathFinder-PRM's error-typing process, does it get better at
> identifying the correct error type — especially on the hardest, most out-of-distribution
> problems?

**Expected evidence pattern.** RetrievalPRM's strongest result was not a single score but a
*trend*: gains grew with problem difficulty. We look for the same signature.

| ProcessBench subset | OOD severity | Prediction if the mechanism is real |
| --- | --- | --- |
| GSM8K | in-distribution | ~no change |
| MATH | moderate | modest gain |
| OlympiadBench | severe | larger gain |
| OmniMATH | extreme | largest gain |

A flat or negative delta is a publishable finding too, and the analysis is written to report
it honestly (see `docs/EXPERIMENTS.md`).

---

## What this repository does

```
                      ┌──────────────────────────────────────────┐
  ProcessBench        │  Condition A (baseline)                  │
  problem + steps ───▶│  PathFinder-PRM-7B, original prompt      │──▶ per-step verdicts
                      └──────────────────────────────────────────┘
                      ┌──────────────────────────────────────────┐
                 ┌───▶│  Condition B (ours)                      │
                 │    │  retrieve k refs → inject into the user  │──▶ per-step verdicts
  retrieval pool │    │  turn → SAME frozen model, same tokens   │
  (PathFinder-   │    └──────────────────────────────────────────┘
   600K) ────────┘
```

The **only** difference between A and B is extra reference text in the user message.
Model weights, special tokens, decoding rule and scoring threshold are byte-identical —
that is what makes the comparison clean.

---

## Quickstart

### 0. Install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
pip install -e .
```

### 1. Prove the pipeline works — no GPU, no downloads, ~5 seconds

```bash
python scripts/smoke.py
pytest
```

`smoke.py` runs the **entire** pipeline (pool → index → retrieval → prompt build →
scoring → ProcessBench F1 → A-vs-B comparison) against a deterministic mock PRM and a tiny
bundled fixture set. Use it to develop and to check your changes before spending GPU time.
Its numbers are meaningless by construction; what it proves is that the plumbing is intact.

### 2. Verify the real model interface

```bash
python scripts/verify_model_interface.py
```

Reproduces the exact worked example from the PathFinder-PRM-7B model card and checks our
adapter returns the same verdict. **Run this before trusting any real number.**

### 3. Real experiment

```bash
python scripts/build_pool.py         --config configs/retrieval.yaml  # download + parse pool
python scripts/build_index.py        --config configs/retrieval.yaml  # SBERT -> PCA -> matrix
python scripts/check_contamination.py --config configs/retrieval.yaml # READ THIS FIRST
python scripts/run_eval.py           --config configs/baseline.yaml   # Condition A
python scripts/run_eval.py           --config configs/retrieval.yaml  # Condition B
python scripts/compare_runs.py --a runs/baseline --b runs/retrieval
```

`compare_runs.py` writes the report table, per-subset deltas, bootstrap CIs, a McNemar test
and the OOD-trend plot into `runs/comparison/`.

### Measured on the built pool: MATH is contaminated

`check_contamination.py` on a 50K-item pool (15,316 unique questions):

| Subset | n | verbatim in pool | near-dup (cos ≥ 0.95) |
| --- | ---: | ---: | ---: |
| gsm8k | 400 | 0 (0.0%) | 0 |
| **math** | 1000 | **596 (59.6%)** | 602 |
| olympiadbench | 1000 | 0 (0.0%) | 4 |
| omnimath | 1000 | 0 (0.0%) | 22 |

Nearly 60% of the MATH subset appears **verbatim** in PathFinder-600K; the other three are
clean. The contamination guard drops these (it fired on 1,149 of 6,568 retrieval queries in
a full pass), but the concentration matters: MATH is the "moderate OOD" cell of the
evidence table, so an unguarded run would fake a trend out of memorisation. Lead with
OlympiadBench and OmniMATH — uncontaminated *and* the most out-of-distribution.
See `docs/EXPERIMENTS.md#controls`.

---

## Hardware reality check

Measured on this machine: **RTX 5070 Laptop, 8 GB VRAM, 15.3 GB system RAM**. A 7B model
in bf16 needs ≈15.2 GB of weights.

| Backend | `prm.backend` | Needs | Status here |
| --- | --- | --- | --- |
| **int4 (torchao)** | `int4` | ≈6.3 GB VRAM | ✅ **use this** — fits, no offload, ~0.36 s/step |
| Mock | `mock` | nothing | ✅ pipeline development only, numbers meaningless |
| 4-bit NF4 | `hf4bit` | ≈5.5 GB VRAM + `bitsandbytes` | ❌ blocked — bitsandbytes' native library is refused by a Windows Application Control policy (`WinError 4551`) |
| fp16 + offload | `hf` + `max_memory`/`offload_folder` | any VRAM + disk | ⚠️ works but ~12 s **per forward pass** — correctness pilots only |
| bf16 | `hf` | 16 GB+ VRAM | ▶️ for numbers directly comparable to the paper |

### Why int4 changes the project

At bf16 the model needs ~15.2 GB. With 8 GB VRAM and ~6 GB free RAM the remainder streams
from **disk on every forward pass**, which measured at ~12 s/pass — a ~150× penalty that
put a full run at roughly 190 hours.

`prm.backend: int4` quantises weights to int4 with torchao during loading, so the whole
model sits in VRAM at ~6.3 GB and nothing offloads. Measured end-to-end on real
ProcessBench data (24 solutions per condition, all four subsets):

| | disk offload (bf16) | int4 |
| --- | ---: | ---: |
| Condition A | ~100 s/solution | **5.4 s/solution** |
| Condition B (retrieval) | — | **28.3 s/solution** |
| full A+B run (3,400 each) | ~190 h | **~32 h** |

Condition B is ~5× slower than A purely because retrieval makes prompts longer (mean 841
tokens vs ~350). Profiling shows retrieval itself costs **71 ms/step — 1.2%**; the other
98.8% is the model forward pass. So prompt length, not retrieval machinery, is the lever.

If ~32 h is still too much, cut the sample rather than the method — `--limit 400` per
subset is ~15 h and still detects the 6–13 point effects the source papers report.

### Long runs are resumable — use it

Model loading peaks host RAM, and on a 16 GB machine that is close enough to the edge that
the OS may kill the process. Predictions are therefore appended to `predictions.jsonl` as
each solution finishes, and `--resume` skips whatever is already there:

```bash
python scripts/run_eval.py --config configs/pilot-int4-retrieval.yaml --name int4-b
# killed at hour 9? just re-run with --resume; at most one solution is lost
python scripts/run_eval.py --config configs/pilot-int4-retrieval.yaml --name int4-b --resume
```

Close other applications before starting a long run. On a resumed run, `wall_seconds` in
`summary.json` covers only the latest session — check `n_resumed` before quoting throughput.

Two details that matter:

- `int4_packing_format: tile_packed_to_4d` is the kernel that works on Blackwell (sm_120).
  torchao's `plain` and `preshuffled` formats need the extra `mslk` package.
- **`lm_head` and the embeddings stay in bf16.** Deliberate: PathFinder's verdict *is* a
  comparison of the `<+>` and `<->` logits from `lm_head`, so quantising that head would
  inject noise straight into the measured quantity.

**Quantisation caveat, unchanged:** absolute F1 will not match the published 69.5. Deltas
stay valid because A and B run at identical precision — never compare an int4 number to a
bf16 one. State the backend beside every number.

> **Caveat you must report.** PathFinder-PRM decides by comparing two logits (`<+>` vs `<->`).
> 4-bit quantisation perturbs exactly those logits, so absolute F1 will not match the
> published 69.5. That is acceptable here **because A and B use the identical backend** and
> the claim is about the *delta*. State the backend in the report; never compare a 4-bit
> number against the paper's bf16 number. See `docs/EXPERIMENTS.md#quantisation`.

---

## Repository layout

```
configs/            experiment configs (baseline / retrieval / smoke)
src/rapfprm/
  data/             ProcessBench loading, retrieval-pool construction
  retrieval/        SBERT encoder, PCA+cosine index, two-stage retriever
  prm/              PRM backends + prompt builders (the frozen contract lives here)
  eval/             ProcessBench runner and official metric
  analysis/         A-vs-B comparison, significance tests, plots
scripts/            CLI entry points (the five commands above)
tests/              unit tests + end-to-end smoke test
docs/               project plan, model-interface notes, experiment protocol
```

## Documentation

- `docs/PROJECT_PLAN.md` — work packages, timeline, division of labour, deliverables
- `docs/MODEL_INTERFACE.md` — the exact frozen PathFinder-PRM contract and why it must not drift
- `docs/EXPERIMENTS.md` — protocol, controls (incl. contamination guard), how to report results

## References

- Zhu et al. (2025). *Retrieval-Augmented Process Reward Model*. arXiv:2502.14361
- Pala et al. (2025). *Error Typing for Smarter Rewards*. arXiv:2505.19706
- Zheng et al. (2024). *ProcessBench*. `Qwen/ProcessBench`
- Model: `declare-lab/PathFinder-PRM-7B` · Pool: `declare-lab/PathFinder-600K`
