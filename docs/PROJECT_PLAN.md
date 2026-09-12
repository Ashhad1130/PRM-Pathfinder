# Project plan

Seminar project · Institute for Computational Linguistics, Heidelberg University
Pratik Goyal & Ashhad Raza Quadri · Supervisor: Lei Tang

## Research question

> If we add retrieval into PathFinder-PRM's error-typing process, does it get better at
> identifying the correct error type — especially on the hardest, most out-of-distribution
> problems?

Neither source paper tried this. RetrievalPRM added retrieval to a *binary* grader;
PathFinder-PRM added error typing but grades every step cold. The gap between them is the
project.

## Status

| Work package | State |
| --- | --- |
| WP0 · repo, config, CI-able tests | **done** — 64 tests, `scripts/smoke.py` green |
| WP1 · frozen PRM adapter | **done** — byte-parity with the model card, verified against the real tokenizer |
| WP2 · retrieval pool + index | **done** — 50K items, 15,316 questions, index built in 78s |
| WP3 · evaluation + metric | **done** — real ProcessBench (3,400 solutions) loads and scores |
| WP4 · analysis + plots | **done** — CIs, McNemar, OOD-trend test |
| WP4b · contamination audit | **done** — `scripts/check_contamination.py`; MATH is 59.6% contaminated |
| WP5 · live model verification | **in progress** — 15 GB checkpoint downloaded; loads with disk offload |
| WP6 · pilot runs (`--limit 25`) | **blocked on hardware** — see below |
| WP7 · full runs A + B | **needs a 16 GB+ GPU** |
| WP8 · ablations | pending — see docs/EXPERIMENTS.md |
| WP9 · report + slides | pending |

Everything through WP4b runs on this laptop. WP6–WP8 need cluster time.

## Environment notes (this machine)

Two machine-level obstacles, both diagnosed and worked around in code:

1. **An Application Control policy blocks unsigned native libraries.** It refuses
   `bitsandbytes` (so 4-bit quantisation is unavailable) and `pyarrow`. The pyarrow block
   is the nastier one: scikit-learn guards `import pyarrow` with `except
   ModuleNotFoundError`, but a blocked DLL raises plain `ImportError`, so sklearn fails to
   import — and `transformers` imports sklearn, so *no model could load at all*.
   `src/rapfprm/compat.py` presents an unloadable pyarrow as absent, restoring the intended
   fallback. It is a no-op on healthy machines.
2. **The project no longer needs pyarrow.** PCA is numpy SVD (not sklearn), ProcessBench is
   read from its published `.json` (not `datasets`), and there is a transformers-only
   mean-pooling encoder when sentence-transformers is unimportable. Only
   `scripts/build_pool.py` still needs `datasets` — and the pool is already built.

## Compute plan

| Stage | Where | Rough cost |
| --- | --- | --- |
| Pipeline development | laptop, mock backend | seconds |
| Pool + index build | laptop CPU | minutes (50K items) |
| Pilot (25/subset) | laptop, `hf4bit` | ~1 h |
| Full run A + B (3400 solutions × 2) | 16 GB+ GPU, bf16 | the real budget item |

Cost driver: PathFinder-PRM scores **one step at a time**, and a solution has ~5–10 steps.
`prm.early_stop: true` stops at the first flagged step, which cuts a large fraction of the
forward passes. Set it to `false` only when you want full per-step traces for error
analysis.

The 8 GB laptop cannot hold the 7B model at bf16 (~15 GB). Options, best first:

1. University cluster / Colab / RunPod at bf16 — required for numbers comparable to the paper.
2. `prm.backend: hf4bit` on the laptop — fits in ~5.5 GB, but shifts the very logits the
   verdict is read from. Fine for deltas, not for absolute numbers. See
   `docs/EXPERIMENTS.md#quantisation`.

## Division of labour (suggested)

| Area | Owner |
| --- | --- |
| PRM adapter, prompt fidelity, live verification | one of us |
| Retrieval pool, index, contamination guard, ablations | the other |
| Evaluation harness + analysis | shared — it is the shared contract |
| Report + slides | shared |

## Risks

| Risk | Mitigation |
| --- | --- |
| Contamination inflates Condition B | similarity guard on by default, counted in every run summary, plus a sensitivity sweep |
| Baseline fails to reproduce 69.5 | `verify_model_interface.py` catches prompt drift before any run; reproduce A before touching B |
| 8 GB VRAM | 4-bit path implemented; bf16 numbers deferred to a cluster |
| Retrieval simply does not help | committed in the pitch to reporting a negative result; `docs/EXPERIMENTS.md` says how to write each outcome |
| Silent schema drift upstream | loaders raise on missing fields instead of coercing; the pool parser skips unparseable rows and fails if it parses none |
| Prompt-length confound | random-reference control isolates "extra text" from "relevant text" |

## Deliverables

- Reproducible code with a one-command smoke test (this repo)
- `runs/comparison/comparison.md` — the results table
- Two figures: F1 by subset, and Δ F1 against OOD severity
- Report framing this explicitly as an **empirical combination of two existing ideas**,
  not a new method — as committed in the pitch

## Open questions for the supervisor

Carried over from the pitch's discussion slide:

1. Is F1 with vs. without retrieval, across all four subsets, sufficient evidence — or
   should we also report per-error-type accuracy (math vs. consistency), which the
   hierarchical model gives us for free and which speaks more directly to "better at
   identifying the correct error type"?
2. Any concern about reusing PathFinder-600K as the retrieval pool, given it shares
   upstream sources with ProcessBench? Our guard drops near-duplicates; is that enough,
   or should the pool exclude MATH-derived items entirely?
3. Does the scope feel right, or should we narrow to two subsets and add the ablations?
