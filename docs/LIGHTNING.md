# Running the experiment on Lightning AI

The laptop path (8 GB VRAM, int4, ~32 h for a full A+B run) is documented in the README.
This page is the cloud path: a Lightning Studio with a real GPU, where the model fits in
bf16 and the numbers are directly comparable to the published ones.

## On a 40 GB A100: one script

If the Studio has a big GPU, there is nothing to assemble. `run_a100.sh` installs the
project, verifies the prompt contract against the model card, builds the pool and index,
measures contamination, runs all five arms at bf16, computes all seven contrasts with their
contamination-excluded twins, draws the figure, profiles the verdicts and prints a summary:

```bash
git clone https://github.com/Ashhad1130/PRM-Pathfinder.git && cd PRM-Pathfinder
git checkout feat/retrieval-control-arm

bash scripts/lightning/run_a100.sh --dry-run        # see the plan, run nothing
bash scripts/lightning/run_a100.sh --pilot          # 25/subset, ~1 h, confirms the setup
tmux new -s prm 'bash scripts/lightning/run_a100.sh --limit 100'
```

The arms it runs:

| | References | Isolates |
| --- | --- | --- |
| A | none | the baseline; should land near the published 69.5 |
| B | retrieved, labels as `<+>`/`<->` | the full treatment |
| C | random, labels as `<+>`/`<->` | prompt length, not relevance |
| D | retrieved, no labels | the reference text alone |
| E | retrieved, labels as words | the verdict tokens, judgement held fixed |

### Surviving seven unattended hours

Three separate things can end a long run, and tmux only fixes one of them.

**The Studio's idle timeout is the one that will actually catch you.** Lightning stops an
idle Studio to protect your credits, and that takes the machine with it — tmux included.
Before starting, open the Studio's compute settings and set auto-shutdown to its longest
value, or turn it off. Files survive the stop; the run does not.

**A closed browser tab** kills a foreground process. That is what tmux is for:

```bash
tmux new -s prm 'bash scripts/lightning/keepalive.sh --limit 100'
tmux attach -t prm      # detach again with ctrl-b then d
```

**A transient failure at hour five** — a CUDA OOM on one long solution, a dropped Hub
connection — is what `keepalive.sh` is for. It re-invokes the runner until the study
completes, which is safe because every stage is resumable and each arm continues from its
own `predictions.jsonl`. It gives up after six consecutive failures, and it refuses to retry
at all when the first attempt dies within a minute, since that means a config or setup
error that no amount of retrying will fix.

Check your credit balance first as well: a seven-hour A100 session is a large withdrawal,
and running out mid-run stops the machine.

The rest of this page is the manual path, the int4 route for smaller cards, and the
troubleshooting table.

---

Three commands for the manual path:

```bash
bash scripts/lightning/setup.sh                      # install + verify, ~5 min
bash scripts/lightning/run_experiment.sh --dry-run   # check the plan
tmux new -s rapfprm 'bash scripts/lightning/run_experiment.sh --limit 400'
```

---

## 1. Create the Studio

1. New Studio → **GPU**. Which one matters:

   | GPU | VRAM | Backend to use | Comparable to the paper? |
   | --- | ---: | --- | --- |
   | L4 / A10G | 24 GB | `hf` (bf16), the configs as they ship | ✅ yes |
   | A100 | 40/80 GB | `hf` (bf16) | ✅ yes, fastest |
   | T4 | 16 GB | `--backend int4` (bf16 is too tight) | ⚠️ deltas only, see the quantisation caveat |
   | CPU-only | — | pool/index build only | — |

   A 7B in bf16 is ~15.2 GB of weights before activations and the KV cache, so 16 GB is
   not enough in practice. `setup.sh` prints a recommendation for whatever it lands on.

2. Clone the repo into the Studio and `cd` into it:

   ```bash
   git clone <your-remote> project && cd project
   ```

   Everything under the Studio home persists across machine switches, which is what makes
   the split below work.

## 2. Setup

```bash
bash scripts/lightning/setup.sh            # installs requirements, verifies torch/CUDA, runs the smoke test
bash scripts/lightning/setup.sh --int4     # add this on a 16 GB GPU (installs torchao)
bash scripts/lightning/setup.sh --flash    # optional: builds flash-attn (10+ min; sdpa is a fine fallback)
```

It is idempotent — re-run it after every machine switch. It sets `HF_HOME` to
`.hf_cache/` inside the repo so the 15 GB checkpoint is downloaded once and survives the
switch from a CPU machine to a GPU one.

The last thing it does is `scripts/smoke.py`: the whole pipeline against a mock PRM, five
seconds, no GPU. If that fails, nothing downstream is worth starting.

## 3. Run

```bash
bash scripts/lightning/run_experiment.sh [options]
```

Stages, in order. Each is skipped when its output is already there:

| Stage | Produces | Time |
| --- | --- | --- |
| `pool` | `data/pool/pool.jsonl` — 50K items from PathFinder-600K | ~10 min |
| `index` | `artifacts/index/` — SBERT → PCA → matrix | ~10 min (GPU) |
| `contamination` | `runs/contamination.json` — **read this before the results** | ~5 min |
| `a` | `runs/baseline/` — Condition A, no retrieval | hours |
| `b` | `runs/retrieval/` — Condition B, retrieval | hours |
| `c` | `runs/control-random/` — Condition C, random references | hours (≈ B) |
| `compare` | `runs/comparison*/` — deltas, CIs, McNemar, OOD plot | seconds |

The `compare` stage writes three contrasts, and a contamination-excluded copy of each
whenever `runs/contamination.json` is present:

| Directory | Contrast | What it answers |
| --- | --- | --- |
| `runs/comparison/` | A → B | the headline delta |
| `runs/comparison-a-vs-control/` | A → C | what extra text buys on its own |
| `runs/comparison-control-vs-b/` | C → B | what *relevance* buys, prompt length held fixed |

C → B is the one the hypothesis rests on. `--no-control` skips Condition C and both control
contrasts; the report then has to say the headline gain has no control behind it.

Useful options (`--help` lists them all):

```bash
--limit 400             # 400 solutions/subset instead of the full 3400
--subsets "olympiadbench omnimath"   # the two uncontaminated, most-OOD subsets
--backend int4          # applied to EVERY arm, never just one
--stages pool,index     # e.g. do the CPU-only prep, then switch to a GPU machine
--no-control            # skip Condition C (saves ~a third of the GPU time, costs the control)
--dry-run               # print the plan, execute nothing
```

### Split the work across machines to save GPU credit

`pool` and `index` do not need the big GPU. On a CPU or cheap GPU machine:

```bash
bash scripts/lightning/run_experiment.sh --stages pool,index,contamination
```

then switch the Studio to the A100/L4 and run:

```bash
bash scripts/lightning/run_experiment.sh --stages a,b,c,compare --limit 400
```

The pool, the index and the HF cache are all inside the repo, so the second machine finds
them already built.

### Long runs

Use tmux. A Studio keeps running when the browser tab closes, but the foreground process
of a closed terminal does not:

```bash
tmux new -s rapfprm 'bash scripts/lightning/run_experiment.sh --limit 400'
tmux attach -t rapfprm      # detach again with ctrl-b then d
```

Every stage also tees to `runs/logs/<timestamp>-<stage>.log`, so progress is readable
from a second terminal:

```bash
tail -f runs/logs/*-eval-b.log
```

**If it dies, re-run the exact same command.** Finished stages are skipped and both eval
stages resume from their `predictions.jsonl`; at most one solution is lost. Start from
scratch only with `--no-resume`.

## What the driver refuses to do

Before anything expensive starts, it compares the arms' configs pairwise (A vs B, and B vs
C when the control is in the plan) and **aborts if they differ in anything but
`retrieval.enabled` or `retrieval.reference_mode`** (`scripts/lightning/_check_parity.py`).
A drifted `prm.max_input_tokens`, a different `limit_per_subset` or a `top_k_steps` that
moved in one arm and not another would turn the measured delta into a config artefact, and
you would only find out after the GPU hours were spent. That is why `--limit`, `--subsets`
and `--backend` apply to every arm at once, and why there is no `--backend-a`.

It also refuses two configs that describe the *same* arm — running those would measure
nothing at all.

## Cost estimate

Timings scale with the model forward pass, which dominates everything (retrieval itself is
~1% of Condition B's step time — see `docs/EXPERIMENTS.md`). Extrapolating from the
measured int4 laptop figures, on a 24 GB GPU in bf16:

| Scope | Solutions (A+B+C) | Rough wall time |
| --- | ---: | ---: |
| `--limit 25` sanity pilot | 300 | ~1.5 h |
| `--limit 400` | 9,600 | ~15–22 h |
| full benchmark | 10,200 | ~22–35 h |

Run a `--limit 25` pilot first and read the actual `wall_seconds` in
`runs/baseline/summary.json` — that is the only number that predicts your machine. Budget
Conditions B and C at roughly 3–5× Condition A each: their prompts are longer (mean 841
tokens vs ~350) and identical to each other in length, which is the point of the control.

`--no-control` is the single biggest saving available and the most expensive one
scientifically. If compute forces the choice, prefer cutting `--limit` over cutting the
control: a smaller sample widens the CIs, a missing control makes the headline claim
unfalsifiable.

## Reading the results

```bash
cat runs/comparison/comparison.md
```

Four things to check before quoting anything:

1. **The backend.** State it beside every number. An int4 F1 and a bf16 F1 are not
   comparable, and neither is comparable to the published 69.5 unless you ran bf16.
2. **`runs/contamination.json`.** ~60% of the MATH subset appears verbatim in the pool.
   Lead with OlympiadBench and OmniMATH — uncontaminated *and* the most OOD.
3. **`n_resumed` in each `summary.json`.** `wall_seconds` covers the latest session only,
   so throughput figures from a resumed run are wrong.
4. **`retrieval.mean_step_similarity` in B's and C's `summary.json`.** C's must sit far
   below B's. If they are close, the random draw was not a meaningful contrast and C is not
   a control.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `CUDA out of memory` while loading | bf16 does not fit. `--backend int4` (after `setup.sh --int4`), or a bigger GPU. |
| Download stalls or 429s | `hf auth login`. `setup.sh` warns when you are not authenticated. |
| Stage fails, you fixed it | Re-run the same command; finished stages are skipped, evals resume. |
| `config parity check failed` | The arms' configs drifted. The message names the exact keys. Only `retrieval.enabled` and `retrieval.reference_mode` may differ. |
| Eval much slower than expected | Check `nvidia-smi` shows the model on the GPU; a silent CPU/disk offload is the usual cause. |
| Disk full | The checkpoint (15 GB) + pool (3 GB) + caches need ~40 GB. `du -sh .hf_cache data artifacts`. |
