# Results of record

Everything here was produced by the run reported in [`../docs/RESULTS.md`](../docs/RESULTS.md):
PathFinder-PRM-7B at int4, ProcessBench, 50 solutions per subset drawn as a stratified
seeded sample, 200 solutions per arm, the same 200 in every arm.

`runs/` is gitignored because it fills up with scratch, pilots and half-finished attempts.
This directory is the subset worth keeping: the five arms that produced the reported
numbers, every contrast between them, and the contamination report they are read against.

```
int4-a/          Condition A   no references
int4-b/          Condition B   retrieved references, gold labels as <+> / <->
int4-c/          Condition C   random references, gold labels          (length control)
int4-d/          Condition D   retrieved references, no labels         (label ablation)
int4-e/          Condition E   retrieved references, labels as words   (token ablation)
int4-b-fallback-encoder/       the first, superseded run of B; see below

a-vs-b/          the headline delta
a-vs-c/          what extra text costs on its own
c-vs-b/          what relevance buys, prompt length held fixed
a-vs-d/          what unlabelled references cost
d-vs-b/          what the labels cost
a-vs-e/          what word-labelled references cost
d-vs-e/          what the judgement costs when written in words
e-vs-b/          what the verdict tokens cost, information held fixed

contamination.json     eval/pool overlap, with the 633 affected uids
arms.png               every arm on one axis
verdict_profile.json   false alarms, misses and misplaced blame per arm
```

Each arm directory holds `config.resolved.yaml` (the exact settings, including the code
revision), `metrics.json`, `summary.json`, `predictions.jsonl` (one row per solution) and
`traces.jsonl` (one row per scored step, with the retrieved reference ids and their
similarities).

Each contrast directory holds `comparison.md` and `comparison.json`, plus an
`uncontaminated-` twin recomputed with the 633 pool-contaminated eval solutions dropped
from both arms.

`int4-b-fallback-encoder/` is the first run of B. It used a fallback encoder that skipped
Sentence-BERT's normalisation step, so its retrieval was degraded (see docs/RESULTS.md). It
is kept for the record and used in no contrast; `int4-b/` is the corrected re-run. C ran
under the same fallback but draws random references, and was verified to be unaffected.

`int4-b/` and `int4-e/` were interrupted and resumed, so their `summary.json` wall time and
retrieval counters cover only the final session; `predictions.jsonl` and `traces.jsonl` are
complete.

Every number is int4. Deltas between arms are valid because all arms ran at identical
precision; absolute F1 is not comparable to the published bf16 69.5, though Condition A
lands close to it at 67.3.
