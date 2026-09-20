# Results of record

Everything here was produced by the run reported in [`../docs/RESULTS.md`](../docs/RESULTS.md):
PathFinder-PRM-7B at int4, ProcessBench, 50 solutions per subset drawn as a stratified
seeded sample, 200 solutions per arm, the same 200 in every arm.

`runs/` is gitignored because it fills up with scratch, pilots and half-finished attempts.
This directory is the subset worth keeping: the four arms that produced the reported
numbers, every contrast between them, and the contamination report they are read against.

```
int4-a/          Condition A   no references
int4-b/          Condition B   retrieved references, gold labels as <+> / <->
int4-c/          Condition C   random references, gold labels          (length control)
int4-d/          Condition D   retrieved references, no labels         (label ablation)

a-vs-b/          the headline delta
a-vs-c/          what extra text costs on its own
c-vs-b/          what relevance buys, prompt length held fixed
a-vs-d/          what unlabelled references cost
d-vs-b/          what the labels cost

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

Every number is int4. Deltas between arms are valid because all arms ran at identical
precision; absolute F1 is not comparable to the published bf16 69.5, though Condition A
lands close to it at 67.3.
