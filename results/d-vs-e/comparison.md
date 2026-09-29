# Condition A (baseline) vs Condition B (retrieval)

- A: `results/int4-d`
- B: `results/int4-e`

| Subset | n | OOD | F1 (A) | F1 (B) | Δ F1 | 95% CI on Δ | McNemar p | Sig. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gsm8k | 50 | in-dist | 65.7 | 69.0 | +3.2 | [-8.0, +15.6] | 1.000 | no |
| math | 50 | moderate | 61.6 | 66.5 | +4.9 | [-4.3, +15.7] | 0.683 | no |
| olympiadbench | 50 | severe | 63.7 | 63.7 | +0.0 | [-5.8, +6.4] | 0.480 | no |
| omnimath | 50 | extreme | 52.5 | 52.5 | +0.0 | [-5.1, +5.2] | 0.617 | no |
| **average** | | | **60.9** | **62.9** | **+2.0** | | | |

## Whole experiment

- Solutions compared: **200** (paired, identical in both arms)
- 95% CI on the average Δ: **[-2.2, +6.4]** — includes zero
- Pooled McNemar: **p = 0.6056**, 15 discordant (9 fixed by B, 6 broken by B), reliable = False

> Pooling is what makes the paired test usable at this sample size; the per-subset tests above rarely reach the ~25 discordant solutions they need. The pooled test asks whether the two arms decide differently overall, which is weaker than the per-subset F1 deltas it sits beside.

## OOD-severity trend

- Slope per OOD rank: **-1.46 F1**
- Spearman(severity, Δ): **-0.74**
- Monotonically increasing: **False**
- Reading: no clear difficulty-scaling pattern

> A McNemar test with fewer than ~25 discordant solutions is not trustworthy; check the `reliable` flag in `comparison.json` before quoting a p-value.