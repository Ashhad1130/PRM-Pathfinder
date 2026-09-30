# Condition A (baseline) vs Condition B (retrieval)

- A: `results/int4-a`
- B: `results/int4-e`

| Subset | n | OOD | F1 (A) | F1 (B) | Δ F1 | 95% CI on Δ | McNemar p | Sig. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gsm8k | 50 | in-dist | 70.0 | 69.0 | -1.0 | [-15.4, +12.4] | 1.000 | no |
| math | 50 | moderate | 72.7 | 66.5 | -6.2 | [-15.8, +2.5] | 0.371 | no |
| olympiadbench | 50 | severe | 69.8 | 63.7 | -6.2 | [-14.1, +0.0] | 0.248 | no |
| omnimath | 50 | extreme | 56.8 | 52.5 | -4.3 | [-17.8, +2.5] | 1.000 | no |
| **average** | | | **67.3** | **62.9** | **-4.4** | | | |

## Whole experiment

- Solutions compared: **200** (paired, identical in both arms)
- 95% CI on the average Δ: **[-10.0, +0.6]** — includes zero
- Pooled McNemar: **p = 0.0801**, 16 discordant (4 fixed by B, 12 broken by B), reliable = False

> Pooling is what makes the paired test usable at this sample size; the per-subset tests above rarely reach the ~25 discordant solutions they need. The pooled test asks whether the two arms decide differently overall, which is weaker than the per-subset F1 deltas it sits beside.

## OOD-severity trend

- Slope per OOD rank: **-0.95 F1**
- Spearman(severity, Δ): **-0.20**
- Monotonically increasing: **False**
- Reading: no clear difficulty-scaling pattern

> A McNemar test with fewer than ~25 discordant solutions is not trustworthy; check the `reliable` flag in `comparison.json` before quoting a p-value.