# Condition A (baseline) vs Condition B (retrieval)

- A: `results/int4-a`
- B: `results/int4-d`

| Subset | n | OOD | F1 (A) | F1 (B) | Δ F1 | 95% CI on Δ | McNemar p | Sig. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gsm8k | 50 | in-dist | 70.0 | 65.7 | -4.3 | [-17.3, +7.7] | 0.617 | no |
| math | 50 | moderate | 72.7 | 61.6 | -11.1 | [-23.9, -0.6] | 0.131 | yes |
| olympiadbench | 50 | severe | 69.8 | 63.7 | -6.2 | [-16.3, +2.1] | 0.371 | no |
| omnimath | 50 | extreme | 56.8 | 52.5 | -4.3 | [-18.0, +3.9] | 1.000 | no |
| **average** | | | **67.3** | **60.9** | **-6.5** | | | |

## Whole experiment

- Solutions compared: **200** (paired, identical in both arms)
- 95% CI on the average Δ: **[-12.8, -1.5]** — excludes zero
- Pooled McNemar: **p = 0.0291**, 21 discordant (5 fixed by B, 16 broken by B), reliable = False

> Pooling is what makes the paired test usable at this sample size; the per-subset tests above rarely reach the ~25 discordant solutions they need. The pooled test asks whether the two arms decide differently overall, which is weaker than the per-subset F1 deltas it sits beside.

## OOD-severity trend

- Slope per OOD rank: **+0.50 F1**
- Spearman(severity, Δ): **+0.40**
- Monotonically increasing: **False**
- Reading: no clear difficulty-scaling pattern

> A McNemar test with fewer than ~25 discordant solutions is not trustworthy; check the `reliable` flag in `comparison.json` before quoting a p-value.