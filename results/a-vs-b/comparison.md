# Condition A (baseline) vs Condition B (retrieval)

- A: `results/int4-a`
- B: `results/int4-b`

| Subset | n | OOD | F1 (A) | F1 (B) | Δ F1 | 95% CI on Δ | McNemar p | Sig. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gsm8k | 50 | in-dist | 70.0 | 55.1 | -14.9 | [-33.2, +2.2] | 0.043 | no |
| math | 50 | moderate | 72.7 | 50.3 | -22.4 | [-38.0, -8.2] | 0.010 | yes |
| olympiadbench | 50 | severe | 69.8 | 49.0 | -20.9 | [-33.7, -9.9] | 0.004 | yes |
| omnimath | 50 | extreme | 56.8 | 45.7 | -11.0 | [-26.0, +1.2] | 0.181 | no |
| **average** | | | **67.3** | **50.0** | **-17.3** | | | |

## Whole experiment

- Solutions compared: **200** (paired, identical in both arms)
- 95% CI on the average Δ: **[-25.2, -10.1]** — excludes zero
- Pooled McNemar: **p = 0.0000**, 51 discordant (8 fixed by B, 43 broken by B), reliable = True

> Pooling is what makes the paired test usable at this sample size; the per-subset tests above rarely reach the ~25 discordant solutions they need. The pooled test asks whether the two arms decide differently overall, which is weaker than the per-subset F1 deltas it sits beside.

## OOD-severity trend

- Slope per OOD rank: **+1.30 F1**
- Spearman(severity, Δ): **+0.40**
- Monotonically increasing: **False**
- Reading: no clear difficulty-scaling pattern

> A McNemar test with fewer than ~25 discordant solutions is not trustworthy; check the `reliable` flag in `comparison.json` before quoting a p-value.