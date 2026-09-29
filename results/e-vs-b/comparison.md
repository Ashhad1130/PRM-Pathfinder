# Condition A (baseline) vs Condition B (retrieval)

- A: `results/int4-e`
- B: `results/int4-b`

| Subset | n | OOD | F1 (A) | F1 (B) | Δ F1 | 95% CI on Δ | McNemar p | Sig. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gsm8k | 50 | in-dist | 69.0 | 55.1 | -13.8 | [-26.0, -4.0] | 0.023 | yes |
| math | 50 | moderate | 66.5 | 50.3 | -16.2 | [-28.8, -4.9] | 0.027 | yes |
| olympiadbench | 50 | severe | 63.7 | 49.0 | -14.7 | [-27.0, -5.2] | 0.023 | yes |
| omnimath | 50 | extreme | 52.5 | 45.7 | -6.8 | [-19.4, +3.2] | 0.302 | no |
| **average** | | | **62.9** | **50.0** | **-12.9** | | | |

## Whole experiment

- Solutions compared: **200** (paired, identical in both arms)
- 95% CI on the average Δ: **[-19.0, -7.7]** — excludes zero
- Pooled McNemar: **p = 0.0000**, 39 discordant (6 fixed by B, 33 broken by B), reliable = True

> Pooling is what makes the paired test usable at this sample size; the per-subset tests above rarely reach the ~25 discordant solutions they need. The pooled test asks whether the two arms decide differently overall, which is weaker than the per-subset F1 deltas it sits beside.

## OOD-severity trend

- Slope per OOD rank: **+2.25 F1**
- Spearman(severity, Δ): **+0.40**
- Monotonically increasing: **False**
- Reading: no clear difficulty-scaling pattern

> A McNemar test with fewer than ~25 discordant solutions is not trustworthy; check the `reliable` flag in `comparison.json` before quoting a p-value.