# Condition A (baseline) vs Condition B (retrieval)

- A: `runs/int4-d`
- B: `runs/int4-b`

| Subset | n | OOD | F1 (A) | F1 (B) | Δ F1 | 95% CI on Δ | McNemar p | Sig. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gsm8k | 50 | in-dist | 65.7 | 54.1 | -11.6 | [-30.0, +7.7] | 0.096 | no |
| math | 50 | moderate | 61.6 | 51.6 | -10.0 | [-23.3, +1.7] | 0.267 | no |
| olympiadbench | 50 | severe | 63.7 | 49.3 | -14.4 | [-26.8, -3.2] | 0.046 | yes |
| omnimath | 50 | extreme | 52.5 | 30.2 | -22.3 | [-38.5, -6.9] | 0.002 | yes |
| **average** | | | **60.9** | **46.3** | **-14.6** | | | |

## Whole experiment

- Solutions compared: **200** (paired, identical in both arms)
- 95% CI on the average Δ: **[-22.0, -7.1]** — excludes zero
- Pooled McNemar: **p = 0.0000**, 50 discordant (9 fixed by B, 41 broken by B), reliable = True

> Pooling is what makes the paired test usable at this sample size; the per-subset tests above rarely reach the ~25 discordant solutions they need. The pooled test asks whether the two arms decide differently overall, which is weaker than the per-subset F1 deltas it sits beside.

## OOD-severity trend

- Slope per OOD rank: **-3.64 F1**
- Spearman(severity, Δ): **-0.80**
- Monotonically increasing: **False**
- Reading: no clear difficulty-scaling pattern

> A McNemar test with fewer than ~25 discordant solutions is not trustworthy; check the `reliable` flag in `comparison.json` before quoting a p-value.