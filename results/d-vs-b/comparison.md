# Condition A (baseline) vs Condition B (retrieval)

- A: `results/int4-d`
- B: `results/int4-b`

| Subset | n | OOD | F1 (A) | F1 (B) | Δ F1 | 95% CI on Δ | McNemar p | Sig. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gsm8k | 50 | in-dist | 65.7 | 55.1 | -10.6 | [-27.1, +6.5] | 0.114 | no |
| math | 50 | moderate | 61.6 | 50.3 | -11.3 | [-24.5, +0.0] | 0.114 | no |
| olympiadbench | 50 | severe | 63.7 | 49.0 | -14.7 | [-26.7, -5.4] | 0.023 | yes |
| omnimath | 50 | extreme | 52.5 | 45.7 | -6.8 | [-18.1, +2.7] | 0.267 | no |
| **average** | | | **60.9** | **50.0** | **-10.8** | | | |

## Whole experiment

- Solutions compared: **200** (paired, identical in both arms)
- 95% CI on the average Δ: **[-17.8, -4.7]** — excludes zero
- Pooled McNemar: **p = 0.0003**, 40 discordant (8 fixed by B, 32 broken by B), reliable = True

> Pooling is what makes the paired test usable at this sample size; the per-subset tests above rarely reach the ~25 discordant solutions they need. The pooled test asks whether the two arms decide differently overall, which is weaker than the per-subset F1 deltas it sits beside.

## OOD-severity trend

- Slope per OOD rank: **+0.79 F1**
- Spearman(severity, Δ): **+0.20**
- Monotonically increasing: **False**
- Reading: no clear difficulty-scaling pattern

> A McNemar test with fewer than ~25 discordant solutions is not trustworthy; check the `reliable` flag in `comparison.json` before quoting a p-value.