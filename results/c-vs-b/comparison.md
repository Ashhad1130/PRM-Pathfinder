# Condition A (baseline) vs Condition B (retrieval)

- A: `runs/int4-c`
- B: `runs/int4-b`

| Subset | n | OOD | F1 (A) | F1 (B) | Δ F1 | 95% CI on Δ | McNemar p | Sig. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gsm8k | 50 | in-dist | 54.1 | 54.1 | +0.0 | [-10.3, +10.5] | 0.683 | no |
| math | 50 | moderate | 54.3 | 51.6 | -2.7 | [-18.2, +12.2] | 1.000 | no |
| olympiadbench | 50 | severe | 34.5 | 49.3 | +14.8 | [-0.0, +29.7] | 0.096 | no |
| omnimath | 50 | extreme | 34.2 | 30.2 | -4.0 | [-21.6, +13.7] | 0.803 | no |
| **average** | | | **44.3** | **46.3** | **+2.0** | | | |

## Whole experiment

- Solutions compared: **200** (paired, identical in both arms)
- 95% CI on the average Δ: **[-5.5, +9.1]** — includes zero
- Pooled McNemar: **p = 0.6650**, 48 discordant (26 fixed by B, 22 broken by B), reliable = True

> Pooling is what makes the paired test usable at this sample size; the per-subset tests above rarely reach the ~25 discordant solutions they need. The pooled test asks whether the two arms decide differently overall, which is weaker than the per-subset F1 deltas it sits beside.

## OOD-severity trend

- Slope per OOD rank: **+0.56 F1**
- Spearman(severity, Δ): **-0.40**
- Monotonically increasing: **False**
- Reading: no clear difficulty-scaling pattern

> A McNemar test with fewer than ~25 discordant solutions is not trustworthy; check the `reliable` flag in `comparison.json` before quoting a p-value.