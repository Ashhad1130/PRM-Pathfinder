# Condition A (baseline) vs Condition B (retrieval)

- A: `results/int4-a`
- B: `results/int4-c`

| Subset | n | OOD | F1 (A) | F1 (B) | Δ F1 | 95% CI on Δ | McNemar p | Sig. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gsm8k | 50 | in-dist | 70.0 | 54.1 | -15.9 | [-34.1, +1.2] | 0.027 | no |
| math | 50 | moderate | 72.7 | 54.3 | -18.4 | [-32.1, -6.3] | 0.016 | yes |
| olympiadbench | 50 | severe | 69.8 | 34.5 | -35.3 | [-50.8, -22.4] | 0.000 | yes |
| omnimath | 50 | extreme | 56.8 | 34.2 | -22.6 | [-40.7, -6.4] | 0.014 | yes |
| **average** | | | **67.3** | **44.3** | **-23.0** | | | |

## Whole experiment

- Solutions compared: **200** (paired, identical in both arms)
- 95% CI on the average Δ: **[-31.2, -15.4]** — excludes zero
- Pooled McNemar: **p = 0.0000**, 61 discordant (7 fixed by B, 54 broken by B), reliable = True

> Pooling is what makes the paired test usable at this sample size; the per-subset tests above rarely reach the ~25 discordant solutions they need. The pooled test asks whether the two arms decide differently overall, which is weaker than the per-subset F1 deltas it sits beside.

## OOD-severity trend

- Slope per OOD rank: **-3.70 F1**
- Spearman(severity, Δ): **-0.80**
- Monotonically increasing: **False**
- Reading: no clear difficulty-scaling pattern

> A McNemar test with fewer than ~25 discordant solutions is not trustworthy; check the `reliable` flag in `comparison.json` before quoting a p-value.