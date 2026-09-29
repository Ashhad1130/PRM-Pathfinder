# Condition A (baseline) vs Condition B (retrieval)

- A: `results/int4-a`
- B: `results/int4-b`
- **Filtered: pool-contaminated solutions removed, per results/contamination.json** (633 solution(s) removed from both conditions). The `n` column below is what survived, so these numbers are not comparable with the unfiltered table.

| Subset | n | OOD | F1 (A) | F1 (B) | Δ F1 | 95% CI on Δ | McNemar p | Sig. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gsm8k | 50 | in-dist | 70.0 | 55.1 | -14.9 | [-33.2, +2.2] | 0.043 | no |
| math | 24 | moderate | 75.0 | 47.3 | -27.7 | [-57.5, -3.2] | 0.077 | yes |
| olympiadbench | 49 | severe | 69.0 | 47.5 | -21.5 | [-35.8, -10.2] | 0.004 | yes |
| omnimath | 50 | extreme | 56.8 | 45.7 | -11.0 | [-26.0, +1.2] | 0.181 | no |
| **average** | | | **67.7** | **48.9** | **-18.8** | | | |

## Whole experiment

- Solutions compared: **173** (paired, identical in both arms)
- 95% CI on the average Δ: **[-27.9, -9.9]** — excludes zero
- Pooled McNemar: **p = 0.0000**, 44 discordant (7 fixed by B, 37 broken by B), reliable = True

> Pooling is what makes the paired test usable at this sample size; the per-subset tests above rarely reach the ~25 discordant solutions they need. The pooled test asks whether the two arms decide differently overall, which is weaker than the per-subset F1 deltas it sits beside.

## OOD-severity trend

- Slope per OOD rank: **+1.76 F1**
- Spearman(severity, Δ): **+0.40**
- Monotonically increasing: **False**
- Reading: no clear difficulty-scaling pattern

> A McNemar test with fewer than ~25 discordant solutions is not trustworthy; check the `reliable` flag in `comparison.json` before quoting a p-value.