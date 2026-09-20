# Condition A (baseline) vs Condition B (retrieval)

- A: `runs/int4-a`
- B: `runs/int4-d`
- **Filtered: pool-contaminated solutions removed, per runs/contamination.json** (633 solution(s) removed from both conditions). The `n` column below is what survived, so these numbers are not comparable with the unfiltered table.

| Subset | n | OOD | F1 (A) | F1 (B) | Δ F1 | 95% CI on Δ | McNemar p | Sig. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gsm8k | 50 | in-dist | 70.0 | 65.7 | -4.3 | [-17.3, +7.7] | 0.617 | no |
| math | 24 | moderate | 75.0 | 67.0 | -8.0 | [-27.9, +10.0] | 0.617 | no |
| olympiadbench | 49 | severe | 69.0 | 62.7 | -6.3 | [-16.5, +2.5] | 0.371 | no |
| omnimath | 50 | extreme | 56.8 | 52.5 | -4.3 | [-18.0, +3.9] | 1.000 | no |
| **average** | | | **67.7** | **62.0** | **-5.7** | | | |

## Whole experiment

- Solutions compared: **173** (paired, identical in both arms)
- 95% CI on the average Δ: **[-13.0, +0.3]** — includes zero
- Pooled McNemar: **p = 0.0990**, 18 discordant (5 fixed by B, 13 broken by B), reliable = False

> Pooling is what makes the paired test usable at this sample size; the per-subset tests above rarely reach the ~25 discordant solutions they need. The pooled test asks whether the two arms decide differently overall, which is weaker than the per-subset F1 deltas it sits beside.

## OOD-severity trend

- Slope per OOD rank: **+0.18 F1**
- Spearman(severity, Δ): **+0.40**
- Monotonically increasing: **False**
- Reading: no clear difficulty-scaling pattern

> A McNemar test with fewer than ~25 discordant solutions is not trustworthy; check the `reliable` flag in `comparison.json` before quoting a p-value.