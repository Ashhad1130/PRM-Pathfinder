# Condition A (baseline) vs Condition B (retrieval)

- A: `results/int4-d`
- B: `results/int4-e`
- **Filtered: pool-contaminated solutions removed, per results/contamination.json** (633 solution(s) removed from both conditions). The `n` column below is what survived, so these numbers are not comparable with the unfiltered table.

| Subset | n | OOD | F1 (A) | F1 (B) | Δ F1 | 95% CI on Δ | McNemar p | Sig. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gsm8k | 50 | in-dist | 65.7 | 69.0 | +3.2 | [-8.0, +15.6] | 1.000 | no |
| math | 24 | moderate | 67.0 | 79.6 | +12.6 | [-5.2, +33.3] | 0.371 | no |
| olympiadbench | 49 | severe | 62.7 | 62.7 | +0.0 | [-6.2, +6.5] | 0.480 | no |
| omnimath | 50 | extreme | 52.5 | 52.5 | +0.0 | [-5.1, +5.2] | 0.617 | no |
| **average** | | | **62.0** | **65.9** | **+4.0** | | | |

## Whole experiment

- Solutions compared: **173** (paired, identical in both arms)
- 95% CI on the average Δ: **[-1.9, +10.6]** — includes zero
- Pooled McNemar: **p = 0.4227**, 14 discordant (9 fixed by B, 5 broken by B), reliable = False

> Pooling is what makes the paired test usable at this sample size; the per-subset tests above rarely reach the ~25 discordant solutions they need. The pooled test asks whether the two arms decide differently overall, which is weaker than the per-subset F1 deltas it sits beside.

## OOD-severity trend

- Slope per OOD rank: **-2.23 F1**
- Spearman(severity, Δ): **-0.74**
- Monotonically increasing: **False**
- Reading: no clear difficulty-scaling pattern

> A McNemar test with fewer than ~25 discordant solutions is not trustworthy; check the `reliable` flag in `comparison.json` before quoting a p-value.