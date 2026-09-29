# Condition A (baseline) vs Condition B (retrieval)

- A: `results/int4-a`
- B: `results/int4-e`
- **Filtered: pool-contaminated solutions removed, per results/contamination.json** (633 solution(s) removed from both conditions). The `n` column below is what survived, so these numbers are not comparable with the unfiltered table.

| Subset | n | OOD | F1 (A) | F1 (B) | Δ F1 | 95% CI on Δ | McNemar p | Sig. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gsm8k | 50 | in-dist | 70.0 | 69.0 | -1.0 | [-15.4, +12.4] | 1.000 | no |
| math | 24 | moderate | 75.0 | 79.6 | +4.6 | [+0.0, +17.6] | 1.000 | no |
| olympiadbench | 49 | severe | 69.0 | 62.7 | -6.3 | [-14.6, +0.0] | 0.248 | no |
| omnimath | 50 | extreme | 56.8 | 52.5 | -4.3 | [-17.8, +2.5] | 1.000 | no |
| **average** | | | **67.7** | **65.9** | **-1.7** | | | |

## Whole experiment

- Solutions compared: **173** (paired, identical in both arms)
- 95% CI on the average Δ: **[-7.0, +3.7]** — includes zero
- Pooled McNemar: **p = 0.3865**, 12 discordant (4 fixed by B, 8 broken by B), reliable = False

> Pooling is what makes the paired test usable at this sample size; the per-subset tests above rarely reach the ~25 discordant solutions they need. The pooled test asks whether the two arms decide differently overall, which is weaker than the per-subset F1 deltas it sits beside.

## OOD-severity trend

- Slope per OOD rank: **-2.06 F1**
- Spearman(severity, Δ): **-0.60**
- Monotonically increasing: **False**
- Reading: no clear difficulty-scaling pattern

> A McNemar test with fewer than ~25 discordant solutions is not trustworthy; check the `reliable` flag in `comparison.json` before quoting a p-value.