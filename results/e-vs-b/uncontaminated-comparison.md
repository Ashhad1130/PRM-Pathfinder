# Condition A (baseline) vs Condition B (retrieval)

- A: `results/int4-e`
- B: `results/int4-b`
- **Filtered: pool-contaminated solutions removed, per results/contamination.json** (633 solution(s) removed from both conditions). The `n` column below is what survived, so these numbers are not comparable with the unfiltered table.

| Subset | n | OOD | F1 (A) | F1 (B) | Δ F1 | 95% CI on Δ | McNemar p | Sig. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gsm8k | 50 | in-dist | 69.0 | 55.1 | -13.8 | [-26.0, -4.0] | 0.023 | yes |
| math | 24 | moderate | 79.6 | 47.3 | -32.3 | [-59.3, -13.2] | 0.023 | yes |
| olympiadbench | 49 | severe | 62.7 | 47.5 | -15.2 | [-27.1, -5.5] | 0.023 | yes |
| omnimath | 50 | extreme | 52.5 | 45.7 | -6.8 | [-19.4, +3.2] | 0.302 | no |
| **average** | | | **65.9** | **48.9** | **-17.0** | | | |

## Whole experiment

- Solutions compared: **173** (paired, identical in both arms)
- 95% CI on the average Δ: **[-25.1, -10.7]** — excludes zero
- Pooled McNemar: **p = 0.0000**, 36 discordant (5 fixed by B, 31 broken by B), reliable = True

> Pooling is what makes the paired test usable at this sample size; the per-subset tests above rarely reach the ~25 discordant solutions they need. The pooled test asks whether the two arms decide differently overall, which is weaker than the per-subset F1 deltas it sits beside.

## OOD-severity trend

- Slope per OOD rank: **+3.82 F1**
- Spearman(severity, Δ): **+0.40**
- Monotonically increasing: **False**
- Reading: no clear difficulty-scaling pattern

> A McNemar test with fewer than ~25 discordant solutions is not trustworthy; check the `reliable` flag in `comparison.json` before quoting a p-value.