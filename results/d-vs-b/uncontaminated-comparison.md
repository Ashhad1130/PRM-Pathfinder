# Condition A (baseline) vs Condition B (retrieval)

- A: `results/int4-d`
- B: `results/int4-b`
- **Filtered: pool-contaminated solutions removed, per results/contamination.json** (633 solution(s) removed from both conditions). The `n` column below is what survived, so these numbers are not comparable with the unfiltered table.

| Subset | n | OOD | F1 (A) | F1 (B) | Δ F1 | 95% CI on Δ | McNemar p | Sig. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gsm8k | 50 | in-dist | 65.7 | 55.1 | -10.6 | [-27.1, +6.5] | 0.114 | no |
| math | 24 | moderate | 67.0 | 47.3 | -19.7 | [-47.6, +0.0] | 0.221 | no |
| olympiadbench | 49 | severe | 62.7 | 47.5 | -15.2 | [-27.7, -5.5] | 0.023 | yes |
| omnimath | 50 | extreme | 52.5 | 45.7 | -6.8 | [-18.1, +2.7] | 0.267 | no |
| **average** | | | **62.0** | **48.9** | **-13.1** | | | |

## Whole experiment

- Solutions compared: **173** (paired, identical in both arms)
- 95% CI on the average Δ: **[-21.9, -5.3]** — excludes zero
- Pooled McNemar: **p = 0.0005**, 36 discordant (7 fixed by B, 29 broken by B), reliable = True

> Pooling is what makes the paired test usable at this sample size; the per-subset tests above rarely reach the ~25 discordant solutions they need. The pooled test asks whether the two arms decide differently overall, which is weaker than the per-subset F1 deltas it sits beside.

## OOD-severity trend

- Slope per OOD rank: **+1.59 F1**
- Spearman(severity, Δ): **+0.40**
- Monotonically increasing: **False**
- Reading: no clear difficulty-scaling pattern

> A McNemar test with fewer than ~25 discordant solutions is not trustworthy; check the `reliable` flag in `comparison.json` before quoting a p-value.