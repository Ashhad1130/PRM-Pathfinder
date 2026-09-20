# Condition A (baseline) vs Condition B (retrieval)

- A: `runs/int4-d`
- B: `runs/int4-b`
- **Filtered: pool-contaminated solutions removed, per runs/contamination.json** (633 solution(s) removed from both conditions). The `n` column below is what survived, so these numbers are not comparable with the unfiltered table.

| Subset | n | OOD | F1 (A) | F1 (B) | Δ F1 | 95% CI on Δ | McNemar p | Sig. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gsm8k | 50 | in-dist | 65.7 | 54.1 | -11.6 | [-30.0, +7.7] | 0.096 | no |
| math | 24 | moderate | 67.0 | 53.3 | -13.7 | [-39.7, +8.8] | 0.505 | no |
| olympiadbench | 49 | severe | 62.7 | 48.1 | -14.6 | [-27.7, -3.7] | 0.046 | yes |
| omnimath | 50 | extreme | 52.5 | 30.2 | -22.3 | [-38.5, -6.9] | 0.002 | yes |
| **average** | | | **62.0** | **46.4** | **-15.6** | | | |

## Whole experiment

- Solutions compared: **173** (paired, identical in both arms)
- 95% CI on the average Δ: **[-26.1, -6.9]** — excludes zero
- Pooled McNemar: **p = 0.0000**, 46 discordant (8 fixed by B, 38 broken by B), reliable = True

> Pooling is what makes the paired test usable at this sample size; the per-subset tests above rarely reach the ~25 discordant solutions they need. The pooled test asks whether the two arms decide differently overall, which is weaker than the per-subset F1 deltas it sits beside.

## OOD-severity trend

- Slope per OOD rank: **-3.30 F1**
- Spearman(severity, Δ): **-1.00**
- Monotonically increasing: **False**
- Reading: no clear difficulty-scaling pattern

> A McNemar test with fewer than ~25 discordant solutions is not trustworthy; check the `reliable` flag in `comparison.json` before quoting a p-value.