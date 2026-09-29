# Condition A (baseline) vs Condition B (retrieval)

- A: `results/int4-a`
- B: `results/int4-c`
- **Filtered: pool-contaminated solutions removed, per results/contamination.json** (633 solution(s) removed from both conditions). The `n` column below is what survived, so these numbers are not comparable with the unfiltered table.

| Subset | n | OOD | F1 (A) | F1 (B) | Δ F1 | 95% CI on Δ | McNemar p | Sig. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gsm8k | 50 | in-dist | 70.0 | 54.1 | -15.9 | [-34.1, +1.2] | 0.027 | no |
| math | 24 | moderate | 75.0 | 62.9 | -12.1 | [-33.8, +6.3] | 0.371 | no |
| olympiadbench | 49 | severe | 69.0 | 35.2 | -33.8 | [-49.7, -19.9] | 0.000 | yes |
| omnimath | 50 | extreme | 56.8 | 34.2 | -22.6 | [-40.7, -6.4] | 0.014 | yes |
| **average** | | | **67.7** | **46.6** | **-21.1** | | | |

## Whole experiment

- Solutions compared: **173** (paired, identical in both arms)
- 95% CI on the average Δ: **[-31.0, -13.1]** — excludes zero
- Pooled McNemar: **p = 0.0000**, 54 discordant (7 fixed by B, 47 broken by B), reliable = True

> Pooling is what makes the paired test usable at this sample size; the per-subset tests above rarely reach the ~25 discordant solutions they need. The pooled test asks whether the two arms decide differently overall, which is weaker than the per-subset F1 deltas it sits beside.

## OOD-severity trend

- Slope per OOD rank: **-4.18 F1**
- Spearman(severity, Δ): **-0.60**
- Monotonically increasing: **False**
- Reading: no clear difficulty-scaling pattern

> A McNemar test with fewer than ~25 discordant solutions is not trustworthy; check the `reliable` flag in `comparison.json` before quoting a p-value.