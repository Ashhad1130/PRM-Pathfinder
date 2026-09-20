# Condition A (baseline) vs Condition B (retrieval)

- A: `runs/int4-c`
- B: `runs/int4-b`
- **Filtered: pool-contaminated solutions removed, per runs/contamination.json** (633 solution(s) removed from both conditions). The `n` column below is what survived, so these numbers are not comparable with the unfiltered table.

| Subset | n | OOD | F1 (A) | F1 (B) | Δ F1 | 95% CI on Δ | McNemar p | Sig. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gsm8k | 50 | in-dist | 54.1 | 54.1 | +0.0 | [-10.3, +10.5] | 0.683 | no |
| math | 24 | moderate | 62.9 | 53.3 | -9.6 | [-37.5, +17.0] | 0.724 | no |
| olympiadbench | 49 | severe | 35.2 | 48.1 | +12.9 | [-1.3, +27.9] | 0.149 | no |
| omnimath | 50 | extreme | 34.2 | 30.2 | -4.0 | [-21.6, +13.7] | 0.803 | no |
| **average** | | | **46.6** | **46.4** | **-0.2** | | | |

## Whole experiment

- Solutions compared: **173** (paired, identical in both arms)
- 95% CI on the average Δ: **[-10.2, +9.1]** — includes zero
- Pooled McNemar: **p = 0.8774**, 42 discordant (22 fixed by B, 20 broken by B), reliable = True

> Pooling is what makes the paired test usable at this sample size; the per-subset tests above rarely reach the ~25 discordant solutions they need. The pooled test asks whether the two arms decide differently overall, which is weaker than the per-subset F1 deltas it sits beside.

## OOD-severity trend

- Slope per OOD rank: **+1.06 F1**
- Spearman(severity, Δ): **+0.00**
- Monotonically increasing: **False**
- Reading: no clear difficulty-scaling pattern

> A McNemar test with fewer than ~25 discordant solutions is not trustworthy; check the `reliable` flag in `comparison.json` before quoting a p-value.