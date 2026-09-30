# Condition A (baseline) vs Condition B (retrieval)

- A: `results/int4-c`
- B: `results/int4-b`
- **Filtered: pool-contaminated solutions removed, per results/contamination.json** (633 solution(s) removed from both conditions). The `n` column below is what survived, so these numbers are not comparable with the unfiltered table.

| Subset | n | OOD | F1 (A) | F1 (B) | Δ F1 | 95% CI on Δ | McNemar p | Sig. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gsm8k | 50 | in-dist | 54.1 | 55.1 | +1.0 | [-12.8, +15.8] | 1.000 | no |
| math | 24 | moderate | 62.9 | 47.3 | -15.6 | [-44.0, +8.5] | 0.450 | no |
| olympiadbench | 49 | severe | 35.2 | 47.5 | +12.3 | [-1.6, +26.8] | 0.114 | no |
| omnimath | 50 | extreme | 34.2 | 45.7 | +11.6 | [+0.0, +27.1] | 0.149 | no |
| **average** | | | **46.6** | **48.9** | **+2.3** | | | |

## Whole experiment

- Solutions compared: **173** (paired, identical in both arms)
- 95% CI on the average Δ: **[-6.2, +11.2]** — includes zero
- Pooled McNemar: **p = 0.1547**, 40 discordant (25 fixed by B, 15 broken by B), reliable = True

> Pooling is what makes the paired test usable at this sample size; the per-subset tests above rarely reach the ~25 discordant solutions they need. The pooled test asks whether the two arms decide differently overall, which is weaker than the per-subset F1 deltas it sits beside.

## OOD-severity trend

- Slope per OOD rank: **+5.94 F1**
- Spearman(severity, Δ): **+0.60**
- Monotonically increasing: **False**
- Reading: consistent with retrieval fixing distribution shift

> A McNemar test with fewer than ~25 discordant solutions is not trustworthy; check the `reliable` flag in `comparison.json` before quoting a p-value.