# Condition A (baseline) vs Condition B (retrieval)

- A: `runs/int4-a`
- B: `runs/int4-b`
- **Filtered: pool-contaminated solutions removed, per runs/contamination.json** (633 solution(s) removed from both conditions). The `n` column below is what survived, so these numbers are not comparable with the unfiltered table.

| Subset | n | OOD | F1 (A) | F1 (B) | Δ F1 | 95% CI on Δ | McNemar p | Sig. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gsm8k | 50 | in-dist | 70.0 | 54.1 | -15.9 | [-35.1, +4.0] | 0.039 | no |
| math | 24 | moderate | 75.0 | 53.3 | -21.7 | [-50.8, +5.8] | 0.228 | no |
| olympiadbench | 49 | severe | 69.0 | 48.1 | -20.9 | [-35.6, -8.2] | 0.009 | yes |
| omnimath | 50 | extreme | 56.8 | 30.2 | -26.6 | [-42.6, -12.3] | 0.001 | yes |
| **average** | | | **67.7** | **46.4** | **-21.3** | | | |

## Whole experiment

- Solutions compared: **173** (paired, identical in both arms)
- 95% CI on the average Δ: **[-32.3, -11.6]** — excludes zero
- Pooled McNemar: **p = 0.0000**, 52 discordant (7 fixed by B, 45 broken by B), reliable = True

> Pooling is what makes the paired test usable at this sample size; the per-subset tests above rarely reach the ~25 discordant solutions they need. The pooled test asks whether the two arms decide differently overall, which is weaker than the per-subset F1 deltas it sits beside.

## OOD-severity trend

- Slope per OOD rank: **-3.12 F1**
- Spearman(severity, Δ): **-0.80**
- Monotonically increasing: **False**
- Reading: no clear difficulty-scaling pattern

> A McNemar test with fewer than ~25 discordant solutions is not trustworthy; check the `reliable` flag in `comparison.json` before quoting a p-value.