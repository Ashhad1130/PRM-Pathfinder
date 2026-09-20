# Condition A (baseline) vs Condition B (retrieval)

- A: `runs/int4-a`
- B: `runs/int4-b`

| Subset | n | OOD | F1 (A) | F1 (B) | Δ F1 | 95% CI on Δ | McNemar p | Sig. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gsm8k | 50 | in-dist | 70.0 | 54.1 | -15.9 | [-35.1, +4.0] | 0.039 | no |
| math | 50 | moderate | 72.7 | 51.6 | -21.1 | [-38.0, -6.2] | 0.034 | yes |
| olympiadbench | 50 | severe | 69.8 | 49.3 | -20.5 | [-35.3, -8.9] | 0.009 | yes |
| omnimath | 50 | extreme | 56.8 | 30.2 | -26.6 | [-42.6, -12.3] | 0.001 | yes |
| **average** | | | **67.3** | **46.3** | **-21.0** | | | |

## Whole experiment

- Solutions compared: **200** (paired, identical in both arms)
- 95% CI on the average Δ: **[-29.5, -13.0]** — excludes zero
- Pooled McNemar: **p = 0.0000**, 59 discordant (8 fixed by B, 51 broken by B), reliable = True

> Pooling is what makes the paired test usable at this sample size; the per-subset tests above rarely reach the ~25 discordant solutions they need. The pooled test asks whether the two arms decide differently overall, which is weaker than the per-subset F1 deltas it sits beside.

## OOD-severity trend

- Slope per OOD rank: **-3.14 F1**
- Spearman(severity, Δ): **-0.80**
- Monotonically increasing: **False**
- Reading: no clear difficulty-scaling pattern

> A McNemar test with fewer than ~25 discordant solutions is not trustworthy; check the `reliable` flag in `comparison.json` before quoting a p-value.