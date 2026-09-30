# Condition A (baseline) vs Condition B (retrieval)

- A: `results/int4-c`
- B: `results/int4-b`

| Subset | n | OOD | F1 (A) | F1 (B) | Δ F1 | 95% CI on Δ | McNemar p | Sig. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gsm8k | 50 | in-dist | 54.1 | 55.1 | +1.0 | [-12.8, +15.8] | 1.000 | no |
| math | 50 | moderate | 54.3 | 50.3 | -4.0 | [-19.9, +10.3] | 0.773 | no |
| olympiadbench | 50 | severe | 34.5 | 49.0 | +14.4 | [+1.2, +29.0] | 0.070 | yes |
| omnimath | 50 | extreme | 34.2 | 45.7 | +11.6 | [+0.0, +27.1] | 0.149 | no |
| **average** | | | **44.3** | **50.0** | **+5.8** | | | |

## Whole experiment

- Solutions compared: **200** (paired, identical in both arms)
- 95% CI on the average Δ: **[-1.1, +12.8]** — includes zero
- Pooled McNemar: **p = 0.1048**, 46 discordant (29 fixed by B, 17 broken by B), reliable = True

> Pooling is what makes the paired test usable at this sample size; the per-subset tests above rarely reach the ~25 discordant solutions they need. The pooled test asks whether the two arms decide differently overall, which is weaker than the per-subset F1 deltas it sits beside.

## OOD-severity trend

- Slope per OOD rank: **+5.00 F1**
- Spearman(severity, Δ): **+0.60**
- Monotonically increasing: **False**
- Reading: consistent with retrieval fixing distribution shift

> A McNemar test with fewer than ~25 discordant solutions is not trustworthy; check the `reliable` flag in `comparison.json` before quoting a p-value.