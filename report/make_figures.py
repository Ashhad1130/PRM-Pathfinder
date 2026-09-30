"""Figures and step-level numbers for the report, computed from the committed traces.

Run from the repository root:  python report/make_figures.py
"""

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
ARMS = ["a", "d", "e", "b", "c"]
NAMES = {"a": "A", "d": "D", "e": "E", "b": "B", "c": "C"}
COLOURS = {"a": "#1f77b4", "d": "#ff7f0e", "e": "#9467bd", "b": "#2ca02c", "c": "#d62728"}


def load_traces(arm):
    path = ROOT / "results" / f"int4-{arm}" / "traces.jsonl"
    with path.open(encoding="utf-8") as handle:
        return {(r["uid"], r["step_index"]): r for r in map(json.loads, handle)}


traces = {arm: load_traces(arm) for arm in ARMS}
# Early stopping means later steps are scored in some arms and not others, so every
# comparison below is restricted to steps that all four arms scored.
common = sorted(set.intersection(*(set(t) for t in traces.values())))

stats = {}
for arm in ARMS:
    rows = [traces[arm][key] for key in common]
    gold_ok = [r for r in rows if r["gold_label"] == -1 or r["step_index"] < r["gold_label"]]
    stage1 = [r for r in gold_ok if not (r["math_ok"] and r["consistency_ok"])]
    stage2 = [r for r in gold_ok if r["is_error"] and r["math_ok"] and r["consistency_ok"]]
    passed = [
        r["correctness_prob"]
        for r in rows
        if r["math_ok"] and r["consistency_ok"] and r["correctness_prob"] is not None
    ]
    stats[arm] = {
        "steps": len(rows),
        "gold_correct_steps": len(gold_ok),
        "stage1_flag_rate_all": sum(not (r["math_ok"] and r["consistency_ok"]) for r in rows) / len(rows),
        "math_flag_rate_all": sum(not r["math_ok"] for r in rows) / len(rows),
        "consistency_flag_rate_all": sum(not r["consistency_ok"] for r in rows) / len(rows),
        "false_flag_stage1": len(stage1) / len(gold_ok),
        "false_flag_stage2": len(stage2) / len(gold_ok),
        "p_correct_mean": statistics.mean(passed),
        "p_correct_median": statistics.median(passed),
        "p_correct_below_half": sum(p < 0.5 for p in passed) / len(passed),
        "probs": passed,
    }

OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif"})
fig, (left, right) = plt.subplots(1, 2, figsize=(3.3, 1.75), gridspec_kw={"width_ratios": [1, 1.25]})

x = np.arange(len(ARMS))
s1 = [100 * stats[a]["false_flag_stage1"] for a in ARMS]
s2 = [100 * stats[a]["false_flag_stage2"] for a in ARMS]
left.bar(x, s1, color="0.35", width=0.66, label="stage 1")
left.bar(x, s2, bottom=s1, color=[COLOURS[a] for a in ARMS], width=0.66, label="stage 2")
left.set_xticks(x, [NAMES[a] for a in ARMS])
left.set_ylabel("correct steps flagged (%)")
left.set_title("(a) false flags by stage", fontsize=8)
left.spines[["top", "right"]].set_visible(False)
left.set_ylim(0, 20)
left.text(-0.45, 19.8, "dark: stage 1\ncolour: stage 2", fontsize=6.5, va="top")

for arm in ARMS:
    probs = np.sort(np.asarray(stats[arm]["probs"]))
    ecdf = np.arange(1, probs.size + 1) / probs.size
    right.step(probs, ecdf, where="post", color=COLOURS[arm], lw=1.1, label=NAMES[arm])
right.axvline(0.5, color="0.5", lw=0.6, ls="--")
right.set_xlim(0, 1)
right.set_ylim(0, 1)
right.set_xlabel(r"stage-2 $P(\langle+\rangle)$")
right.set_ylabel("cumulative share")
right.set_title("(b) correctness score", fontsize=8)
right.spines[["top", "right"]].set_visible(False)
right.legend(frameon=False, fontsize=6.5, loc="upper left", handlelength=1)

fig.tight_layout(pad=0.3, w_pad=0.8)
fig.savefig(OUT / "stages.pdf")

print(f"{len(common)} steps scored in every arm")
for arm in ARMS:
    printable = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats[arm].items() if k != "probs"}
    print(NAMES[arm], printable)


# Solution-level cluster bootstrap for the step-level differences against arm A: steps from
# the same solution are not independent, so whole solutions are resampled.
rng = np.random.default_rng(17)
gold_ok_keys = [k for k in common if traces["a"][k]["gold_label"] == -1 or k[1] < traces["a"][k]["gold_label"]]
by_solution = {}
for key in gold_ok_keys:
    by_solution.setdefault(key[0], []).append(key)
solutions = sorted(by_solution)


def stage_flags(arm, keys):
    rows = [traces[arm][k] for k in keys]
    s1 = sum(not (r["math_ok"] and r["consistency_ok"]) for r in rows)
    s2 = sum(r["is_error"] and r["math_ok"] and r["consistency_ok"] for r in rows)
    return s1, s2, len(rows)


print(f"\nbootstrap over {len(solutions)} solutions, {len(gold_ok_keys)} gold-correct steps")
for arm in ["d", "e", "b", "c"]:
    diffs = {"stage1": [], "stage2": []}
    for _ in range(2000):
        draw = rng.choice(len(solutions), size=len(solutions), replace=True)
        keys = [k for i in draw for k in by_solution[solutions[i]]]
        a1, a2, n = stage_flags("a", keys)
        x1, x2, _ = stage_flags(arm, keys)
        diffs["stage1"].append(100 * (x1 - a1) / n)
        diffs["stage2"].append(100 * (x2 - a2) / n)
    a1, a2, n = stage_flags("a", gold_ok_keys)
    x1, x2, _ = stage_flags(arm, gold_ok_keys)
    point = {"stage1": 100 * (x1 - a1) / n, "stage2": 100 * (x2 - a2) / n}
    for stage in ("stage1", "stage2"):
        lo, hi = np.percentile(diffs[stage], [2.5, 97.5])
        print(f"A->{NAMES[arm]} {stage}: {point[stage]:+.1f} pts [{lo:+.1f}, {hi:+.1f}]")
