#!/usr/bin/env bash
#
# The whole study, end to end, on one big-GPU Lightning Studio. Written for a 40 GB A100,
# where the 7B fits in bf16 and the numbers are directly comparable to the published 69.5.
#
#   bash scripts/lightning/run_a100.sh --pilot     # 25/subset, ~1 h, check before committing
#   bash scripts/lightning/run_a100.sh             # 100/subset, the default
#   bash scripts/lightning/run_a100.sh --full      # the entire benchmark, 3400/subset
#
# Run it under tmux. A closed browser tab kills a foreground process and these are hours:
#
#   tmux new -s prm 'bash scripts/lightning/run_a100.sh'
#   tmux attach -t prm          # detach with ctrl-b then d
#
# What it does, in order:
#   install      pip install the project, verify torch sees the GPU
#   verify       prompt parity against the model card, then the mock-backend smoke test
#   pool         download PathFinder-600K and parse it into a retrieval pool
#   index        SBERT -> PCA -> artifacts/index/
#   contamination  measure eval/pool overlap and record the affected uids
#   A            baseline, no references
#   B            retrieved references, gold labels as <+> / <->
#   C            random references, gold labels        (controls for prompt length)
#   D            retrieved references, no labels       (controls for the labels)
#   E            retrieved references, labels as words (controls for the tokens)
#   compare      seven contrasts, each also recomputed without contaminated eval items
#   report       the multi-arm figure, the verdict profile, and a summary table
#
# Everything is resumable. If it dies at hour six, run the identical command again: finished
# stages are skipped and each arm continues from its own predictions.jsonl.
#
# Options:
#   --pilot            25 solutions/subset
#   --limit N          N solutions/subset (default 100)
#   --full             the entire benchmark
#   --subsets "a b"    restrict to these ProcessBench subsets
#   --arms "A B C"     run only these arms (default "A B C D E")
#   --skip-install     the environment is already set up
#   --no-resume        start every arm from scratch
#   --dry-run          print the plan and exit

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

LIMIT=100
SUBSETS=""
ARMS="A B C D E"
SKIP_INSTALL=0
RESUME=1
DRY_RUN=0

while [ $# -gt 0 ]; do
  case "$1" in
    --pilot)        LIMIT=25 ;;
    --limit)        LIMIT="$2"; shift ;;
    --full)         LIMIT="" ;;
    --subsets)      SUBSETS="$2"; shift ;;
    --arms)         ARMS="$2"; shift ;;
    --skip-install) SKIP_INSTALL=1 ;;
    --no-resume)    RESUME=0 ;;
    --dry-run)      DRY_RUN=1 ;;
    -h|--help)      sed -n '2,45p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)              die "unknown option: $1 (try --help)" ;;
  esac
  shift
done

# ---- the five arms -------------------------------------------------------------------
# Validate up front. config_for() is called inside $(...) further down, and a `die` in a
# subshell only kills the subshell — the parent would sail on with an empty config path and
# report success over a study it never ran.
for arm in $ARMS; do
  case "$arm" in
    A|B|C|D|E) ;;
    *) die "unknown arm '$arm' in --arms \"$ARMS\" (valid: A B C D E)" ;;
  esac
done

# label -> config : run name : one-line description
config_for() { case "$1" in
  A) echo "configs/baseline.yaml" ;;
  B) echo "configs/retrieval.yaml" ;;
  C) echo "configs/control-random.yaml" ;;
  D) echo "configs/ablation-nolabels.yaml" ;;
  E) echo "configs/ablation-wordlabels.yaml" ;;
  *) die "unknown arm: $1 (valid: A B C D E)" ;;
esac; }

name_for() { "$PY" "$PROJECT_ROOT/scripts/lightning/_cfg_get.py" "$(config_for "$1")" run name; }

describe() { case "$1" in
  A) echo "baseline, no references" ;;
  B) echo "retrieved references, labels as <+>/<->" ;;
  C) echo "random references, labels as <+>/<->" ;;
  D) echo "retrieved references, no labels" ;;
  E) echo "retrieved references, labels as words" ;;
esac; }

has_arm() { case " $ARMS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

OUT_DIR="runs"
LOG_DIR="$OUT_DIR/logs"
CONTAM="$OUT_DIR/contamination.json"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"

EVAL_FLAGS=()
[ -n "$LIMIT" ] && EVAL_FLAGS+=(--limit "$LIMIT")
[ "$RESUME" = 1 ] && EVAL_FLAGS+=(--resume)
if [ -n "$SUBSETS" ]; then
  # shellcheck disable=SC2206  -- deliberate word split: --subsets takes nargs="+"
  EVAL_FLAGS+=(--subsets $SUBSETS)
fi

PARITY_FLAGS=()
[ -n "$LIMIT" ] && PARITY_FLAGS+=(--limit "$LIMIT")
if [ -n "$SUBSETS" ]; then
  # shellcheck disable=SC2206
  PARITY_FLAGS+=(--subsets $SUBSETS)
fi

# ---- plan ----------------------------------------------------------------------------
cat <<PLAN

  Retrieval-Augmented PathFinder-PRM - full study
  ----------------------------------------------
  arms         : $ARMS
  subsets      : ${SUBSETS:-gsm8k math olympiadbench omnimath}
  limit/subset : ${LIMIT:-none (full benchmark, 3400 solutions)}
  precision    : bf16 (from the configs; all arms identical)
  resume       : $([ "$RESUME" = 1 ] && echo yes || echo "no - every arm restarts")
  logs         : $LOG_DIR/$STAMP-<stage>.log

PLAN

for arm in $ARMS; do
  printf '  %s  %-42s -> %s/%s\n' "$arm" "$(describe "$arm")" "$OUT_DIR" "$(name_for "$arm")"
done
echo

if [ "$DRY_RUN" = 1 ]; then
  log "dry run - nothing executed"
  exit 0
fi

TIMINGS=()

run_stage() {  # run_stage <name> <command...>
  local name="$1"; shift
  local logfile="$LOG_DIR/$STAMP-$name.log"
  local started elapsed
  started=$(date +%s)
  log "stage $name starting  (log: $logfile)"
  if ! "$@" 2>&1 | tee "$logfile"; then
    elapsed=$(( $(date +%s) - started ))
    die "stage '$name' failed after ${elapsed}s. Full output: $logfile
      Fix the cause and re-run the identical command - finished stages are skipped and
      each arm resumes from its own predictions.jsonl."
  fi
  elapsed=$(( $(date +%s) - started ))
  TIMINGS+=("$name ${elapsed}s")
  log "stage $name done in ${elapsed}s"
}

# ---- install and verify ---------------------------------------------------------------
if [ "$SKIP_INSTALL" = 0 ]; then
  run_stage install bash "$PROJECT_ROOT/scripts/lightning/setup.sh" --no-smoke
fi

"$PY" - <<'PYEOF' || die "no usable GPU - this script assumes one; see docs/LIGHTNING.md"
import sys
import torch

if not torch.cuda.is_available():
    print("  torch cannot see a GPU.")
    sys.exit(1)
props = torch.cuda.get_device_properties(0)
vram = props.total_memory / 1024**3
print(f"  GPU: {props.name}, {vram:.1f} GiB")
if vram < 20:
    print(
        "  This card is too small for bf16 (a 7B needs ~15.2 GB of weights before\n"
        "  activations). Either switch to a 24 GB+ machine, or use the int4 path:\n"
        "    bash scripts/lightning/run_experiment.sh --limit 50 \\\n"
        "        --config-a configs/pilot-int4.yaml \\\n"
        "        --config-b configs/pilot-int4-retrieval.yaml \\\n"
        "        --config-c configs/pilot-int4-random.yaml"
    )
    sys.exit(1)
PYEOF

run_stage verify-prompt "$PY" scripts/verify_model_interface.py
run_stage smoke "$PY" scripts/smoke.py

# ---- data ------------------------------------------------------------------------------
POOL_DIR="$("$PY" "$PROJECT_ROOT/scripts/lightning/_cfg_get.py" configs/retrieval.yaml data pool_dir)"
INDEX_DIR="$("$PY" "$PROJECT_ROOT/scripts/lightning/_cfg_get.py" configs/retrieval.yaml retrieval index_dir)"

if [ -s "$POOL_DIR/pool.jsonl" ]; then
  log "stage pool skipped - $POOL_DIR/pool.jsonl already exists"
else
  run_stage pool "$PY" scripts/build_pool.py --config configs/retrieval.yaml
fi

if [ -s "$INDEX_DIR/index.npz" ]; then
  log "stage index skipped - $INDEX_DIR/index.npz already exists"
else
  run_stage index "$PY" scripts/build_index.py --config configs/retrieval.yaml
fi

if [ -s "$CONTAM" ]; then
  log "stage contamination skipped - $CONTAM already exists"
else
  run_stage contamination "$PY" scripts/check_contamination.py \
    --config configs/retrieval.yaml --out "$CONTAM"
fi

# ---- parity ----------------------------------------------------------------------------
# Every arm must match every other one outside the single key that defines it. Checked
# before any GPU time is spent, because a drifted prm.* turns each delta into an artefact.
check_pair() {  # check_pair <arm-a> <arm-b> [extra flags...]
  local a="$1" b="$2"; shift 2
  "$PY" "$PROJECT_ROOT/scripts/lightning/_check_parity.py" \
    "$(config_for "$a")" "$(config_for "$b")" \
    ${PARITY_FLAGS+"${PARITY_FLAGS[@]}"} "$@" \
    || die "config parity failed between arms $a and $b"
}

log "checking arm parity"
has_arm A && has_arm B && check_pair A B
has_arm B && has_arm C && check_pair B C
has_arm B && has_arm D && check_pair B D --allow prompt.include_reference_labels
has_arm B && has_arm E && check_pair B E --allow prompt.label_style

# ---- the arms ---------------------------------------------------------------------------
# A first: it is the fidelity anchor, and a broken adapter should surface before the
# reference-carrying arms spend three times as long per solution.
for arm in $ARMS; do
  run_stage "eval-$arm" "$PY" scripts/run_eval.py --config "$(config_for "$arm")" \
    ${EVAL_FLAGS+"${EVAL_FLAGS[@]}"}
done

# ---- contrasts ---------------------------------------------------------------------------
COMPARE_FLAGS=()
[ -s "$CONTAM" ] && COMPARE_FLAGS+=(--exclude-contaminated "$CONTAM")

contrast() {  # contrast <from-arm> <to-arm> <out-suffix> <question>
  local from="$1" to="$2" suffix="$3" question="$4"
  has_arm "$from" && has_arm "$to" || return 0
  local a b
  a="$OUT_DIR/$(name_for "$from")"
  b="$OUT_DIR/$(name_for "$to")"
  [ -s "$a/metrics.json" ] && [ -s "$b/metrics.json" ] || {
    warn "skipping $from->$to ($question): an arm has no metrics"
    return 0
  }
  run_stage "cmp-$suffix" "$PY" scripts/compare_runs.py --a "$a" --b "$b" \
    --out "$OUT_DIR/comparison-$suffix" ${COMPARE_FLAGS+"${COMPARE_FLAGS[@]}"}
}

contrast A B "a-vs-b"   "the headline delta"
contrast A C "a-vs-c"   "what extra text costs on its own"
contrast C B "c-vs-b"   "what relevance buys, length held fixed"
contrast A D "a-vs-d"   "what unlabelled references cost"
contrast D B "d-vs-b"   "what the labels cost"
contrast E B "e-vs-b"   "what the verdict tokens cost, judgement held fixed"
contrast D E "d-vs-e"   "what the judgement costs, written as words"

# ---- report ------------------------------------------------------------------------------
PLOT_ARGS=()
ANALYSIS_ARGS=()
for arm in $ARMS; do
  PLOT_ARGS+=("$arm: $(describe "$arm")=$OUT_DIR/$(name_for "$arm")")
  ANALYSIS_ARGS+=("$arm=$OUT_DIR/$(name_for "$arm")")
done

run_stage figure "$PY" scripts/plot_arms.py --runs "${PLOT_ARGS[@]}" \
  --out "$OUT_DIR/comparison/arms.png" \
  --title "PathFinder-PRM-7B (bf16), ProcessBench, ${LIMIT:-full} per subset"

run_stage verdicts "$PY" scripts/error_analysis.py --runs "${ANALYSIS_ARGS[@]}" \
  --out "$OUT_DIR/comparison/verdict_profile.json"

# ---- summary -------------------------------------------------------------------------------
echo
log "all requested stages finished"
for t in ${TIMINGS+"${TIMINGS[@]}"}; do printf '    %s\n' "$t"; done
echo

"$PY" "$PROJECT_ROOT/scripts/lightning/_summary.py" "$OUT_DIR" $ARMS
