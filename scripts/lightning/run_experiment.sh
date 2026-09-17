#!/usr/bin/env bash
#
# Run the whole A-vs-B experiment on a Lightning AI Studio, end to end.
#
#   bash scripts/lightning/run_experiment.sh --dry-run     # print the plan, touch nothing
#   bash scripts/lightning/run_experiment.sh --limit 400   # ~15 h on a 24 GB GPU
#   bash scripts/lightning/run_experiment.sh               # full benchmark, 3400/subset
#
# Stages, in order:
#   pool           download PathFinder-600K, parse it into data/pool/pool.jsonl
#   index          SBERT -> PCA -> artifacts/index/
#   contamination  measure eval/pool overlap -> runs/contamination.json   (READ IT)
#   a              Condition A, no retrieval            -> runs/<name-a>/
#   b              Condition B, retrieval               -> runs/<name-b>/
#   c              Condition C, RANDOM references       -> runs/<name-c>/
#   compare        deltas, bootstrap CIs, McNemar, plot -> runs/comparison*/
#
# Condition C is the control that decides what B's gain means: same prompt length, same
# guard, references drawn uniformly instead of ranked. B > C is evidence for retrieval;
# B ~ C means the gain was extra context. It costs about as much GPU time as B, so
# --no-control skips it — but then the headline claim has no control behind it.
#
# Every stage is skipped when its output already exists, and both eval stages resume from
# their own predictions.jsonl. So: if the run dies at hour 9, re-run the same command.
#
# Options:
#   --limit N            cap solutions per subset (default: whatever the config says)
#   --subsets "a b"      restrict to these ProcessBench subsets
#   --backend NAME       mock | hf | hf4bit | int4 - applied to BOTH conditions
#   --pool-max-items N   cap the retrieval pool (default: 50000, from the config)
#   --config-a PATH      default configs/baseline.yaml
#   --config-b PATH      default configs/retrieval.yaml
#   --config-c PATH      default configs/control-random.yaml
#   --name-a NAME        default: run.name from config A
#   --name-b NAME        default: run.name from config B
#   --name-c NAME        default: run.name from config C
#   --no-control         skip Condition C (and its comparisons)
#   --stages LIST        comma-separated subset of the stages above
#   --force              rebuild pool/index/contamination even if present
#   --no-resume          start the eval stages from scratch instead of resuming
#   --dry-run            print the plan and exit

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

CONFIG_A="configs/baseline.yaml"
CONFIG_B="configs/retrieval.yaml"
CONFIG_C="configs/control-random.yaml"
NAME_A=""
NAME_B=""
NAME_C=""
LIMIT=""
SUBSETS=""
BACKEND=""
POOL_MAX=""
STAGES="pool,index,contamination,a,b,c,compare"
FORCE=0
RESUME=1
DRY_RUN=0

while [ $# -gt 0 ]; do
  case "$1" in
    --limit)          LIMIT="$2"; shift ;;
    --subsets)        SUBSETS="$2"; shift ;;
    --backend)        BACKEND="$2"; shift ;;
    --pool-max-items) POOL_MAX="$2"; shift ;;
    --config-a)       CONFIG_A="$2"; shift ;;
    --config-b)       CONFIG_B="$2"; shift ;;
    --config-c)       CONFIG_C="$2"; shift ;;
    --name-a)         NAME_A="$2"; shift ;;
    --name-b)         NAME_B="$2"; shift ;;
    --name-c)         NAME_C="$2"; shift ;;
    --no-control)     STAGES="$(printf '%s' "$STAGES" | sed -e 's/,c,/,/' -e 's/^c,//' -e 's/,c$//')" ;;
    --stages)         STAGES="$2"; shift ;;
    --force)          FORCE=1 ;;
    --no-resume)      RESUME=0 ;;
    --dry-run)        DRY_RUN=1 ;;
    -h|--help)        sed -n '2,33p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)                die "unknown option: $1 (try --help)" ;;
  esac
  shift
done

[ -f "$CONFIG_A" ] || die "no such config: $CONFIG_A"
[ -f "$CONFIG_B" ] || die "no such config: $CONFIG_B"

cfg_get() {  # cfg_get <file> <section> <key>
  "$PY" "$PROJECT_ROOT/scripts/lightning/_cfg_get.py" "$1" "$2" "$3"
}

[ -n "$NAME_A" ] || NAME_A="$(cfg_get "$CONFIG_A" run name)"
[ -n "$NAME_B" ] || NAME_B="$(cfg_get "$CONFIG_B" run name)"
[ "$NAME_A" != "$NAME_B" ] || die "A and B would write to the same run directory ($NAME_A)"

has_stage() { case ",$STAGES," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

WANT_CONTROL=0
if has_stage c; then
  [ -f "$CONFIG_C" ] || die "no such config: $CONFIG_C (or pass --no-control)"
  [ -n "$NAME_C" ] || NAME_C="$(cfg_get "$CONFIG_C" run name)"
  if [ "$NAME_C" = "$NAME_A" ] || [ "$NAME_C" = "$NAME_B" ]; then
    die "the control would write to the same run directory as another arm ($NAME_C)"
  fi
  WANT_CONTROL=1
fi

OUT_DIR="$(cfg_get "$CONFIG_A" run out_dir)"; OUT_DIR="${OUT_DIR:-runs}"
RUN_A="$OUT_DIR/$NAME_A"
RUN_B="$OUT_DIR/$NAME_B"
RUN_C="$OUT_DIR/$NAME_C"
POOL_DIR="$(cfg_get "$CONFIG_B" data pool_dir)"; POOL_DIR="${POOL_DIR:-data/pool}"
INDEX_DIR="$(cfg_get "$CONFIG_B" retrieval index_dir)"; INDEX_DIR="${INDEX_DIR:-artifacts/index}"
CONTAM="$OUT_DIR/contamination.json"

LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"

# ------------------------------------------------------------------ A/B parity check
# The whole design rests on exactly one thing differing between the conditions. A silent
# drift in prm.* turns the measured delta into a config artefact, so refuse to start on one.
PARITY_FLAGS=()
[ -n "$BACKEND" ] && PARITY_FLAGS+=(--backend "$BACKEND")
[ -n "$LIMIT" ]   && PARITY_FLAGS+=(--limit "$LIMIT")
if [ -n "$SUBSETS" ]; then
  # shellcheck disable=SC2206  -- deliberate word split
  PARITY_FLAGS+=(--subsets $SUBSETS)
fi
"$PY" "$PROJECT_ROOT/scripts/lightning/_check_parity.py" "$CONFIG_A" "$CONFIG_B" \
  ${PARITY_FLAGS+"${PARITY_FLAGS[@]}"} || die "config parity check failed (A vs B)"

if [ "$WANT_CONTROL" = 1 ]; then
  # The control is only a control if it matches B in everything but reference_mode.
  "$PY" "$PROJECT_ROOT/scripts/lightning/_check_parity.py" "$CONFIG_B" "$CONFIG_C" \
    ${PARITY_FLAGS+"${PARITY_FLAGS[@]}"} || die "config parity check failed (B vs C)"
fi

# ---------------------------------------------------------------------------- plan
EVAL_FLAGS=()
[ -n "$LIMIT" ]   && EVAL_FLAGS+=(--limit "$LIMIT")
[ -n "$BACKEND" ] && EVAL_FLAGS+=(--backend "$BACKEND")
[ "$RESUME" = 1 ] && EVAL_FLAGS+=(--resume)
if [ -n "$SUBSETS" ]; then
  # shellcheck disable=SC2206  -- deliberate word split: --subsets takes nargs="+"
  EVAL_FLAGS+=(--subsets $SUBSETS)
fi

PLAN_SUBSETS="${SUBSETS:-$(cfg_get "$CONFIG_A" data subsets)}"
PLAN_LIMIT="${LIMIT:-$(cfg_get "$CONFIG_A" data limit_per_subset)}"
[ -n "$PLAN_LIMIT" ] || PLAN_LIMIT="none (full benchmark, 3400 solutions)"
PLAN_BACKEND="${BACKEND:-$(cfg_get "$CONFIG_A" prm backend)}"
PLAN_RESUME=$([ "$RESUME" = 1 ] && echo "yes" || echo "no (every eval starts from scratch)")
if [ "$WANT_CONTROL" = 1 ]; then
  PLAN_CONTROL="$CONFIG_C  ->  $RUN_C   (random-reference control)"
else
  PLAN_CONTROL="skipped - B's gain will have no control behind it"
fi

cat <<PLAN

  Retrieval-Augmented PathFinder-PRM - experiment plan
  ---------------------------------------------------
  stages       : $STAGES
  condition A  : $CONFIG_A  ->  $RUN_A
  condition B  : $CONFIG_B  ->  $RUN_B
  condition C  : $PLAN_CONTROL
  subsets      : $PLAN_SUBSETS
  limit/subset : $PLAN_LIMIT
  backend      : $PLAN_BACKEND
  pool         : $POOL_DIR      index: $INDEX_DIR
  resume       : $PLAN_RESUME
  logs         : $LOG_DIR/$STAMP-<stage>.log

PLAN

if [ "$DRY_RUN" = 1 ]; then
  log "dry run - nothing executed"
  exit 0
fi

# --------------------------------------------------------------------------- stages
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
      Fix the cause and re-run the same command - finished stages are skipped and the
      eval stages resume from their predictions.jsonl."
  fi
  elapsed=$(( $(date +%s) - started ))
  TIMINGS+=("$name ${elapsed}s")
  log "stage $name done in ${elapsed}s"
}

skip_stage() {  # skip_stage <name> <reason>
  log "stage $1 skipped - $2 (use --force to redo)"
  TIMINGS+=("$1 skipped")
}

if has_stage pool; then
  if [ "$FORCE" = 0 ] && [ -s "$POOL_DIR/pool.jsonl" ]; then
    skip_stage pool "$POOL_DIR/pool.jsonl already exists"
  else
    POOL_FLAGS=()
    [ -n "$POOL_MAX" ] && POOL_FLAGS+=(--max-items "$POOL_MAX")
    run_stage pool "$PY" scripts/build_pool.py --config "$CONFIG_B" ${POOL_FLAGS+"${POOL_FLAGS[@]}"}
  fi
fi

if has_stage index; then
  if [ "$FORCE" = 0 ] && [ -s "$INDEX_DIR/index.npz" ]; then
    skip_stage index "$INDEX_DIR/index.npz already exists"
  else
    run_stage index "$PY" scripts/build_index.py --config "$CONFIG_B"
  fi
fi

if has_stage contamination; then
  if [ "$FORCE" = 0 ] && [ -s "$CONTAM" ]; then
    skip_stage contamination "$CONTAM already exists"
  else
    run_stage contamination "$PY" scripts/check_contamination.py --config "$CONFIG_B" --out "$CONTAM"
  fi
fi

# The two eval stages are the expensive ones - hours each. Condition A runs first: it is
# the fidelity anchor, and a broken adapter should surface before B's longer prompts burn
# GPU hours.
if has_stage a; then
  run_stage eval-a "$PY" scripts/run_eval.py --config "$CONFIG_A" --name "$NAME_A" \
    ${EVAL_FLAGS+"${EVAL_FLAGS[@]}"}
fi

if has_stage b; then
  run_stage eval-b "$PY" scripts/run_eval.py --config "$CONFIG_B" --name "$NAME_B" \
    ${EVAL_FLAGS+"${EVAL_FLAGS[@]}"}
fi

if has_stage c; then
  run_stage eval-c "$PY" scripts/run_eval.py --config "$CONFIG_C" --name "$NAME_C" \
    ${EVAL_FLAGS+"${EVAL_FLAGS[@]}"}
fi

if has_stage compare; then
  [ -s "$RUN_A/metrics.json" ] || die "no metrics in $RUN_A - run the 'a' stage first"
  [ -s "$RUN_B/metrics.json" ] || die "no metrics in $RUN_B - run the 'b' stage first"

  # The contamination report names the eval solutions that overlap the pool. When it is
  # there, every comparison is also written with those solutions removed from both arms.
  COMPARE_FLAGS=()
  [ -s "$CONTAM" ] && COMPARE_FLAGS+=(--exclude-contaminated "$CONTAM")

  run_stage compare "$PY" scripts/compare_runs.py --a "$RUN_A" --b "$RUN_B" \
    --out "$OUT_DIR/comparison" ${COMPARE_FLAGS+"${COMPARE_FLAGS[@]}"}

  if [ "$WANT_CONTROL" = 1 ] && [ -s "$RUN_C/metrics.json" ]; then
    # A -> C isolates the effect of extra text alone; C -> B isolates the effect of that
    # text being RELEVANT, with prompt length held fixed. B's headline delta is only about
    # retrieval to the extent that the second one survives.
    run_stage compare-control "$PY" scripts/compare_runs.py --a "$RUN_A" --b "$RUN_C" \
      --out "$OUT_DIR/comparison-a-vs-control" ${COMPARE_FLAGS+"${COMPARE_FLAGS[@]}"}
    run_stage compare-relevance "$PY" scripts/compare_runs.py --a "$RUN_C" --b "$RUN_B" \
      --out "$OUT_DIR/comparison-control-vs-b" ${COMPARE_FLAGS+"${COMPARE_FLAGS[@]}"}
  elif [ "$WANT_CONTROL" = 1 ]; then
    warn "no metrics in $RUN_C - skipping the control comparisons"
  fi
fi

# -------------------------------------------------------------------------- summary
echo
log "all requested stages finished"
for t in ${TIMINGS+"${TIMINGS[@]}"}; do printf '    %s\n' "$t"; done

if [ "$WANT_CONTROL" = 1 ] && [ -s "$OUT_DIR/comparison-control-vs-b/comparison.json" ]; then
  CONTROL_RESULTS="
    $OUT_DIR/comparison-a-vs-control/comparison.md - A -> C: what extra text buys on its own
    $OUT_DIR/comparison-control-vs-b/comparison.md - C -> B: what RELEVANCE buys once
                                        prompt length is held fixed. This is the number the
                                        hypothesis actually rests on; if it is flat, B's
                                        headline gain was context, not retrieval.
"
else
  CONTROL_RESULTS="
    (no random-reference control was run, so B's gain has nothing to separate retrieval
     from prompt length - say so in the report, or run the 'c' stage.)
"
fi

cat <<DONE

  Results
    $OUT_DIR/comparison/comparison.md - the headline table: A -> B, per-subset deltas,
                                        CIs, McNemar
    $OUT_DIR/comparison/ood_trend.png - does the gain grow with OOD severity?
    $OUT_DIR/comparison/uncontaminated-comparison.md - the same table with pool-overlapping
                                        eval solutions dropped from both arms
    $CONTAM - overlap between the eval sets and the pool
$CONTROL_RESULTS
  Before quoting a number: state the backend beside it, check "n_resumed" in each
  summary.json before quoting throughput, and read the MATH row against the
  contamination table in docs/EXPERIMENTS.md.
DONE
