#!/usr/bin/env bash
#
# Keep the study going until it is actually finished.
#
#   tmux new -s prm 'bash scripts/lightning/keepalive.sh --limit 100'
#
# run_a100.sh is resumable by design, so the safe response to almost any failure is to run
# it again: finished stages skip and each arm continues from its own predictions.jsonl. This
# wrapper does that automatically. A CUDA OOM on one long solution, a dropped Hub connection
# during the pool download, or the process being killed at hour five then costs a retry
# rather than the night.
#
# Every argument is passed straight through to run_a100.sh, except these:
#   --max-attempts N   give up after N consecutive failures (default 6)
#   --retry-delay S    seconds between attempts (default 60)
#
# It does NOT survive the Studio itself stopping. Lightning shuts an idle Studio down to
# save credits, and that takes the machine with it. Set auto-shutdown to its longest setting
# (or off) in the Studio's compute panel before starting a multi-hour run. If the Studio does
# stop, your files are still there: restart the machine and run this command again.

set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

MAX_ATTEMPTS=6
RETRY_DELAY=60
PASSTHROUGH=()

while [ $# -gt 0 ]; do
  case "$1" in
    --max-attempts) MAX_ATTEMPTS="$2"; shift ;;
    --retry-delay)  RETRY_DELAY="$2"; shift ;;
    -h|--help)      sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)              PASSTHROUGH+=("$1") ;;
  esac
  shift
done

STARTED=$(date +%s)
attempt=1

while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
  log "attempt $attempt of $MAX_ATTEMPTS"

  # Only the first attempt reinstalls; afterwards the environment is already there and
  # pip would just burn minutes re-resolving the same wheels.
  EXTRA=()
  [ "$attempt" -gt 1 ] && EXTRA+=(--skip-install)

  if bash "$PROJECT_ROOT/scripts/lightning/run_a100.sh" \
      ${PASSTHROUGH+"${PASSTHROUGH[@]}"} ${EXTRA+"${EXTRA[@]}"}; then
    elapsed=$(( $(date +%s) - STARTED ))
    log "study finished after $((elapsed / 3600))h $(((elapsed % 3600) / 60))m and $attempt attempt(s)"
    exit 0
  else
    # Capture it here: after the `if` closes, $? is the status of the compound statement
    # (zero when no branch matched), not of the command that actually failed.
    status=$?
  fi

  warn "attempt $attempt failed with status $status"

  # A config or parity error will fail identically forever; retrying it wastes the night.
  # Those exit before any model loads, so a failure inside the first minute is the tell.
  if [ "$attempt" -eq 1 ]; then
    first_run_seconds=$(( $(date +%s) - STARTED ))
    if [ "$first_run_seconds" -lt 60 ]; then
      die "the first attempt failed in ${first_run_seconds}s, which means it never got as
      far as loading the model. That is a setup or config problem, not a transient one, and
      retrying will not fix it. Read the error above."
    fi
  fi

  attempt=$((attempt + 1))
  [ "$attempt" -le "$MAX_ATTEMPTS" ] && {
    log "retrying in ${RETRY_DELAY}s"
    sleep "$RETRY_DELAY"
  }
done

die "gave up after $MAX_ATTEMPTS attempts. Progress is preserved in runs/; re-run this
  command once the underlying problem is fixed and it will continue from where it stopped."
