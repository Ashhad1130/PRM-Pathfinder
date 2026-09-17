#!/usr/bin/env bash
#
# One-time setup for a Lightning AI Studio (or any fresh Linux GPU box).
#
#   bash scripts/lightning/setup.sh              # install + verify
#   bash scripts/lightning/setup.sh --flash      # also build flash-attn (slow, optional)
#   bash scripts/lightning/setup.sh --int4       # also install torchao (needed on <20 GB GPUs)
#   bash scripts/lightning/setup.sh --no-smoke   # skip the end-to-end pipeline check
#
# Idempotent: safe to re-run after a machine switch.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

WITH_FLASH=0
WITH_INT4=0
RUN_SMOKE=1
RUN_TESTS=0

while [ $# -gt 0 ]; do
  case "$1" in
    --flash)    WITH_FLASH=1 ;;
    --int4)     WITH_INT4=1 ;;
    --no-smoke) RUN_SMOKE=0 ;;
    --tests)    RUN_TESTS=1 ;;
    -h|--help)  sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)          die "unknown option: $1 (try --help)" ;;
  esac
  shift
done

log "project root : $PROJECT_ROOT"
log "python       : $("$PY" -V 2>&1) at $(command -v "$PY")"
log "HF_HOME      : $HF_HOME"

# ---------------------------------------------------------------------------- hardware
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | while read -r line; do
    log "gpu          : $line"
  done
else
  warn "no nvidia-smi — this box has no GPU. Pool/index build works; the eval stages need one."
fi
log "free disk    : $(df -h "$PROJECT_ROOT" | awk 'NR==2 {print $4}') (need ~40 GB: 15 GB model + 3 GB pool + caches)"

# --------------------------------------------------------------------------- install
log "installing requirements (this takes a few minutes on a cold Studio)"
"$PY" -m pip install --upgrade pip setuptools wheel >/dev/null
"$PY" -m pip install -r requirements.txt
"$PY" -m pip install -e .
"$PY" -m pip install hf_transfer

if [ "$WITH_INT4" = 1 ]; then
  log "installing torchao for prm.backend: int4"
  "$PY" -m pip install torchao
fi

if [ "$WITH_FLASH" = 1 ]; then
  log "building flash-attn — this can take 10+ minutes, and sdpa is a fine fallback"
  "$PY" -m pip install flash-attn --no-build-isolation || \
    warn "flash-attn failed to build; the adapter falls back to sdpa automatically"
fi

# ---------------------------------------------------------------------------- verify
log "verifying torch / CUDA"
"$PY" - <<'PYEOF'
import torch, transformers
print(f"  torch        : {torch.__version__}  cuda={torch.version.cuda}")
print(f"  transformers : {transformers.__version__}")
print(f"  cuda avail   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    vram = props.total_memory / 1024**3
    print(f"  device       : {props.name}  {vram:.1f} GiB  sm_{props.major}{props.minor}")
    # A 7B in bf16 is ~15.2 GB of weights; leave room for activations and the KV cache.
    if vram >= 20:
        rec = "hf (bf16) — configs/baseline.yaml + configs/retrieval.yaml as-is"
    elif vram >= 17:
        rec = "hf (bf16), but keep prm.max_input_tokens at 4096 — it is tight"
    else:
        rec = "int4 (torchao) — re-run setup.sh with --int4, then pass --backend int4"
    print(f"  recommended  : prm.backend: {rec}")
else:
    print("  recommended  : CPU-only box — build the pool and index here, switch to a GPU to evaluate")
PYEOF

# Hub access. The model and the pool are both public, but an unauthenticated Studio hits
# rate limits on a 15 GB download often enough to be worth flagging early.
if "$PY" -c "from huggingface_hub import HfApi; HfApi().whoami()" >/dev/null 2>&1; then
  log "huggingface  : logged in"
else
  warn "not logged in to Hugging Face. Public repos still work, but if downloads stall run: hf auth login"
fi

# ------------------------------------------------------------------------- pipeline
if [ "$RUN_TESTS" = 1 ]; then
  log "running pytest"
  "$PY" -m pytest
fi

if [ "$RUN_SMOKE" = 1 ]; then
  log "running the end-to-end smoke pipeline (mock PRM, no GPU, ~5 s)"
  "$PY" scripts/smoke.py
fi

cat <<'MSG'

Setup done. Next:

  bash scripts/lightning/run_experiment.sh --dry-run      # see the plan and the stages
  bash scripts/lightning/run_experiment.sh --limit 400    # ~15 h, detects the reported effects
  bash scripts/lightning/run_experiment.sh                # full 3400/subset

Long runs: start it under tmux so a dropped browser tab does not kill it —
  tmux new -s rapfprm 'bash scripts/lightning/run_experiment.sh --limit 400'
  tmux attach -t rapfprm     # detach with ctrl-b d
MSG
