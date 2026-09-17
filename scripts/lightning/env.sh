# Shared environment for the Lightning AI scripts. Sourced, never executed.
#
# Everything here is overridable from the outside: `HF_HOME=/x bash scripts/lightning/...`
# wins over the defaults below.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PROJECT_ROOT
cd "$PROJECT_ROOT"

# A Lightning Studio persists its home directory across machine switches, so a cache that
# lives inside the repo survives the CPU -> GPU switch. That is what lets you build the
# pool and the index on a cheap CPU machine and keep the 15 GB checkpoint download when
# you move to the GPU.
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf_cache}"
mkdir -p "$HF_HOME"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# The comparison report prints deltas with a Greek delta. A console that is not UTF-8
# (a Windows terminal, a bare `docker exec`) would crash the script on that one character.
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

PY="${PY:-python}"

# Faster checkpoint downloads. huggingface_hub deprecated HF_HUB_ENABLE_HF_TRANSFER in
# favour of Xet; setting the old variable on a current hub build does nothing except print
# a deprecation warning on every invocation, so only the new switch is set here. Older hub
# builds ignore it, which costs nothing.
if [ -z "${HF_XET_HIGH_PERFORMANCE:-}" ]; then
  export HF_XET_HIGH_PERFORMANCE=1
fi

log()  { printf '\033[1;36m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '\033[1;33m[%s] WARN\033[0m %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die()  { printf '\033[1;31m[%s] ERROR\033[0m %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }
