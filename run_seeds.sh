#!/bin/bash
# Non-interactive multi-seed trainer for LOCAL machines (the workhorse behind
# ./run.sh's "local" path).
#
# Trains every (env, algorithm, seed) combination, pinning each concurrent run to
# one GPU (round-robin), then aggregates the per-seed result CSVs for this run
# tag into:
#     results/<tag>_runs.csv   one row per seed
#     results/<tag>.csv        one row per (env, algorithm): mean/std/95% CI
#
# Every run is stamped with EXP_RUN_TAG=<tag>, which is what scripts/skrl/train.py
# uses to decide where to drop its per-seed CSV — so the aggregation picks up
# exactly this batch.
#
# Usage:
#   ./run_seeds.sh --algorithms ppo --env PointMaze-v1 --seeds 5 --parallel 2 --gpu all
#   ./run_seeds.sh --algorithms "ppo,sac" --env PointMaze-v1 --seeds "0 1 2" -y
#   ./run_seeds.sh --aggregate-only --tag run_20260723_120000
#
# Flags:
#   --algorithms LIST    comma/space list, or 'all'      (default ppo)
#   --env NAME           an Isaac Lab task id, or 'all'  (default PointMaze-v1)
#   --seeds SPEC         a count (5 -> 0..4) or an explicit list ("0 1 7")
#   --parallel N         concurrent runs (default 1 = sequential)
#   --gpu SPEC           index, comma list (0,2), or 'all'  (default all)
#   --tag NAME           run tag / output CSV basename (default run_<timestamp>)
#   --num-timesteps N    override the config's training timesteps
#   --extra "ARGS"       appended verbatim to every train.py call
#   --aggregate-only     skip training, just aggregate --tag
#   -y, --yes            skip the confirmation prompt
#   --detached           internal: set on the nohup re-exec
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -t 1 ]] && command -v tput >/dev/null 2>&1 && [[ "$(tput colors 2>/dev/null || echo 0)" -ge 8 ]]; then
    C_BOLD="$(tput bold)"; C_DIM="$(tput dim)"; C_RESET="$(tput sgr0)"
    C_CYAN="$(tput setaf 6)"; C_GREEN="$(tput setaf 2)"; C_YELLOW="$(tput setaf 3)"; C_RED="$(tput setaf 1)"; C_BLUE="$(tput setaf 4)"
else
    C_BOLD=""; C_DIM=""; C_RESET=""; C_CYAN=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_BLUE=""
fi
_rule() { printf '%s\n' "${C_DIM}────────────────────────────────────────────────────────────${C_RESET}" >&2; }
_header() { echo "" >&2; _rule; printf '%s\n' "${C_BOLD}${C_CYAN}  $1${C_RESET}" >&2; _rule; }
_info()    { printf '%s\n' "  ${C_BLUE}ℹ${C_RESET}  $1" >&2; }
_success() { printf '%s\n' "  ${C_GREEN}✓${C_RESET}  $1" >&2; }
_warn()    { printf '%s\n' "  ${C_YELLOW}⚠${C_RESET}  $1" >&2; }
_error()   { printf '%s\n' "  ${C_RED}✗${C_RESET}  $1" >&2; }

ALL_ALGOS=(ppo sac trpo)
ALL_ENVS=(PointMaze-v1)

ALGOS_ARG="ppo"; ENV_ARG="PointMaze-v1"; SEEDS_ARG="5"; PARALLEL=1; GPU_SPEC="all"
TAG=""; NUM_TIMESTEPS=""; EXTRA=""; YES=0; DETACHED=0; AGG_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --algorithms|--algorithm|--algo) ALGOS_ARG="$2"; shift 2 ;;
        --env) ENV_ARG="$2"; shift 2 ;;
        --seeds) SEEDS_ARG="$2"; shift 2 ;;
        --parallel) PARALLEL="$2"; shift 2 ;;
        --gpu) GPU_SPEC="$2"; shift 2 ;;
        --tag) TAG="$2"; shift 2 ;;
        --num-timesteps) NUM_TIMESTEPS="$2"; shift 2 ;;
        --extra) EXTRA="$2"; shift 2 ;;
        --aggregate-only) AGG_ONLY=1; shift ;;
        -y|--yes) YES=1; shift ;;
        --detached) DETACHED=1; YES=1; shift ;;
        -h|--help) grep -E '^# ?' "$0" | sed -E 's/^# ?//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

TAG="${TAG:-run_$(date '+%Y%m%d_%H%M%S')}"

# ── aggregate-only ────────────────────────────────────────────────────────── #
if [[ "$AGG_ONLY" -eq 1 ]]; then
    python scripts/aggregate_seeds.py --run-tag "$TAG" --out "results/$TAG"
    exit $?
fi

# ── resolve lists ─────────────────────────────────────────────────────────── #
if [[ "$ALGOS_ARG" == "all" ]]; then ALGOS=("${ALL_ALGOS[@]}"); else IFS=', ' read -r -a ALGOS <<< "$ALGOS_ARG"; fi
if [[ "$ENV_ARG" == "all" ]]; then ENVS=("${ALL_ENVS[@]}"); else IFS=', ' read -r -a ENVS <<< "$ENV_ARG"; fi
if [[ "$SEEDS_ARG" =~ ^[0-9]+$ ]]; then mapfile -t SEEDS < <(seq 0 $((SEEDS_ARG - 1))); else IFS=', ' read -r -a SEEDS <<< "$SEEDS_ARG"; fi

# ── GPUs ──────────────────────────────────────────────────────────────────── #
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [[ "$NUM_GPUS" -lt 1 ]]; then _error "nvidia-smi reports 0 GPUs — aborting."; exit 1; fi
if [[ "$GPU_SPEC" == "all" ]]; then mapfile -t GPUS < <(seq 0 $((NUM_GPUS - 1))); else IFS=', ' read -r -a GPUS <<< "$GPU_SPEC"; fi
NG=${#GPUS[@]}
[[ "$NG" -lt 1 ]] && { _error "No GPUs selected."; exit 1; }

TOTAL=$(( ${#ENVS[@]} * ${#ALGOS[@]} * ${#SEEDS[@]} ))
LOG_DIR="logs/seeds/$TAG"

if [[ "$DETACHED" -ne 1 ]]; then
    _header "explorationRL multi-seed run"
    _info "Tag:         ${C_BOLD}$TAG${C_RESET}"
    _info "Algorithms:  ${C_BOLD}${ALGOS[*]}${C_RESET}"
    _info "Env(s):      ${C_BOLD}${ENVS[*]}${C_RESET}"
    _info "Seeds:       ${C_BOLD}${SEEDS[*]}${C_RESET}"
    _info "Total runs:  ${C_BOLD}$TOTAL${C_RESET}   concurrency ${C_BOLD}$PARALLEL${C_RESET}   GPUs ${C_BOLD}${GPUS[*]}${C_RESET}"
    _info "Output CSV:  ${C_BOLD}results/$TAG.csv${C_RESET}"
    _rule
    if [[ "$YES" -ne 1 ]]; then
        printf '%s' "  ${C_YELLOW}?${C_RESET} ${C_BOLD}Launch now (detached via nohup)?${C_RESET} ${C_DIM}[y/N]${C_RESET} " >&2
        read -r CONFIRM
        [[ "$CONFIRM" =~ ^[Yy]$ ]] || { _warn "Aborted."; exit 0; }
    fi
    mkdir -p "$LOG_DIR"
    NOHUP_LOG="$LOG_DIR/nohup.log"
    nohup "$SCRIPT_DIR/run_seeds.sh" \
        --algorithms "$ALGOS_ARG" --env "$ENV_ARG" --seeds "$SEEDS_ARG" \
        --parallel "$PARALLEL" --gpu "$GPU_SPEC" --tag "$TAG" \
        ${NUM_TIMESTEPS:+--num-timesteps "$NUM_TIMESTEPS"} ${EXTRA:+--extra "$EXTRA"} \
        --detached > "$NOHUP_LOG" 2>&1 &
    disown
    _success "Launched (PID $!) — detached, safe to close this terminal."
    _info "Tail: ${C_BOLD}tail -f $NOHUP_LOG${C_RESET}"
    exit 0
fi

# ── detached execution ────────────────────────────────────────────────────── #
mkdir -p "$LOG_DIR"
export EXP_RUN_TAG="$TAG"
echo "[$(date '+%F %T')] starting $TOTAL run(s) on GPUs [${GPUS[*]}], concurrency $PARALLEL"

gpu_i=0
for ENV in "${ENVS[@]}"; do
    for ALGO in "${ALGOS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            GPU="${GPUS[$(( gpu_i % NG ))]}"; gpu_i=$(( gpu_i + 1 ))
            RUN_LOG="$LOG_DIR/${ENV}_${ALGO}_seed${SEED}.log"
            echo "[$(date '+%F %T')] $ENV / $ALGO / seed $SEED  (GPU $GPU)"
            (
                CUDA_VISIBLE_DEVICES=$GPU python scripts/skrl/train.py \
                    --task "$ENV" --algorithm "$ALGO" --seed "$SEED" \
                    ${NUM_TIMESTEPS:+--num_timesteps "$NUM_TIMESTEPS"} $EXTRA \
                    > "$RUN_LOG" 2>&1
                echo "$?" > "$RUN_LOG.status"
            ) &
            while [[ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]]; do wait -n 2>/dev/null || true; done
        done
    done
done
wait

echo "[$(date '+%F %T')] training done — aggregating"
python scripts/aggregate_seeds.py --run-tag "$TAG" --out "results/$TAG" \
    || echo "Aggregation failed — rerun: ./run_seeds.sh --aggregate-only --tag $TAG"
echo "[$(date '+%F %T')] all done. CSV: results/$TAG.csv"
