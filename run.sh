#!/bin/bash
# One interactive entry point for multi-seed training — LOCAL or SLURM CLUSTER.
#
# A thin front-end that asks where to run and the usual knobs (algorithm, env,
# seeds, concurrency, GPUs), then dispatches:
#
#   LOCAL   → hands the answers to ./run_seeds.sh (which detaches via nohup,
#             trains every (env, algo, seed), and aggregates the result CSV).
#   CLUSTER → additionally asks the SLURM knobs (partition, GPUs/job, wall-time,
#             account/qos, env-activation line), generates a self-contained
#             sbatch job that runs the SAME (env, algo, seed) loop on the
#             allocated GPUs and aggregates at the end, then submits it.
#
# Both paths stamp every run with the same EXP_RUN_TAG, so the aggregator picks
# up exactly this batch and writes:
#   results/<tag>_runs.csv   one row per seed
#   results/<tag>.csv        one row per (env, algorithm): mean/std/CI over seeds
#
# NOTE: for a hyperparameter SWEEP (W&B, space in search/configs/), use
# search/search.sh instead — this script trains fixed configs across seeds.
#
# Usage:
#   ./run.sh                 # fully interactive
#   ./run.sh --help          # this header
#
# Every prompt has a sensible default in [brackets]; press Enter to accept it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

case "${1:-}" in
    -h|--help) grep -E '^# ?' "$0" | sed -E 's/^# ?//'; exit 0 ;;
esac

# ── Pretty output ─────────────────────────────────────────────────────────── #
if [[ -t 1 ]] && command -v tput >/dev/null 2>&1 && [[ "$(tput colors 2>/dev/null || echo 0)" -ge 8 ]]; then
    C_BOLD="$(tput bold)"; C_DIM="$(tput dim)"; C_RESET="$(tput sgr0)"
    C_CYAN="$(tput setaf 6)"; C_GREEN="$(tput setaf 2)"; C_YELLOW="$(tput setaf 3)"
    C_RED="$(tput setaf 1)"; C_BLUE="$(tput setaf 4)"; C_MAGENTA="$(tput setaf 5)"
else
    C_BOLD=""; C_DIM=""; C_RESET=""; C_CYAN=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_BLUE=""; C_MAGENTA=""
fi
_rule() { printf '%s\n' "${C_DIM}────────────────────────────────────────────────────────────${C_RESET}" >&2; }
_header() { echo "" >&2; _rule; printf '%s\n' "${C_BOLD}${C_CYAN}  $1${C_RESET}" >&2; _rule; }
_info()    { printf '%s\n' "  ${C_BLUE}ℹ${C_RESET}  $1" >&2; }
_success() { printf '%s\n' "  ${C_GREEN}✓${C_RESET}  $1" >&2; }
_warn()    { printf '%s\n' "  ${C_YELLOW}⚠${C_RESET}  $1" >&2; }
_error()   { printf '%s\n' "  ${C_RED}✗${C_RESET}  $1" >&2; }
_kv()      { printf '  %s%-18s%s %s\n' "${C_DIM}" "$1" "${C_RESET}" "${C_BOLD}$2${C_RESET}" >&2; }

prompt_choice() {
    local prompt_text="$1"; shift
    local opts=("$@") opt
    printf '\n%s\n' "  ${C_MAGENTA}?${C_RESET} ${C_BOLD}$prompt_text${C_RESET}" >&2
    PS3="  ${C_DIM}› ${C_RESET}"
    select opt in "${opts[@]}"; do [[ -n "$opt" ]] && { echo "$opt"; return; }; _warn "Invalid choice, try again."; done
}
prompt_default() {
    local text="$1" def="$2" reply
    printf '\n%s' "  ${C_MAGENTA}?${C_RESET} ${C_BOLD}${text}${C_RESET} ${C_DIM}[${def}]${C_RESET}: " >&2
    read -r reply; echo "${reply:-$def}"
}

_header "explorationRL run launcher"

# ── Where to run ──────────────────────────────────────────────────────────── #
MODE=$(prompt_choice "Where should these runs execute?" \
    "local  (this machine, GPUs here)" \
    "cluster (submit to SLURM)")
case "$MODE" in local*) MODE=local ;; cluster*) MODE=cluster ;; esac
_success "Mode: ${C_BOLD}$MODE${C_RESET}"

# ── Shared knobs ──────────────────────────────────────────────────────────── #
ALL_ALGOS=(ppo sac trpo)
ALGO_ARG=$(prompt_choice "Algorithm to run (or 'all'):" "${ALL_ALGOS[@]}" all)

ISAAC_ENVS=(PointMaze-v1)
ENV_ARG=$(prompt_choice "Env (or 'all'):" "${ISAAC_ENVS[@]}" all)

SEEDS_ARG=$(prompt_default "Seeds — a count (5 → 0..4) or an explicit list" "5")

SEEDING=$(prompt_choice "Run seeds in parallel or sequentially?" \
    "sequential (one run at a time)" \
    "parallel   (several runs at once)")
if [[ "$SEEDING" == parallel* ]]; then
    PARALLEL=$(prompt_default "How many concurrent runs?" "2")
else
    PARALLEL=1
fi

NUM_TIMESTEPS=$(prompt_default "Override training timesteps (blank = use the config's)" "")
EXTRA=$(prompt_default "Extra args appended verbatim to every train.py call (blank = none)" "")
DEFAULT_TAG="run_$(date '+%Y%m%d_%H%M%S')"
TAG=$(prompt_default "Run tag / output CSV basename" "$DEFAULT_TAG")

# ═══════════════════════════════════════════════════════════════════════════ #
# LOCAL: delegate to run_seeds.sh (owns detach + GPU round-robin + aggregate)
# ═══════════════════════════════════════════════════════════════════════════ #
if [[ "$MODE" == local ]]; then
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    [[ "$NUM_GPUS" -lt 1 ]] && { _error "nvidia-smi reports 0 GPUs — aborting."; exit 1; }
    mapfile -t GPU_NAMES < <(nvidia-smi -L 2>/dev/null | sed -E 's/^GPU ([0-9]+): (.*) \(UUID.*/\1: \2/')
    _info "Detected GPUs:"; printf '      %s\n' "${GPU_NAMES[@]}" >&2
    GPU_SPEC=$(prompt_default "GPUs to use — index, comma list (0,2,3), or 'all'" "all")

    _header "Local run summary"
    _kv "Tag" "$TAG"; _kv "Algorithm(s)" "$ALGO_ARG"; _kv "Env" "$ENV_ARG"
    _kv "Seeds" "$SEEDS_ARG"; _kv "Concurrency" "$PARALLEL"; _kv "GPUs" "$GPU_SPEC"
    [[ -n "$NUM_TIMESTEPS" ]] && _kv "Timesteps" "$NUM_TIMESTEPS"
    [[ -n "$EXTRA" ]] && _kv "Extra" "$EXTRA"
    _rule
    CONFIRM=$(prompt_default "Launch now (detached via nohup)? [y/N]" "N")
    [[ "$CONFIRM" =~ ^[Yy]$ ]] || { _warn "Aborted."; exit 0; }

    exec ./run_seeds.sh \
        --algorithms "$ALGO_ARG" --env "$ENV_ARG" --seeds "$SEEDS_ARG" \
        --parallel "$PARALLEL" --gpu "$GPU_SPEC" --tag "$TAG" \
        ${NUM_TIMESTEPS:+--num-timesteps "$NUM_TIMESTEPS"} ${EXTRA:+--extra "$EXTRA"} \
        -y
fi

# ═══════════════════════════════════════════════════════════════════════════ #
# CLUSTER: gather SLURM knobs, generate an sbatch job, submit it
# ═══════════════════════════════════════════════════════════════════════════ #
need() { command -v "$1" >/dev/null 2>&1 || { _error "'$1' not found — are you on a SLURM login node?"; exit 1; }; }
need sbatch; need sinfo

if [[ "$ENV_ARG" == "all" ]]; then ENVS=("${ISAAC_ENVS[@]}"); else ENVS=("$ENV_ARG"); fi
if [[ "$ALGO_ARG" == "all" ]]; then ALGOS=("${ALL_ALGOS[@]}"); else IFS=', ' read -r -a ALGOS <<< "$ALGO_ARG"; fi
if [[ "$SEEDS_ARG" =~ ^[0-9]+$ ]]; then mapfile -t SEEDS < <(seq 0 $((SEEDS_ARG - 1))); else IFS=', ' read -r -a SEEDS <<< "$SEEDS_ARG"; fi

# ── Partition discovery ───────────────────────────────────────────────────── #
_header "Partition / GPU discovery"
echo "" >&2
{
    printf 'PARTITION|AVAIL|TIMELIMIT|GRES\n'
    sinfo -h -o "%P|%a|%l|%G" 2>/dev/null | sed 's/\*//' | sort -u
} | column -t -s '|' 2>/dev/null | sed 's/^/  /' >&2 || _warn "sinfo unavailable."
_rule
PARTITION=$(prompt_default "SLURM partition" "gpu")
[[ -z "$PARTITION" ]] && { _error "No partition chosen."; exit 1; }

GPUS_PER_JOB=$(prompt_default "GPUs per job (--gres=gpu:N; seeds round-robin over them)" "1")
GPU_TYPE=$(prompt_default "Pin GPU model (e.g. a100; blank = any on the partition)" "")
WALLTIME=$(prompt_default "Wall-time HH:MM:SS" "24:00:00")
ACCOUNT=$(prompt_default "SLURM account (blank = none)" "")
QOS=$(prompt_default "SLURM qos (blank = none)" "")
ACTIVATE=$(prompt_default "Env activation line" "conda activate env_isaaclab")

CPUS_PER_GPU=8
CPUS_PER_TASK=$(( CPUS_PER_GPU * GPUS_PER_JOB ))
MEM="$(( 8 * PARALLEL ))G"
GRES="gpu:${GPU_TYPE:+$GPU_TYPE:}$GPUS_PER_JOB"
TOTAL=$(( ${#ENVS[@]} * ${#ALGOS[@]} * ${#SEEDS[@]} ))
LOG_DIR="logs/seeds/$TAG"
JOB_SCRIPT="$LOG_DIR/job.sbatch"

_header "Cluster run summary"
_kv "Tag" "$TAG"; _kv "Algorithm(s)" "${ALGOS[*]}"; _kv "Env(s)" "${ENVS[*]}"
_kv "Seeds" "${SEEDS[*]}"; _kv "Total runs" "$TOTAL"; _kv "Partition" "$PARTITION"
_kv "GRES" "$GRES"; _kv "Concurrency" "$PARALLEL  (seeds at once, over $GPUS_PER_JOB GPU/job)"
_kv "Wall-time" "$WALLTIME"; _kv "CPUs/task" "$CPUS_PER_TASK"; _kv "Memory" "$MEM"
[[ -n "$ACCOUNT" ]] && _kv "Account" "$ACCOUNT"
[[ -n "$QOS" ]]     && _kv "QOS" "$QOS"
_kv "Activate" "$ACTIVATE"; _kv "Output CSV" "results/$TAG.csv"
_rule
CONFIRM=$(prompt_default "Submit this job now? [y/N]" "N")
[[ "$CONFIRM" =~ ^[Yy]$ ]] || { _warn "Aborted — nothing submitted."; exit 0; }

mkdir -p "$LOG_DIR"

# ── Generate the self-contained sbatch job ────────────────────────────────── #
{
    echo "#!/bin/bash"
    echo "#SBATCH --job-name=erl-$TAG"
    echo "#SBATCH --partition=$PARTITION"
    echo "#SBATCH --gres=$GRES"
    echo "#SBATCH --cpus-per-task=$CPUS_PER_TASK"
    echo "#SBATCH --time=$WALLTIME"
    echo "#SBATCH --output=$LOG_DIR/slurm_%j.out"
    echo "#SBATCH --mem=$MEM"
    [[ -n "$ACCOUNT" ]] && echo "#SBATCH --account=$ACCOUNT"
    [[ -n "$QOS" ]]     && echo "#SBATCH --qos=$QOS"
    cat <<EOF
set -uo pipefail
cd "$SCRIPT_DIR"
${ACTIVATE:+$ACTIVATE}

export EXP_RUN_TAG="$TAG"
LOG_DIR="$LOG_DIR"
PARALLEL=$PARALLEL

# GPUs SLURM gave THIS job (read the allocation, not nvidia-smi, so workers never
# spill onto GPUs the job doesn't own).
if [[ -n "\${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra JOB_GPUS <<< "\$CUDA_VISIBLE_DEVICES"
else
    mapfile -t JOB_GPUS < <(seq 0 \$(( ${GPUS_PER_JOB} - 1 )))
fi
NG=\${#JOB_GPUS[@]}
[[ "\$NG" -lt 1 ]] && { echo "No GPUs allocated — aborting."; exit 1; }
echo "[\$(date '+%F %T')] job \$SLURM_JOB_ID on GPUs [\${JOB_GPUS[*]}]"

ENVS=(${ENVS[*]})
ALGOS=(${ALGOS[*]})
SEEDS=(${SEEDS[*]})

gpu_i=0
for ENV in "\${ENVS[@]}"; do
    for ALGO in "\${ALGOS[@]}"; do
        for SEED in "\${SEEDS[@]}"; do
            GPU="\${JOB_GPUS[\$(( gpu_i % NG ))]}"; gpu_i=\$(( gpu_i + 1 ))
            RUN_LOG="\$LOG_DIR/\${ENV}_\${ALGO}_seed\${SEED}.log"
            echo "[\$(date '+%F %T')] \$ENV / \$ALGO / seed \$SEED  (GPU \$GPU)"
            (
                CUDA_VISIBLE_DEVICES=\$GPU python scripts/skrl/train.py \\
                    --task "\$ENV" --algorithm "\$ALGO" --seed "\$SEED" \\
                    ${NUM_TIMESTEPS:+--num_timesteps $NUM_TIMESTEPS} ${EXTRA} \\
                    > "\$RUN_LOG" 2>&1
                echo "\$?" > "\$RUN_LOG.status"
            ) &
            while [[ "\$(jobs -rp | wc -l)" -ge "\$PARALLEL" ]]; do wait -n 2>/dev/null || true; done
        done
    done
done
wait

echo "[\$(date '+%F %T')] training done — aggregating"
python scripts/aggregate_seeds.py --run-tag "$TAG" --out "results/$TAG" \\
    || echo "Aggregation failed — rerun: ./run_seeds.sh --aggregate-only --tag $TAG"
echo "[\$(date '+%F %T')] all done. CSV: results/$TAG.csv"
EOF
} > "$JOB_SCRIPT"
chmod +x "$JOB_SCRIPT"

_header "Submitting"
_info "Job script: ${C_DIM}$JOB_SCRIPT${C_RESET}"
SUBMIT_OUT=$(sbatch "$JOB_SCRIPT" 2>&1) || { _error "sbatch failed: $SUBMIT_OUT"; exit 1; }
_success "$SUBMIT_OUT"
_info "Monitor:  ${C_BOLD}squeue --me${C_RESET}   |   log: ${C_BOLD}$LOG_DIR/slurm_*.out${C_RESET}"
_info "CSV lands at: ${C_BOLD}results/$TAG.csv${C_RESET}"
echo "" >&2
