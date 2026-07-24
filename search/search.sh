#!/bin/bash
# One interactive entry point for W&B hyperparameter sweeps — LOCAL or CLUSTER.
#
# Prompts for where to run (local / SLURM cluster), the algorithm, the env, and
# how many self-restarting search agents to spawn, previews the exact metric and
# hyperparameter ranges, then launches:
#
#   LOCAL   → N self-restarting `wandb agent` workers, detached via nohup so the
#             search survives closing the terminal.
#   CLUSTER → creates the sweep once on the login node, then submits self-
#             resubmitting sbatch jobs that each run agents on their allocated
#             GPUs and renew themselves before the wall-time kill.
#
# The searched space is NOT defined here — it lives in search/configs/, one yaml
# per algorithm (see its README.md). search/build_sweep.py turns the chosen one
# into a W&B sweep yaml. To change what is searched, edit search/configs/.
#
# Usage:
#   ./search.sh                                   # fully interactive
#   ./search.sh --mode local  --algorithm ppo  --env PointMaze-v1 --num-agents 3 --gpu 0 -y
#   ./search.sh --mode cluster --algorithm sac --env PointMaze-v1 --partition gpu \
#       --num-jobs 4 --gpus-per-job 1 --agents-per-gpu 2 --time 12:00:00 -y
#   ./search.sh --stop <log-dir-name>             # halt a running CLUSTER search
#
# Flags (all optional; anything omitted is prompted for):
#   --mode local|cluster   where to run
#   --algorithm NAME       a stem in search/configs/ (ppo, sac, trpo, ...)
#   --env NAME             an Isaac Lab task id (e.g. PointMaze-v1)
#   --num-agents N         LOCAL: parallel wandb agents (default 3)
#   --gpu N                LOCAL: pin agents to this GPU (default: round-robin)
#   --method M             bayes (default) | grid | random
#   --timeout T            per-run watchdog (default 24h)
#   --runs-per-agent N     stop each agent after N runs (default 0 = unbounded)
#   --partition NAME       CLUSTER: SLURM partition
#   --num-jobs N           CLUSTER: independent sbatch jobs (default: prompted)
#   --gpus-per-job N       CLUSTER: --gres=gpu:N per job (default 1)
#   --agents-per-gpu N     CLUSTER: wandb agents per GPU (default 1)
#   --time HH:MM:SS        CLUSTER: per-job wall-time (default 24:00:00)
#   --account NAME         CLUSTER: --account (omitted if unset)
#   --qos NAME             CLUSTER: --qos (omitted if unset)
#   --activate CMD         CLUSTER: env-activation line (e.g. 'conda activate env_isaaclab')
#   --stop TAG             CLUSTER: STOP + scancel a running search (log-dir name)
#   -y, --yes              skip the confirmation prompt
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="$SCRIPT_DIR/configs"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Pretty output (colors only on a real terminal; nohup logs stay clean) ─── #
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
    select opt in "${opts[@]}"; do [[ -n "$opt" ]] && { echo "$opt"; return; }; _warn "Invalid choice."; done
}
prompt_default() { local text="$1" def="$2" reply; printf '\n%s' "  ${C_MAGENTA}?${C_RESET} ${C_BOLD}${text}${C_RESET} ${C_DIM}[${def}]${C_RESET}: " >&2; read -r reply; echo "${reply:-$def}"; }

# ── Defaults / flag parsing ───────────────────────────────────────────────── #
MODE=""; ALGORITHM=""; ENV_ARG=""; NUM_AGENTS=""; GPU_ARG=""; METHOD="bayes"
PER_RUN_TIMEOUT=24h; RUNS_PER_AGENT=0; YES=0; DETACHED=0; STOP_TAG=""
PARTITION=""; NUM_JOBS=""; GPUS_PER_JOB=1; AGENTS_PER_GPU=""; WALLTIME="24:00:00"
ACCOUNT=""; QOS=""; ACTIVATE="conda activate env_isaaclab"; CPUS_PER_GPU=8; MEM=""

# Isaac Lab task ids that can be swept (keep in sync with build_sweep.ISAACLAB_ENVS).
ISAAC_ENVS=("PointMaze-v1")

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --algorithm|--algo) ALGORITHM="$2"; shift 2 ;;
        --env) ENV_ARG="$2"; shift 2 ;;
        --num-agents) NUM_AGENTS="$2"; shift 2 ;;
        --gpu) GPU_ARG="$2"; shift 2 ;;
        --method) METHOD="$2"; shift 2 ;;
        --timeout) PER_RUN_TIMEOUT="$2"; shift 2 ;;
        --runs-per-agent) RUNS_PER_AGENT="$2"; shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --num-jobs) NUM_JOBS="$2"; shift 2 ;;
        --gpus-per-job) GPUS_PER_JOB="$2"; shift 2 ;;
        --agents-per-gpu) AGENTS_PER_GPU="$2"; shift 2 ;;
        --time) WALLTIME="$2"; shift 2 ;;
        --account) ACCOUNT="$2"; shift 2 ;;
        --qos) QOS="$2"; shift 2 ;;
        --activate) ACTIVATE="$2"; shift 2 ;;
        --stop) STOP_TAG="$2"; shift 2 ;;
        -y|--yes) YES=1; shift ;;
        --detached) DETACHED=1; YES=1; shift ;;  # internal: set on the nohup re-exec
        -h|--help) grep -E '^# ?' "$0" | sed -E 's/^# ?//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ── --stop: halt a running cluster search ─────────────────────────────────── #
if [[ -n "$STOP_TAG" ]]; then
    command -v scancel >/dev/null 2>&1 || { _error "'scancel' not found — not on a SLURM node?"; exit 1; }
    LOG_DIR="$SCRIPT_DIR/logs/$STOP_TAG"; JOBNAME="$STOP_TAG"
    [[ -f "$LOG_DIR/jobname" ]] && JOBNAME="$(cat "$LOG_DIR/jobname")"
    if [[ -d "$LOG_DIR" ]]; then touch "$LOG_DIR/STOP"; _success "Wrote STOP sentinel: $LOG_DIR/STOP"
    else _warn "No log dir $LOG_DIR — will still scancel by name '$JOBNAME'."; fi
    scancel --name="$JOBNAME" || true
    _success "Requested cancellation of all jobs named '$JOBNAME'."
    exit 0
fi

if [[ "$DETACHED" -ne 1 ]]; then _header "explorationRL sweep launcher"; fi

# ── Mode ──────────────────────────────────────────────────────────────────── #
if [[ -z "$MODE" ]]; then
    MODE=$(prompt_choice "Where should this sweep run?" "local (this machine)" "cluster (SLURM)")
fi
case "$MODE" in local*) MODE=local ;; cluster*) MODE=cluster ;; esac
[[ "$DETACHED" -ne 1 ]] && _success "Mode: ${C_BOLD}$MODE${C_RESET}"

# ── Algorithm (globbed from configs/) ─────────────────────────────────────── #
mapfile -t ALGOS < <(find "$CONFIG_DIR" -maxdepth 1 -name '*.yaml' -printf '%f\n' | sed 's/\.yaml$//' | sort)
[[ "${#ALGOS[@]}" -eq 0 ]] && { _error "No algorithm configs in $CONFIG_DIR."; exit 1; }
[[ -z "$ALGORITHM" ]] && ALGORITHM=$(prompt_choice "Algorithm to sweep:" "${ALGOS[@]}")
[[ ! -f "$CONFIG_DIR/$ALGORITHM.yaml" ]] && { _error "No config '$ALGORITHM.yaml'. Available: ${ALGOS[*]}"; exit 1; }
[[ "$DETACHED" -ne 1 ]] && _success "Algorithm: ${C_BOLD}$ALGORITHM${C_RESET}"

# ── Env ───────────────────────────────────────────────────────────────────── #
[[ -z "$ENV_ARG" ]] && ENV_ARG=$(prompt_choice "Env to sweep:" "${ISAAC_ENVS[@]}")
_found=0; for e in "${ISAAC_ENVS[@]}"; do [[ "$e" == "$ENV_ARG" ]] && _found=1; done
[[ "$_found" -ne 1 ]] && { _error "Unknown env '$ENV_ARG'. Known: ${ISAAC_ENVS[*]}"; exit 1; }
[[ "$DETACHED" -ne 1 ]] && _success "Env: ${C_BOLD}$ENV_ARG${C_RESET}"

cd "$REPO_DIR"

# ── Preview the generated sweep ───────────────────────────────────────────── #
if [[ "$DETACHED" -ne 1 ]]; then
    _header "Generated sweep ($ENV_ARG, $ALGORITHM)"
    if ! _SWEEP_PREVIEW="$(python search/build_sweep.py --algorithm "$ALGORITHM" --env "$ENV_ARG" --method "$METHOD")"; then
        _error "build_sweep.py failed — check search/configs/$ALGORITHM.yaml."; exit 1
    fi
    if [[ -n "$C_CYAN" ]]; then echo "$_SWEEP_PREVIEW" | sed -E "s/^([A-Za-z_.]+):/${C_CYAN}\1${C_RESET}:/" >&2
    else echo "$_SWEEP_PREVIEW" >&2; fi
    _rule
fi

# ═══════════════════════════════════════════════════════════════════════════ #
# LOCAL
# ═══════════════════════════════════════════════════════════════════════════ #
if [[ "$MODE" == local ]]; then
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    [[ "$NUM_GPUS" -lt 1 ]] && { _error "nvidia-smi reports 0 GPUs — aborting."; exit 1; }
    if [[ -z "$NUM_AGENTS" ]]; then NUM_AGENTS=$(prompt_default "Parallel wandb agents (each keeps searching)" "3"); fi
    if [[ -z "$GPU_ARG" && "$DETACHED" -ne 1 ]]; then
        mapfile -t GPU_NAMES < <(nvidia-smi -L 2>/dev/null | sed -E 's/^GPU ([0-9]+): (.*) \(UUID.*/\1: \2/')
        GPU_CHOICE=$(prompt_choice "GPU to pin agents to (or 'all' to round-robin):" "${GPU_NAMES[@]}" "all")
        [[ "$GPU_CHOICE" != "all" ]] && GPU_ARG="${GPU_CHOICE%%:*}"
    fi

    if [[ "$DETACHED" -ne 1 ]]; then
        _header "Local sweep summary"
        _kv "Algorithm" "$ALGORITHM"; _kv "Env" "$ENV_ARG"; _kv "Agents" "$NUM_AGENTS"
        _kv "GPU" "${GPU_ARG:-all (round-robin)}"; _kv "Method" "$METHOD"
        _rule
        if [[ "$YES" -ne 1 ]]; then
            CONFIRM=$(prompt_default "Launch this sweep now, detached via nohup? [y/N]" "N")
            [[ "$CONFIRM" =~ ^[Yy]$ ]] || { _warn "Aborted."; exit 0; }
        fi
        # Detach: re-exec under nohup with every choice resolved as flags.
        mkdir -p "$SCRIPT_DIR/logs"
        NOHUP_LOG="$SCRIPT_DIR/logs/nohup_${ALGORITHM}_${ENV_ARG}_$(date '+%Y%m%d_%H%M%S').log"
        nohup "$SCRIPT_DIR/search.sh" --mode local --algorithm "$ALGORITHM" --env "$ENV_ARG" \
            --num-agents "$NUM_AGENTS" --method "$METHOD" --timeout "$PER_RUN_TIMEOUT" \
            --runs-per-agent "$RUNS_PER_AGENT" ${GPU_ARG:+--gpu "$GPU_ARG"} \
            --detached > "$NOHUP_LOG" 2>&1 &
        disown
        _success "Launched (PID $!) — detached, safe to close this terminal."
        _info "Tail: ${C_BOLD}tail -f $NOHUP_LOG${C_RESET}"
        exit 0
    fi

    # ── detached launch ───────────────────────────────────────────────────── #
    RUN_TS="$(date '+%Y%m%d_%H%M%S')"
    LOG_DIR="$SCRIPT_DIR/logs/${ALGORITHM}_${ENV_ARG}_${RUN_TS}"
    mkdir -p "$LOG_DIR"
    SWEEP_YAML="$LOG_DIR/sweep.yaml"
    python search/build_sweep.py --algorithm "$ALGORITHM" --env "$ENV_ARG" --method "$METHOD" --out "$SWEEP_YAML" > /dev/null
    SWEEP_INIT_OUTPUT=$(wandb sweep "$SWEEP_YAML" 2>&1) || true
    SWEEP_ID=$(echo "$SWEEP_INIT_OUTPUT" | grep -oE "wandb agent .*" | awk '{print $3}' || true)
    [[ -z "$SWEEP_ID" ]] && { echo "Failed to create sweep:"; echo "$SWEEP_INIT_OUTPUT"; exit 1; }
    echo "$SWEEP_ID" > "$LOG_DIR/sweep_id"
    echo "[$(date '+%F %T')] sweep $SWEEP_ID — $NUM_AGENTS agent(s)" >> "$LOG_DIR/launch.log"

    for j in $(seq 1 "$NUM_AGENTS"); do
        GPU="${GPU_ARG:-$(( (j - 1) % NUM_GPUS ))}"
        LOGFILE="$LOG_DIR/agent_${j}.log"
        (
            run_count=0
            while true; do
                echo "[$(date '+%F %T')] (re)starting agent for $SWEEP_ID (GPU $GPU)" >> "$LOGFILE"
                CUDA_VISIBLE_DEVICES=$GPU timeout "$PER_RUN_TIMEOUT" wandb agent --count 1 "$SWEEP_ID" >> "$LOGFILE" 2>&1
                run_count=$((run_count + 1))
                if [[ "$RUNS_PER_AGENT" -gt 0 && "$run_count" -ge "$RUNS_PER_AGENT" ]]; then
                    echo "[$(date '+%F %T')] reached $RUNS_PER_AGENT runs — stopping." >> "$LOGFILE"; break
                fi
                sleep 5
            done
        ) &
    done
    echo "[$(date '+%F %T')] all $NUM_AGENTS agent(s) launched." >> "$LOG_DIR/launch.log"
    wait
    exit 0
fi

# ═══════════════════════════════════════════════════════════════════════════ #
# CLUSTER
# ═══════════════════════════════════════════════════════════════════════════ #
need() { command -v "$1" >/dev/null 2>&1 || { _error "'$1' not found — are you on a SLURM login node?"; exit 1; }; }
need sbatch; need sinfo; need squeue; need scontrol

# ── Partition discovery (best-effort) ─────────────────────────────────────── #
_header "Partition / GPU discovery"
{
    printf 'PARTITION|AVAIL|TIMELIMIT|GRES\n'
    sinfo -h -o "%P|%a|%l|%G" 2>/dev/null | sed 's/\*//' | sort -u
} | column -t -s '|' 2>/dev/null | sed 's/^/  /' >&2 || _warn "sinfo unavailable."
_rule
[[ -z "$PARTITION" ]] && PARTITION=$(prompt_default "SLURM partition" "gpu")
[[ -z "$PARTITION" ]] && { _error "No partition chosen."; exit 1; }
[[ -z "$NUM_JOBS" ]]  && NUM_JOBS=$(prompt_default "Independent sbatch jobs to submit" "2")
[[ -z "$AGENTS_PER_GPU" ]] && AGENTS_PER_GPU=$(prompt_default "wandb agents per GPU" "1")
[[ "$AGENTS_PER_GPU" =~ ^[0-9]+$ && "$AGENTS_PER_GPU" -ge 1 ]] || { _error "agents-per-gpu must be a positive integer."; exit 1; }

# Every worker authenticates to W&B as this account (shared-home ~/.netrc).
if ! wandb login --verify >/dev/null 2>&1; then
    _error "wandb is not authenticated here — run 'wandb login' first."; exit 1
fi

CPUS_PER_TASK=$(( CPUS_PER_GPU * GPUS_PER_JOB ))
GRES="gpu:${GPUS_PER_JOB}"
AGENTS_PER_JOB=$(( GPUS_PER_JOB * AGENTS_PER_GPU ))
TOTAL_WORKERS=$(( NUM_JOBS * AGENTS_PER_JOB ))

_header "Cluster sweep summary"
_kv "Algorithm" "$ALGORITHM"; _kv "Env" "$ENV_ARG"; _kv "Partition" "$PARTITION"
_kv "Jobs" "$NUM_JOBS"; _kv "GPUs/job" "$GPUS_PER_JOB"; _kv "Agents/GPU" "$AGENTS_PER_GPU"
_kv "Total workers" "$TOTAL_WORKERS"; _kv "Wall-time" "$WALLTIME  (self-resubmits on USR1@180)"
[[ -n "$ACCOUNT" ]] && _kv "Account" "$ACCOUNT"; [[ -n "$QOS" ]] && _kv "QOS" "$QOS"
_kv "Activate" "$ACTIVATE"
_rule
if [[ "$YES" -ne 1 ]]; then
    CONFIRM=$(prompt_default "Submit these jobs now? [y/N]" "N")
    [[ "$CONFIRM" =~ ^[Yy]$ ]] || { _warn "Aborted — nothing submitted."; exit 0; }
fi

# ── Create the sweep once on the login node ───────────────────────────────── #
RUN_TS="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="$SCRIPT_DIR/logs/${ALGORITHM}_${ENV_ARG}_${RUN_TS}"
mkdir -p "$LOG_DIR"
SWEEP_YAML="$LOG_DIR/sweep.yaml"
python search/build_sweep.py --algorithm "$ALGORITHM" --env "$ENV_ARG" --method "$METHOD" --out "$SWEEP_YAML" > /dev/null
SWEEP_INIT_OUTPUT=$(wandb sweep "$SWEEP_YAML" 2>&1) || true
SWEEP_ID=$(echo "$SWEEP_INIT_OUTPUT" | grep -oE "wandb agent .*" | awk '{print $3}' || true)
[[ -z "$SWEEP_ID" ]] && { _error "Failed to create sweep."; echo "$SWEEP_INIT_OUTPUT" | sed 's/^/      /' >&2; exit 1; }
JOBNAME="erl-${ALGORITHM}-${ENV_ARG}-${RUN_TS}"; JOBNAME="${JOBNAME:0:60}"
echo "$JOBNAME" > "$LOG_DIR/jobname"; echo "$SWEEP_ID" > "$LOG_DIR/sweep_id"
_success "Sweep created: ${C_BOLD}$SWEEP_ID${C_RESET}"

# ── Generate a self-resubmitting worker script and submit N copies ────────── #
WB_ENTITY=""; WB_PROJECT=""
if [[ "$SWEEP_ID" == */*/* ]]; then WB_ENTITY="${SWEEP_ID%%/*}"; _rest="${SWEEP_ID#*/}"; WB_PROJECT="${_rest%%/*}"; fi
JOB_SCRIPT="$LOG_DIR/job.sbatch"
{
    echo "#!/bin/bash"
    echo "#SBATCH --job-name=$JOBNAME"
    echo "#SBATCH --partition=$PARTITION"
    echo "#SBATCH --gres=$GRES"
    echo "#SBATCH --cpus-per-task=$CPUS_PER_TASK"
    echo "#SBATCH --time=$WALLTIME"
    echo "#SBATCH --signal=B:USR1@180"
    echo "#SBATCH --output=$LOG_DIR/slurm_%j.out"
    [[ -n "$ACCOUNT" ]] && echo "#SBATCH --account=$ACCOUNT"
    [[ -n "$QOS" ]]     && echo "#SBATCH --qos=$QOS"
    [[ -n "$MEM" ]]     && echo "#SBATCH --mem=$MEM"
    cat <<EOF
set -uo pipefail
cd "$REPO_DIR"
${ACTIVATE:+$ACTIVATE}

STOP_FILE="$LOG_DIR/STOP"
SWEEP_ID="$SWEEP_ID"
AGENTS_PER_GPU=$AGENTS_PER_GPU
PER_RUN_TIMEOUT="$PER_RUN_TIMEOUT"
RUNS_PER_AGENT=$RUNS_PER_AGENT
${WB_ENTITY:+export WANDB_ENTITY="$WB_ENTITY"}
${WB_PROJECT:+export WANDB_PROJECT="$WB_PROJECT"}

# One job → one replacement on the pre-kill USR1 (unless a STOP sentinel exists).
resubmit() {
    if [[ -f "\$STOP_FILE" ]]; then echo "[\$(date '+%F %T')] USR1 but STOP present — not resubmitting."; exit 0; fi
    echo "[\$(date '+%F %T')] USR1 — resubmitting before wall-time kill."
    sbatch "\$0" || echo "[\$(date '+%F %T')] WARNING: resubmit failed."
    exit 0
}
trap resubmit USR1

# Pin each worker to a GPU SLURM actually allocated THIS job.
if [[ -n "\${CUDA_VISIBLE_DEVICES:-}" ]]; then IFS=',' read -ra JOB_GPUS <<< "\$CUDA_VISIBLE_DEVICES"
elif [[ -n "\${SLURM_GPUS_ON_NODE:-}" ]]; then mapfile -t JOB_GPUS < <(seq 0 \$(( SLURM_GPUS_ON_NODE - 1 )))
else mapfile -t JOB_GPUS < <(seq 0 \$(( \$(nvidia-smi -L 2>/dev/null | wc -l) - 1 ))); fi
[[ "\${#JOB_GPUS[@]}" -lt 1 ]] && { echo "No GPUs allocated — aborting."; exit 1; }
echo "[\$(date '+%F %T')] job \$SLURM_JOB_ID: GPUs [\${JOB_GPUS[*]}] × \$AGENTS_PER_GPU agent(s) on \$SWEEP_ID"

for gpu in "\${JOB_GPUS[@]}"; do
    for (( a=0; a<AGENTS_PER_GPU; a++ )); do
        (
            run_count=0
            while true; do
                [[ -f "\$STOP_FILE" ]] && break
                echo "[\$(date '+%F %T')] gpu \$gpu agent \$a: (re)starting wandb agent"
                CUDA_VISIBLE_DEVICES=\$gpu timeout "\$PER_RUN_TIMEOUT" wandb agent --count 1 "\$SWEEP_ID"
                run_count=\$(( run_count + 1 ))
                if [[ "\$RUNS_PER_AGENT" -gt 0 && "\$run_count" -ge "\$RUNS_PER_AGENT" ]]; then break; fi
                [[ -f "\$STOP_FILE" ]] && break
                sleep 5
            done
        ) &
    done
done
wait
EOF
} > "$JOB_SCRIPT"
chmod +x "$JOB_SCRIPT"

_header "Submitting $NUM_JOBS job(s)"
for (( k=1; k<=NUM_JOBS; k++ )); do
    SUBMIT_OUT=$(sbatch "$JOB_SCRIPT" 2>&1) || { _error "sbatch failed: $SUBMIT_OUT"; continue; }
    _success "  $SUBMIT_OUT"
done
_info "Stop this search: ${C_BOLD}$0 --stop $(basename "$LOG_DIR")${C_RESET}"
_info "Monitor: ${C_BOLD}squeue --me${C_RESET}  |  W&B project explorationRL-Search"
echo "" >&2
