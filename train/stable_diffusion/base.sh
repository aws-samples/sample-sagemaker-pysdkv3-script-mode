#!/bin/bash
#
# SageMaker Accelerate Training Script
# - Fine-tunes a model using Accelerate + (optionally) DeepSpeed Zero3
# - Optionally runs local inference and an evaluation harness
#
# Usage:
#   ./sm_accelerate_train.sh --config <CONFIG_YAML> [--num_process <N>]
#
# Notes:
#   --num_process is per-machine GPU process count (maps to accelerate --num_processes).
#   This script is safe under `set -euo pipefail` and avoids unbound-var issues.

set -euo pipefail

#need to address bug in local SM Studio container
#export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

############################################
# Configuration (use absolute paths)
############################################
readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults (can be overridden by CLI)
NUM_GPUS=""                # per machine (input); we’ll also compute totals
CONFIG_PATH=""
RUN_EVAL=false
MLFLOW_ARN=""              # SageMaker managed MLflow tracking server ARN (optional)
MLFLOW_EXPERIMENT_NAME=""  # MLflow experiment name (optional)

# Repo-local assets (absolute)
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"
ACCELERATE_CONFIG="" #"${SCRIPT_DIR}/configs/accelerate/ds_zero3.yaml"
TRAINING_SCRIPT="" #"${SCRIPT_DIR}/sft.py"
INFERENCE_SCRIPT="${SCRIPT_DIR}/inference.py"
EVAL_HARNESS_DIR="${SCRIPT_DIR}/evaluation_harness"
MERGE_SCRIPT="${SCRIPT_DIR}/utils/merge_adapter_weights.py"

############################################
# Logging
############################################
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1" >&2; }

############################################
# Usage
############################################
show_usage() {
    cat << EOF
Usage: $SCRIPT_NAME --config <CONFIG_YAML> [--num_process <N>] [--run-eval]

Arguments:
  --config CONFIG_YAML    Path to training configuration YAML file

Options:
  --num_process N         Per-machine process count (usually = GPUs per node)
  --run-eval              Run local inference + evaluation harness after training
  --help, -h              Show this help message

Examples:
  $SCRIPT_NAME --config ${SCRIPT_DIR}/recipes/llama_sft.yaml
  $SCRIPT_NAME --config ${SCRIPT_DIR}/configs/custom.yaml --num_process 4 --run-eval
EOF
}

############################################
# Validators
############################################
validate_file_exists() {
    [[ -f "$1" ]] || { log_error "$2 not found: $1"; exit 1; }
}
validate_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]] || { log_error "$2 must be a positive integer, got: $1"; exit 1; }
}

############################################
# Argument parsing
############################################
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --num_process) NUM_GPUS="${2:-}"; shift 2 ;;
            --config)      CONFIG_PATH="${2:-}"; shift 2 ;;
            --accelerate-config)   ACCELERATE_CONFIG="${2:-}"; shift 2 ;;
            --training-script)    TRAINING_SCRIPT="${2:-}"; shift 2 ;;
            --mlflow-arn)          MLFLOW_ARN="${2:-}"; shift 2 ;;
            --mlflow-experiment-name)  MLFLOW_EXPERIMENT_NAME="${2:-}"; shift 2 ;;
            --help|-h)     show_usage; exit 0 ;;
            *)             log_error "Unknown argument: $1"; show_usage; exit 1 ;;
        esac
    done
}

############################################
# GPU discovery (safe with set -u)
############################################
resolve_num_gpus() {
    if [[ -n "${NUM_GPUS:-}" ]]; then
        NUM_GPUS="$(printf '%s' "$NUM_GPUS" | tr -d '[:space:]')"   # trim whitespace
    elif [[ -n "${SM_NUM_GPUS:-}" ]]; then
        NUM_GPUS="$(printf '%s' "$SM_NUM_GPUS" | tr -d '[:space:]')" # trim whitespace
    elif command -v nvidia-smi &> /dev/null; then
        NUM_GPUS="$(nvidia-smi -L | wc -l | tr -d '[:space:]')"      # trim whitespace
        [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]
    else
        log_error "Unable to determine GPU count. Please specify --num_process."
        exit 1
    fi
    
    USE_DISCRETE_GPU_FOR_VALIDATION=$(yq -r '.use_discrete_gpu_for_validation' "$CONFIG_PATH")

    if [[ "$USE_DISCRETE_GPU_FOR_VALIDATION" == "true" ]]; then
        log_info "Using discrete gpu for validation - this should only be used on single instance, small multi-gpu instances"
        if [[ $NUM_GPUS > 1 ]]; then
            NUM_GPUS=$((NUM_GPUS - 1))
            log_info "Resetting available GPUS to $NUM_GPUS"
        else
            log_error "'use_discrete_gpu_for_validation' is set, but NUM_GPUS=$NUM_GPUS. This number has to be > 1 to allow for offloading."
            exit 1
        fi
    fi
}

############################################
# Input validation (includes eval assets)
############################################
validate_inputs() {
    [[ -z "$CONFIG_PATH" ]] && { log_error "--config is required"; show_usage; exit 1; }
    [[ -z "$TRAINING_SCRIPT" ]] && { log_error "--training-script is required"; show_usage; exit 1; }
    [[ -z "$ACCELERATE_CONFIG" ]] && { log_error "--accelerate-config is required"; show_usage; exit 1; }

    validate_file_exists "$CONFIG_PATH" "Configuration file"
    validate_file_exists "$TRAINING_SCRIPT" "Training script"
    validate_file_exists "$ACCELERATE_CONFIG" "Accelerate configuration"

    resolve_num_gpus
    validate_positive_integer "$NUM_GPUS" "GPU count"
}

############################################
# Dependencies (uv + tools used later)
############################################
install_dependencies() {
    # uv (fast pip)
    if ! command -v uv &> /dev/null; then
        python3 -m pip install --upgrade uv || { log_error "Failed to install uv"; exit 1; }
    fi

    # Project deps
    [[ -f "$REQUIREMENTS_FILE" ]] && uv pip install --system -r "$REQUIREMENTS_FILE"

    # yq is required for dynamic accelerate backend regardless of eval
    if ! command -v yq &> /dev/null; then
        uv pip install --system yq || { log_error "Failed to install yq"; exit 1; }
    fi
}

############################################
# Secrets (fetch from AWS Secrets Manager)
############################################
# If SECRETS_ARN is set (passed via the ModelTrainer `environment` parameter),
# fetch the JSON secret and export each key as an environment variable so
# downstream tools pick them up automatically — e.g. HF_TOKEN for the gated
# Stable Diffusion 3.5 model download, and WANDB_API_KEY for W&B logging.
#
# Passing the ARN (not the raw tokens) keeps secrets out of the job definition,
# the notebook, and CloudTrail. The execution role just needs
# secretsmanager:GetSecretValue on this secret. boto3 is already present in the
# training image (pulled in by sagemaker-mlflow).
load_secrets() {
    if [[ -z "${SECRETS_ARN:-}" ]]; then
        log_info "No SECRETS_ARN provided; skipping Secrets Manager fetch."
        return
    fi

    log_info "Fetching secrets from Secrets Manager: ${SECRETS_ARN}"
    local secrets_file keys
    secrets_file="$(mktemp)"

    # Python writes 'export KEY=value' lines to a temp file (never to stdout, so
    # secret values never land in the logs) and prints only the key names.
    keys="$(python3 - "$SECRETS_ARN" "$secrets_file" <<'PY'
import json, shlex, sys
import boto3

arn, out_path = sys.argv[1], sys.argv[2]
region = arn.split(":")[3]  # region is embedded in the ARN
client = boto3.client("secretsmanager", region_name=region)
secret = json.loads(client.get_secret_value(SecretId=arn)["SecretString"])
with open(out_path, "w") as f:
    for key, value in secret.items():
        f.write(f"export {key}={shlex.quote(str(value))}\n")
print(", ".join(secret.keys()))  # key names only — safe to log
PY
)"

    # shellcheck disable=SC1090
    source "$secrets_file"
    rm -f "$secrets_file"
    log_success "Exported secrets as environment variables: ${keys}"
}

############################################
# Accelerate presence (binary or importable)
############################################
check_accelerate_installation() {
    if command -v accelerate &> /dev/null; then
        return
    fi
    # Some images expose accelerate only as a Python module
    python3 - <<'PY' || { echo >&2 "[ERROR] accelerate not found"; exit 1; }
import importlib, sys
try:
    importlib.import_module("accelerate")
except Exception:
    sys.exit(1)
PY
}

############################################
# Accelerate config sanity (warning-only)
############################################
verify_accelerate_config() {
    if grep -q "rdzv_backend: static" "$ACCELERATE_CONFIG"; then
        log_warning "Found 'rdzv_backend: static' in $ACCELERATE_CONFIG. Prefer 'c10d' for SageMaker multi-node."
    elif grep -q "rdzv_backend: c10d" "$ACCELERATE_CONFIG"; then
        log_info "Accelerate config uses rdzv_backend: c10d"
    else
        log_warning "Could not verify rdzv_backend in $ACCELERATE_CONFIG (CLI overrides will still work)."
    fi
}

############################################
# Patch accelerate backend dynamically (via yq)
############################################
set_dynamic_rdzv_backend() {
    local desired="static"
    (( NUM_MACHINES > 1 )) && desired="c10d"
    log_info "Setting accelerate rdzv_backend -> ${desired} in: $ACCELERATE_CONFIG"

    if yq --version 2>&1 | grep -qi 'mikefarah'; then
        NEW_BACKEND="$desired" yq -i -y '.rdzv_backend = strenv(NEW_BACKEND)' "$ACCELERATE_CONFIG"
    else
        if ! command -v jq >/dev/null; then
            log_error "Python yq detected but jq is missing. Install jq or switch to mikefarah/yq."
            exit 1
        fi
        NEW_BACKEND="$desired" yq -yi '.rdzv_backend = env.NEW_BACKEND' "$ACCELERATE_CONFIG"
    fi

    yq -r '.rdzv_backend' "$ACCELERATE_CONFIG" | xargs -I{} echo "[INFO] Effective rdzv_backend: {}"
}

############################################
# Distributed environment (pure bash)
############################################
setup_distributed_environment() {
    log_info "Setting up distributed training environment variables"

    _trim() { local s="$1"; s="${s#"${s%%[![:space:]]*}"}"; s="${s%"${s##*[![:space:]]}"}"; printf '%s' "$s"; }
    _unquote() { local s="$1"; [[ "$s" == \"*\" ]] && s="${s#\"}"; [[ "$s" == *\" ]] && s="${s%\"}"; printf '%s' "$s"; }

    local hosts_json="${SM_HOSTS:-"[\"localhost\"]"}"
    local inner
    inner="$(printf '%s' "$hosts_json" | sed -e 's/^\s*\[\s*//' -e 's/\s*\]\s*$//')"
    IFS=',' read -r -a _items <<< "$inner"

    local hosts=()
    local item cleaned
    for item in "${_items[@]}"; do
        cleaned="$(_unquote "$(_trim "$item")")"
        [[ -n "$cleaned" ]] && hosts+=("$cleaned")
    done
    if [[ ${#hosts[@]} -eq 0 ]]; then hosts=("127.0.0.1"); fi

    NUM_MACHINES=${#hosts[@]}

    local current="${SM_CURRENT_HOST:-localhost}"
    MACHINE_RANK=0
    for i in "${!hosts[@]}"; do
        if [[ "${hosts[$i]}" == "$current" ]]; then MACHINE_RANK="$i"; break; fi
    done

    MASTER_ADDR="${hosts[0]}"
    MASTER_PORT="${MASTER_PORT:-29500}"

    if command -v getent &> /dev/null; then
        local ip; ip="$(getent ahostsv4 "$MASTER_ADDR" 2>/dev/null | awk 'NR==1{print $1}')"
        [[ -n "${ip:-}" ]] && MASTER_ADDR="$ip"
    fi

    # --- NEW: compute per-node and total procs, export Elastic envs ---
    PER_NODE_PROCS="${NUM_GPUS}"
    TOTAL_PROCS=$(( NUM_MACHINES * PER_NODE_PROCS ))
    export MACHINE_RANK MASTER_ADDR MASTER_PORT NUM_MACHINES PER_NODE_PROCS TOTAL_PROCS
    export LOCAL_WORLD_SIZE="$PER_NODE_PROCS"
    export WORLD_SIZE="$TOTAL_PROCS"
    export NODE_RANK="$MACHINE_RANK"
    # ------------------------------------------------------------------

    log_info "Distributed setup:"
    log_info "  - Num machines: ${NUM_MACHINES}"
    log_info "  - Per-node procs (GPUs): ${PER_NODE_PROCS}"
    log_info "  - Total processes: ${TOTAL_PROCS}"
    log_info "  - Machine rank: ${MACHINE_RANK}"
    log_info "  - Master addr: ${MASTER_ADDR}"
    log_info "  - Master port: ${MASTER_PORT}"

    log_info "SageMaker Environment Variables:"
    log_info "  - SM_HOSTS: ${SM_HOSTS:-NOT SET}"
    log_info "  - SM_CURRENT_HOST: ${SM_CURRENT_HOST:-NOT SET}"
    log_info "  - SM_NUM_GPUS: ${SM_NUM_GPUS:-NOT SET}"
    log_info "  - SM_NUM_CPUS: ${SM_NUM_CPUS:-NOT SET}"
}

############################################
# Training launch
############################################
launch_training() {
    log_info "Starting training:"
    log_info "  - Config file: $CONFIG_PATH"
    log_info "  - Accelerate config: $ACCELERATE_CONFIG"
    log_info "  - Training script: $TRAINING_SCRIPT"
    log_info "  - Machine rank: $MACHINE_RANK"
    log_info "  - Per-machine processes (GPUs): $PER_NODE_PROCS"
    log_info "  - Num machines: $NUM_MACHINES"
    log_info "  - Total processes: $TOTAL_PROCS"

    # Forward MLflow settings to the training script only when provided, so runs
    # log to the SageMaker managed MLflow tracking server (see mlflow_arn handling
    # in the training script). Left empty -> training falls back to local tracking.
    # The notebook may interpolate Python None into the literal string "None", so
    # treat "None" (any case) the same as empty -> skip tracking, don't forward it.
    local mlflow_args=()
    local mlflow_arn_lc="$(printf '%s' "$MLFLOW_ARN" | tr '[:upper:]' '[:lower:]')"
    if [[ -n "$MLFLOW_ARN" && "$mlflow_arn_lc" != "none" ]]; then
        mlflow_args+=(--mlflow_arn "$MLFLOW_ARN")
        [[ -n "$MLFLOW_EXPERIMENT_NAME" ]] && mlflow_args+=(--mlflow_experiment_name "$MLFLOW_EXPERIMENT_NAME")
    fi

    # minimal: pass TOTAL_PROCS (matches 'questionnaire' semantics) and keep topology flags
    if accelerate launch \
        --config_file "$ACCELERATE_CONFIG" \
        --num_machines "$NUM_MACHINES" \
        --machine_rank "$MACHINE_RANK" \
        --num_processes "$TOTAL_PROCS" \
        --main_process_ip "$MASTER_ADDR" \
        --main_process_port "$MASTER_PORT" \
        "$TRAINING_SCRIPT" \
        --config "$CONFIG_PATH" \
        ${mlflow_args[@]+"${mlflow_args[@]}"}
    then
        log_success "Training completed successfully!"
    else
        local exit_code=$?
        log_error "Training failed with exit code: $exit_code"
        exit "$exit_code"
    fi
}

############################################
# Stage serving artifacts into the model dir
############################################
# Copy the inference-time requirements into $SM_MODEL_DIR so they land at the
# ROOT of model.tar.gz. The DJL Serving inference container installs a
# root-level requirements.txt at startup (it ignores one nested under code/),
# which is how our inference handler gets boto3 + diffusers at deploy time.
# This keeps the model artifact self-contained: weights + the deps to serve them.
stage_serving_artifacts() {
    local model_dir="${SM_MODEL_DIR:-/opt/ml/model}"
    local reqs="${SCRIPT_DIR}/serving-requirements.txt"
    if [[ -f "$reqs" ]]; then
        cp "$reqs" "${model_dir}/requirements.txt"
        log_success "Staged serving requirements.txt into ${model_dir} (packaged at model.tar.gz root)"
    else
        log_warning "No serving-requirements.txt found at ${reqs}; skipping."
    fi
}

############################################
# Main
############################################
main() {
    parse_arguments "$@"
    install_dependencies
    load_secrets
    validate_inputs
    setup_distributed_environment
    set_dynamic_rdzv_backend

    # print the deepspeed configuration
    log_info "******************* Start of DeepSpeed Configuration *******************"
    more "$ACCELERATE_CONFIG"
    log_info "******************** End of DeepSpeed Configuration ********************"

    check_accelerate_installation
    verify_accelerate_config
    launch_training
    stage_serving_artifacts

    log_success "All steps completed successfully"
}

main "$@"
