#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GNN_REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
DUMMY_REPO="${DUMMY_REPO:-/Users/chrisizeh/Documents/CERN/TICL-HGCAL-Dummy-Data}"

python_has_pipeline_deps() {
  "$1" -c 'import awkward, matplotlib, scipy, seaborn, sklearn, torch, torch_geometric, uproot' >/dev/null 2>&1
}

if [[ -n "${PYTHON:-}" ]]; then
  if ! python_has_pipeline_deps "${PYTHON}"; then
    echo "Selected PYTHON is missing one or more pipeline dependencies: ${PYTHON}" >&2
    echo "Required imports: awkward, matplotlib, scipy, seaborn, sklearn, torch, torch_geometric, uproot" >&2
    exit 1
  fi
else
  python_candidates=(
    "${DUMMY_REPO}/.venv/bin/python"
    "/Users/chrisizeh/Documents/PhD/TICL-Pipeline/.venv/bin/python"
    "/Users/chrisizeh/Documents/PhD/.venv/bin/python"
    "/Users/chrisizeh/Documents/PhD/venv/bin/python"
    "${GNN_REPO}/.venv-threshold/bin/python"
    "python3"
  )
  PYTHON=""
  for candidate in "${python_candidates[@]}"; do
    if command -v "${candidate}" >/dev/null 2>&1 && python_has_pipeline_deps "${candidate}"; then
      PYTHON="${candidate}"
      break
    fi
  done
  if [[ -z "${PYTHON}" ]]; then
    echo "No Python with all pipeline dependencies was found." >&2
    echo "Required imports: awkward, matplotlib, scipy, seaborn, sklearn, torch, torch_geometric, uproot" >&2
    echo "Set PYTHON=/path/to/python and rerun after installing them." >&2
    exit 1
  fi
fi

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-dummy_gnn_rootlike_${RUN_STAMP}}"
BASE_FOLDER="${BASE_FOLDER:-${GNN_REPO}/data}"
DATA_ROOT="${DATA_ROOT:-${BASE_FOLDER}/linking_dataset/${RUN_NAME}}"
RAW_DATA_DIR="${RAW_DATA_DIR:-${DATA_ROOT}/histo}"
PROCESSED_ROOT="${PROCESSED_ROOT:-${DATA_ROOT}/gnn_dataset}"
MODEL_ROOT="${MODEL_ROOT:-${BASE_FOLDER}/training_data/${RUN_NAME}}"
VALIDATION_OUTPUT_DIR="${VALIDATION_OUTPUT_DIR:-${MODEL_ROOT}/validation_plots}"

TRAIN_FILES="${TRAIN_FILES:-80}"
VAL_FILES="${VAL_FILES:-20}"
TEST_FILES="${TEST_FILES:-20}"
EVENTS_PER_FILE="${EVENTS_PER_FILE:-10}"
SEED="${SEED:-12345}"
GEN_WORKERS="${GEN_WORKERS:-1}"

EPOCHS="${EPOCHS:-30}"
NUM_WORKERS="${NUM_WORKERS:-1}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5}"
PLOT_EVERY="${PLOT_EVERY:-5}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-20}"
LR="${LR:-0.001}"
CONTRASTIVE_WEIGHT="${CONTRASTIVE_WEIGHT:-0.001}"
FOCAL_WEIGHT="${FOCAL_WEIGHT:-1.0}"
MARGIN="${MARGIN:-0.5}"

VALIDATION_LIMIT="${VALIDATION_LIMIT:-50}"
VALIDATION_DEVICE="${VALIDATION_DEVICE:-${DEVICE:-cpu}}"
AUTO_THRESHOLD="${AUTO_THRESHOLD:-1}"
THRESHOLD_STEP="${THRESHOLD_STEP:-0.01}"
PIECEWISE_AXIS="${PIECEWISE_AXIS:-energy}"
PIECEWISE_BINS="${PIECEWISE_BINS:-2}"

export PYTHONPATH="${GNN_REPO}/tracksterLinker:${DUMMY_REPO}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/matplotlib-cache}"
mkdir -p "${MPLCONFIGDIR}" "${MODEL_ROOT}"

echo "Python: ${PYTHON}"
echo "Dummy repo: ${DUMMY_REPO}"
echo "GNN repo: ${GNN_REPO}"
echo "Run name: ${RUN_NAME}"
echo "Raw ROOT data: ${RAW_DATA_DIR}"
echo "Processed GNNDataset root: ${PROCESSED_ROOT}"
echo "Model output: ${MODEL_ROOT}"
echo "Validation output: ${VALIDATION_OUTPUT_DIR}"

generation_args=(
  "${DUMMY_REPO}/scripts/generate_hgcal_like_dummy.py"
  --output-dir "${RAW_DATA_DIR}"
  --train-files "${TRAIN_FILES}"
  --val-files "${VAL_FILES}"
  --test-files "${TEST_FILES}"
  --events-per-file "${EVENTS_PER_FILE}"
  --seed "${SEED}"
  --workers "${GEN_WORKERS}"
)

echo
echo "Generating dummy ROOT data..."
"${PYTHON}" "${generation_args[@]}"

train_args=(
  "${GNN_REPO}/scripts/run_dummy_reco_experiment.py"
  --dataset-kind gnn
  --base-folder "${BASE_FOLDER}"
  --run-name "${RUN_NAME}"
  --data-folder "${DATA_ROOT}"
  --raw-data-dir "${RAW_DATA_DIR}"
  --processed-data-dir "${PROCESSED_ROOT}"
  --model-folder "${MODEL_ROOT}"
  --architecture punet
  --epochs "${EPOCHS}"
  --num-workers "${NUM_WORKERS}"
  --checkpoint-every "${CHECKPOINT_EVERY}"
  --plot-every "${PLOT_EVERY}"
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE}"
  --lr "${LR}"
  --contrastive-weight "${CONTRASTIVE_WEIGHT}"
  --focal-weight "${FOCAL_WEIGHT}"
  --margin "${MARGIN}"
  --seed "${SEED}"
)

if [[ -n "${DEVICE:-}" ]]; then
  train_args+=(--device "${DEVICE}")
fi
if [[ -n "${LIMIT_TRAIN:-}" ]]; then
  train_args+=(--limit-train "${LIMIT_TRAIN}")
fi
if [[ -n "${LIMIT_VAL:-}" ]]; then
  train_args+=(--limit-val "${LIMIT_VAL}")
fi
if [[ -n "${LIMIT_TEST:-}" ]]; then
  train_args+=(--limit-test "${LIMIT_TEST}")
fi

echo
echo "Processing GNNDataset splits and training focal + focal_contrastive..."
"${PYTHON}" "${train_args[@]}"

latest_checkpoint() {
  "${PYTHON}" -c 'import glob, os, sys
paths = glob.glob(os.path.join(sys.argv[1], "*_dict.pt"))
if not paths:
    raise SystemExit(f"No *_dict.pt checkpoint found in {sys.argv[1]}")
print(max(paths, key=os.path.getmtime))' "$1"
}

FOCAL_CKPT="$(latest_checkpoint "${MODEL_ROOT}/focal")"
CONTRASTIVE_CKPT="$(latest_checkpoint "${MODEL_ROOT}/focal_contrastive")"
VALIDATION_DATASET="${PROCESSED_ROOT}/dataset_dummy_reco_val"

eval_args=(
  "${GNN_REPO}/scripts/evaluate_dummy_gnn_binned.py"
  --dataset "${VALIDATION_DATASET}"
  --output-dir "${VALIDATION_OUTPUT_DIR}"
  --focal-checkpoint "${FOCAL_CKPT}"
  --contrastive-checkpoint "${CONTRASTIVE_CKPT}"
  --device "${VALIDATION_DEVICE}"
  --threshold-step "${THRESHOLD_STEP}"
)

if [[ "${AUTO_THRESHOLD}" == "1" ]]; then
  eval_args+=(--auto-threshold)
fi
if [[ -n "${PIECEWISE_AXIS}" ]]; then
  eval_args+=(--piecewise-threshold-axis "${PIECEWISE_AXIS}" --piecewise-threshold-bins "${PIECEWISE_BINS}")
fi
if [[ -n "${PIECEWISE_EDGES:-}" ]]; then
  eval_args+=(--piecewise-threshold-edges "${PIECEWISE_EDGES}")
fi
if [[ -n "${VALIDATION_LIMIT}" && "${VALIDATION_LIMIT}" != "all" ]]; then
  eval_args+=(--limit "${VALIDATION_LIMIT}")
fi

echo
echo "Running binned validation plots..."
"${PYTHON}" "${eval_args[@]}"

echo
echo "Done."
echo "Focal checkpoint: ${FOCAL_CKPT}"
echo "Focal+contrastive checkpoint: ${CONTRASTIVE_CKPT}"
echo "Validation plots: ${VALIDATION_OUTPUT_DIR}"
