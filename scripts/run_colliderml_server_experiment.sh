#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
DATASET="${DATASET:-ttbar_pu0}"
RUN_NAME="${RUN_NAME:-colliderml_${DATASET}_det11_14_graphutils_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/${RUN_NAME}}"
MODEL_DIR="${MODEL_DIR:-data/training_data/${RUN_NAME}}"
VALIDATION_DIR="${VALIDATION_DIR:-${MODEL_DIR}/binned_validation}"

TRAIN_EVENTS="${TRAIN_EVENTS:-160}"
VAL_EVENTS="${VAL_EVENTS:-20}"
TEST_EVENTS="${TEST_EVENTS:-20}"
EVENT_START="${EVENT_START:-0}"

DETECTORS="${DETECTORS:-11,14}"
ETA_BIN_WIDTH="${ETA_BIN_WIDTH:-0.035}"
PHI_BIN_WIDTH="${PHI_BIN_WIDTH:-0.035}"
DEPTH_BIN_WIDTH="${DEPTH_BIN_WIDTH:-35}"
EDGE_DELTA_R="${EDGE_DELTA_R:-0.20}"
MIN_TRUTH_PURITY="${MIN_TRUTH_PURITY:-0.45}"

ARCHITECTURE="${ARCHITECTURE:-gnn}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-1e-3}"
DEVICE="${DEVICE:-auto}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PLOT_EVERY="${PLOT_EVERY:-5}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-20}"
SEED="${SEED:-12345}"

echo "== ColliderML server experiment =="
echo "repo:        $REPO_ROOT"
echo "dataset:     $DATASET"
echo "events:      train=$TRAIN_EVENTS val=$VAL_EVENTS test=$TEST_EVENTS start=$EVENT_START"
echo "detectors:   $DETECTORS"
echo "output:      $OUTPUT_ROOT"
echo "model dir:   $MODEL_DIR"
echo "architecture:$ARCHITECTURE"
echo

"$PYTHON" - <<'PY'
missing = []
for module in ["torch", "torch_geometric", "uproot", "awkward", "numpy", "matplotlib", "seaborn", "colliderml", "polars"]:
    try:
        __import__(module)
    except Exception as exc:
        missing.append(f"{module}: {exc}")
if missing:
    raise SystemExit("Missing Python dependencies:\n  " + "\n  ".join(missing))
PY

echo "== Build processed graphs =="
"$PYTHON" scripts/create_colliderml_gnn_dataset.py \
  --dataset "$DATASET" \
  --auto-download \
  --train-events "$TRAIN_EVENTS" \
  --val-events "$VAL_EVENTS" \
  --test-events "$TEST_EVENTS" \
  --event-start "$EVENT_START" \
  --output-root "$OUTPUT_ROOT" \
  --overwrite \
  --detectors "$DETECTORS" \
  --eta-bin-width "$ETA_BIN_WIDTH" \
  --phi-bin-width "$PHI_BIN_WIDTH" \
  --depth-bin-width "$DEPTH_BIN_WIDTH" \
  --edge-delta-r "$EDGE_DELTA_R" \
  --min-truth-purity "$MIN_TRUTH_PURITY"

echo
echo "== Graph summary =="
"$PYTHON" scripts/summarize_processed_graphs.py "$OUTPUT_ROOT"

echo
echo "== Train/evaluate reconstruction models =="
"$PYTHON" scripts/run_dummy_reco_experiment.py \
  --dataset-kind processed \
  --processed-data-dir "$OUTPUT_ROOT" \
  --model-folder "$MODEL_DIR" \
  --architecture "$ARCHITECTURE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --device "$DEVICE" \
  --num-workers "$NUM_WORKERS" \
  --plot-every "$PLOT_EVERY" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
  --seed "$SEED"

echo
echo "== Binned validation plots =="
read -r FOCAL_CHECKPOINT FOCAL_THRESHOLD CONTRASTIVE_CHECKPOINT CONTRASTIVE_THRESHOLD < <(
  "$PYTHON" - "$MODEL_DIR" <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
rows = []
for name in ["focal", "focal_contrastive"]:
    metadata = json.loads((model_dir / name / "metadata.json").read_text())
    rows.extend([metadata["checkpoint"], str(metadata["threshold"])])
print(" ".join(rows))
PY
)

"$PYTHON" scripts/evaluate_dummy_gnn_binned.py \
  --dataset "$OUTPUT_ROOT/dataset_colliderml_reco_test" \
  --output-dir "$VALIDATION_DIR" \
  --focal-checkpoint "$FOCAL_CHECKPOINT" \
  --focal-threshold "$FOCAL_THRESHOLD" \
  --contrastive-checkpoint "$CONTRASTIVE_CHECKPOINT" \
  --contrastive-threshold "$CONTRASTIVE_THRESHOLD" \
  --auto-threshold \
  --device "$DEVICE"

echo
echo "Done."
echo "processed graphs: $OUTPUT_ROOT"
echo "model outputs:    $MODEL_DIR"
echo "validation plots: $VALIDATION_DIR"
