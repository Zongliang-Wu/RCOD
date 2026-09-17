#!/bin/bash
# ==============================================================================
# Script to run RCOD_O inference on sample images or custom data
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

GPU_ID=${1:-"0"}
TIMESTEP=${2:-"500"} # 250 (Fidelity), 500 (Neutral), 750 (Realism)
INPUT_DIR=${3:-"$REPO_ROOT/assets/samples"}
OUTPUT_DIR=${4:-"$REPO_ROOT/results/RCOD_O_t${TIMESTEP}"}
CKPT_PATH=${5:-"$REPO_ROOT/weights/rcod_o.pkl"}

# Check weight existence
if [ ! -f "$CKPT_PATH" ]; then
    echo "[ERROR] Pretrained weight not found at '$CKPT_PATH'."
    echo "Please place model weights in 'weights/rcod_o.pkl' or run 'bash scripts/download_weights.sh'."
    exit 1
fi

cd "$REPO_ROOT/RCOD_O"

PYTHON_BIN=${PYTHON:-""}
if [ -z "$PYTHON_BIN" ]; then
    if command -v python &>/dev/null; then
        PYTHON_BIN="python"
    elif command -v python3 &>/dev/null; then
        PYTHON_BIN="python3"
    fi
fi

echo "Running RCOD_O inference..."
echo "  Python:     $PYTHON_BIN"
echo "  Timestep:   $TIMESTEP"
echo "  Input dir:  $INPUT_DIR"
echo "  Output dir: $OUTPUT_DIR"
echo "  Checkpoint: $CKPT_PATH"

"$PYTHON_BIN" inference_rcod_o.py \
    -i "$INPUT_DIR" \
    -o "$OUTPUT_DIR" \
    --osediff_path "$CKPT_PATH" \
    --timestep "$TIMESTEP" \
    --device "cuda:$GPU_ID"
