#!/bin/bash
# ==============================================================================
# Script to run RCOD_S inference (Manual or Adaptive)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

GPU_ID=${1:-"0"}
MODE=${2:-"adaptive"} # "adaptive" or timestep e.g. "250", "500", "750"
INPUT_DIR=${3:-"$REPO_ROOT/assets/samples"}
OUTPUT_DIR=${4:-"$REPO_ROOT/results/RCOD_S_${MODE}"}
CKPT_PATH=${5:-"$REPO_ROOT/weights/rcod_s.pkl"}
MEM_PATH=${6:-"$REPO_ROOT/weights/rcod_s_mem.pkl"}

# Check weight existence
if [ ! -f "$CKPT_PATH" ]; then
    echo "[ERROR] Pretrained weight not found at '$CKPT_PATH'."
    echo "Please place model weights in 'weights/rcod_s.pkl' or run 'bash scripts/download_weights.sh'."
    exit 1
fi

if [ "$MODE" == "adaptive" ] && [ ! -f "$MEM_PATH" ]; then
    echo "[ERROR] MEM weight not found at '$MEM_PATH'."
    echo "Please place MEM weight in 'weights/rcod_s_mem.pkl' or run 'bash scripts/download_weights.sh'."
    exit 1
fi

cd "$REPO_ROOT/RCOD_S"

PYTHON_BIN=${PYTHON:-""}
if [ -z "$PYTHON_BIN" ]; then
    if command -v python &>/dev/null; then
        PYTHON_BIN="python"
    elif command -v python3 &>/dev/null; then
        PYTHON_BIN="python3"
    fi
fi

echo "Running RCOD_S inference..."
echo "  Python:     $PYTHON_BIN"
echo "  Mode:       $MODE"
echo "  Input dir:  $INPUT_DIR"
echo "  Output dir: $OUTPUT_DIR"
echo "  Checkpoint: $CKPT_PATH"

if [ "$MODE" == "adaptive" ]; then
    "$PYTHON_BIN" inference_rcod_s.py \
        -i "$INPUT_DIR" \
        -o "$OUTPUT_DIR" \
        --pretrained_path "$CKPT_PATH" \
        --mlp_path "$MEM_PATH" \
        --adaptive \
        --device "cuda:$GPU_ID"
else
    "$PYTHON_BIN" inference_rcod_s.py \
        -i "$INPUT_DIR" \
        -o "$OUTPUT_DIR" \
        --pretrained_path "$CKPT_PATH" \
        --timestep "$MODE" \
        --device "cuda:$GPU_ID"
fi
