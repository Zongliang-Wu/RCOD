#!/usr/bin/env bash
# ==============================================================================
# Download Pretrained Weights for RCOD (AAAI-26)
# ==============================================================================
set -e

WEIGHTS_DIR="$(dirname "$0")/../weights"
mkdir -p "$WEIGHTS_DIR"

echo "======================================================================"
echo "Preparing RCOD weights directory: $WEIGHTS_DIR"
echo "======================================================================"

# 1. RCOD Trained Checkpoints:
# - rcod_o.pkl     - RCOD_O checkpoint (OSEDiff base, SD 2.1)
# - rcod_s.pkl     - RCOD_S checkpoint (S3Diff base, SD-Turbo)
# - rcod_s_mem.pkl - Metric Estimation Module (MEM) for RCOD_S-Adap
echo "Downloading RCOD trained models from Hugging Face (MMQDD/RCOD)..."
if command -v huggingface-cli &>/dev/null; then
    huggingface-cli download MMQDD/RCOD --local-dir "$WEIGHTS_DIR"
else
    echo "huggingface-cli not found. You can install it via:"
    echo "  pip install huggingface_hub"
    echo "Or download manually from: https://huggingface.co/MMQDD/RCOD"
fi

# 2. Auxiliary checkpoints:
# - de_net.pth: included in RCOD_S/assets/mm-realsr/de_net.pth
DE_NET_SRC="$(dirname "$0")/../RCOD_S/assets/mm-realsr/de_net.pth"
if [ -f "$DE_NET_SRC" ] && [ ! -f "$WEIGHTS_DIR/de_net.pth" ]; then
    echo "Linking de_net.pth into weights/..."
    ln -sf "$DE_NET_SRC" "$WEIGHTS_DIR/de_net.pth"
fi

echo ""
echo "External auxiliary models (download if training or using full pipeline):"
echo "  - DAPE.pth: https://drive.google.com/file/d/1KIV6VewwO2eDC9g4Gcvgm-a0LDI7Lmwm/view?usp=drive_link"
echo "  - ram_swin_large_14m.pth: https://huggingface.co/spaces/xinyu1205/recognize-anything/blob/main/ram_swin_large_14m.pth"
echo "======================================================================"
