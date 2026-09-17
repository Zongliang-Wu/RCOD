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

# Target checkpoints:
# 1. rcod_o.pkl     - RCOD_O checkpoint (OSEDiff base, SD 2.1)
# 2. rcod_s.pkl     - RCOD_S checkpoint (S3Diff base, SD-Turbo)
# 3. rcod_s_mem.pkl - Metric Estimation Module (MEM) for RCOD_S-Adap
# 4. de_net.pth     - Degradation Estimator Network (MM-RealSR)
# 5. DAPE.pth       - Domain-Adaptive Prior Extractor (for visual prompt)
# 6. ram_swin_large_14m.pth - RAM tag extractor

echo "Pretrained checkpoints will be released on Hugging Face Hub:"
echo "  https://huggingface.co/Zongliang-Wu/RCOD"
echo ""
echo "You can download weights using huggingface-cli:"
echo "  pip install huggingface_hub"
echo "  huggingface-cli download Zongliang-Wu/RCOD --local-dir $WEIGHTS_DIR"
echo ""
echo "Or download manually and place checkpoints into the '$WEIGHTS_DIR' folder."
echo "======================================================================"
