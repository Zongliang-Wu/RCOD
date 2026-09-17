#!/usr/bin/env bash
set -e

# Testing script for RCOD_S
# Usage: bash scripts/test/test_rcod_s.sh [timestep] [gpu_id]
# Timestep: 249 (Fidelity), 499 (Neutral), 749 (Realism), or 'adaptive'
TIMESTEP=${1:-499}
GPU_ID=${2:-0}

INPUT_IMG=${INPUT_IMG:-"datasets/RealSR/test_LR"}
OUT_DIR=${OUT_DIR:-"results/rcod_s_eval/t${TIMESTEP}"}
CKPT_PATH=${CKPT_PATH:-"weights/rcod_s.pkl"}
SD_PATH=${SD_PATH:-"weights/sd-turbo"}
DE_NET_PATH=${DE_NET_PATH:-"assets/mm-realsr/de_net.pth"}
MEM_PATH=${MEM_PATH:-"weights/rcod_s_mem.pkl"}

EXTRA_ARGS=""
if [ "$TIMESTEP" == "adaptive" ]; then
    EXTRA_ARGS="--adaptive --mem_path=${MEM_PATH}"
else
    EXTRA_ARGS="--timestep=${TIMESTEP}"
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" python inference_rcod_s.py \
    --input_image="${INPUT_IMG}" \
    --output_dir="${OUT_DIR}" \
    --ckpt_path="${CKPT_PATH}" \
    --sd_path="${SD_PATH}" \
    --de_net_path="${DE_NET_PATH}" \
    ${EXTRA_ARGS}
