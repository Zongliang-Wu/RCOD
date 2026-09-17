#!/usr/bin/env bash
set -e

# Testing script for RCOD_O
# Select desired timestep: 249 (Fidelity), 499 (Neutral), 749 (Realism)
TIMESTEP=${1:-499}
GPU_ID=${2:-0}

INPUT_IMG=${INPUT_IMG:-"datasets/RealSR/test_LR"}
GT_IMG=${GT_IMG:-"datasets/RealSR/test_HR"}
OUT_DIR=${OUT_DIR:-"results/rcod_o_eval/t${TIMESTEP}"}
CKPT_PATH=${CKPT_PATH:-"weights/rcod_o.pkl"}
SD_PATH=${SD_PATH:-"weights/stable-diffusion-2-1-base"}

CUDA_VISIBLE_DEVICES="${GPU_ID}" python test_rcod_o.py \
    --input_image="${INPUT_IMG}" \
    --gt_imgs="${GT_IMG}" \
    --output_dir="${OUT_DIR}" \
    --osediff_path="${CKPT_PATH}" \
    --pretrained_model_name_or_path="${SD_PATH}" \
    --model_name="rcod_o" \
    --ram_ft_path="weights/DAPE.pth" \
    --ram_path="weights/ram_swin_large_14m.pth" \
    --model_gen_timestep="${TIMESTEP}" \
    --process_size=512
