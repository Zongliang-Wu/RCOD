#!/usr/bin/env bash
set -e

# Benchmark inference speed for RCOD_O
GPU_ID=${1:-0}
ITERS=${2:-20}
SIZE=${3:-512}

MODEL_PATH=${MODEL_PATH:-"weights/rcod_o.pkl"}
SD_PATH=${SD_PATH:-"weights/stable-diffusion-2-1-base"}

CUDA_VISIBLE_DEVICES="${GPU_ID}" python RCOD_O/benchmark_speed.py \
    --model_path="${MODEL_PATH}" \
    --pretrained_model_name_or_path="${SD_PATH}" \
    --inference_iterations="${ITERS}" \
    --process_size="${SIZE}"
