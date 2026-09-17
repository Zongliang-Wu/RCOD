#!/usr/bin/env bash
set -e

# Training script for RCOD_S
GPUS=${GPUS:-"0,1,2,3"}
SD_TURBO_PATH=${SD_TURBO_PATH:-"weights/sd-turbo"}
DE_NET_PATH=${DE_NET_PATH:-"assets/mm-realsr/de_net.pth"}
CONFIG_PATH=${CONFIG_PATH:-"configs/sr.yaml"}
OUT_DIR=${OUT_DIR:-"experience/rcod_s_train"}

CUDA_VISIBLE_DEVICES="${GPUS}" accelerate launch --main_process_port 29501 src/train_rcod_s.py \
    --sd_path="${SD_TURBO_PATH}" \
    --de_net_path="${DE_NET_PATH}" \
    --base_config="${CONFIG_PATH}" \
    --model_name="rcod_s_notext" \
    --output_dir="${OUT_DIR}" \
    --train_batch_size=4 \
    --learning_rate=2e-5 \
    --max_train_steps=50000 \
    --checkpointing_steps=500 \
    --mixed_precision="fp16" \
    --gradient_accumulation_steps=4 \
    --report_to="tensorboard"
