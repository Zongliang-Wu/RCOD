#!/usr/bin/env bash
set -e

# Training script for RCOD_O
GPUS=${GPUS:-"0,1,2,3"}
SD_PATH=${SD_PATH:-"weights/stable-diffusion-2-1-base"}
OUT_DIR=${OUT_DIR:-"experience/rcod_o_train"}

CUDA_VISIBLE_DEVICES="${GPUS}" accelerate launch train_rcod_o.py \
    --pretrained_model_name_or_path="${SD_PATH}" \
    --ram_path="weights/ram_swin_large_14m.pth" \
    --ram_ft_path="weights/DAPE.pth" \
    --model_name="rcod_o" \
    --learning_rate="5e-5" \
    --train_batch_size="2" \
    --gradient_accumulation_steps="1" \
    --enable_xformers_memory_efficient_attention \
    --checkpointing_steps="500" \
    --mixed_precision="fp16" \
    --report_to="tensorboard" \
    --seed="123" \
    --output_dir="${OUT_DIR}" \
    --neg_prompt="painting, oil painting, illustration, drawing, art, sketch, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth" \
    --cfg_vsd="7.5" \
    --lora_rank="4" \
    --lambda_lpips="2" \
    --lambda_l2="1" \
    --lambda_vsd="1" \
    --lambda_vsd_lora="1" \
    --deg_file_path="dataloaders/params_realesrgan.yml" \
    --tracker_project_name="train_rcod_o"
