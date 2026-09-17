# Evaluation & IQA Metrics Calculation Script for RCOD
# Supports FR: PSNR, SSIM, LPIPS, DISTS
# Supports NR: NIQE, MUSIQ, MANIQA, CLIPIQA
# Supports Distribution: FID

import os
import sys
import glob
import argparse
import logging
from datetime import datetime
import time

import cv2
import numpy as np
import torch
import pyiqa
from basicsr.utils import img2tensor

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def setup_logger(logger_name, root, phase, level=logging.INFO, screen=True, tofile=True):
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        fmt='%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
        datefmt='%y-%m-%d %H:%M:%S'
    )
    logger.setLevel(level)

    if tofile:
        os.makedirs(root, exist_ok=True)
        log_file = os.path.join(root, f"{phase}_{get_timestamp()}.log")
        fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)
    return logger

def main():
    parser = argparse.ArgumentParser(description="Full IQA Metrics Evaluation for RCOD")
    parser.add_argument("--inp_imgs", "-i", type=str, required=True, help="Path to restored SR images directory")
    parser.add_argument("--gt_imgs", "-g", type=str, required=True, help="Path to ground truth HR images directory")
    parser.add_argument("--log_dir", "-l", type=str, default=None, help="Directory path to save evaluation logs")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--calc_fid", action="store_true", default=True, help="Whether to calculate FID")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available() and "cuda" in args.device:
        device_idx = int(args.device.split(":")[-1]) if ":" in args.device else 0
        torch.cuda.set_device(device_idx)

    log_dir = args.log_dir if args.log_dir else os.path.join(args.inp_imgs, "metrics")
    logger = setup_logger("rcod_eval", log_dir, "eval_metrics")

    logger.info("Initializing IQA metrics (PSNR, SSIM, LPIPS, DISTS, NIQE, MUSIQ, MANIQA, CLIPIQA)...")
    iqa_metrics = {
        'PSNR': pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(device),
        'SSIM': pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr').to(device),
        'LPIPS': pyiqa.create_metric('lpips', device=device),
        'DISTS': pyiqa.create_metric('dists', device=device),
        'CLIPIQA': pyiqa.create_metric('clipiqa', device=device),
        'NIQE': pyiqa.create_metric('niqe', device=device),
        'MUSIQ': pyiqa.create_metric('musiq', device=device),
        'MANIQA': pyiqa.create_metric('maniqa-pipal', device=device)
    }
    
    fid_metric = None
    if args.calc_fid:
        logger.info("Initializing FID metric...")
        fid_metric = pyiqa.create_metric('fid', device=device)

    img_sr_list = sorted(glob.glob(os.path.join(args.inp_imgs, "*.png")) + glob.glob(os.path.join(args.inp_imgs, "*.jpg")))
    img_gt_list = sorted(glob.glob(os.path.join(args.gt_imgs, "*.png")) + glob.glob(os.path.join(args.gt_imgs, "*.jpg")))

    logger.info(f"Found {len(img_sr_list)} SR images and {len(img_gt_list)} GT images.")
    assert len(img_sr_list) == len(img_gt_list), f"Image count mismatch: SR ({len(img_sr_list)}) vs GT ({len(img_gt_list)})"

    metrics_accum = {metric: 0.0 for metric in iqa_metrics.keys()}

    for i in range(len(img_sr_list)):
        sr_path = img_sr_list[i]
        gt_path = img_gt_list[i]
        fname = os.path.basename(sr_path)

        sr_img = cv2.imread(sr_path, cv2.IMREAD_COLOR)
        gt_img = cv2.imread(gt_path, cv2.IMREAD_COLOR)

        sr_tensor = img2tensor(sr_img, bgr2rgb=True, float32=True).unsqueeze(0).to(device).contiguous() / 255.0
        gt_tensor = img2tensor(gt_img, bgr2rgb=True, float32=True).unsqueeze(0).to(device).contiguous() / 255.0

        with torch.no_grad():
            for name, metric in iqa_metrics.items():
                if name in ['CLIPIQA', 'NIQE', 'MUSIQ', 'MANIQA']:
                    val = metric(sr_tensor).item()
                else:
                    val = metric(sr_tensor, gt_tensor).item()
                metrics_accum[name] += val

    num_images = len(img_sr_list)
    avg_metrics = {k: v / num_images for k, v in metrics_accum.items()}

    fid_value = 0.0
    if fid_metric is not None:
        logger.info("Calculating FID...")
        fid_value = fid_metric(args.gt_imgs, args.inp_imgs).item()
        avg_metrics['FID'] = fid_value

    summary_str = "; ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()])
    logger.info(f"\n================ Final Average Metrics ================\n{summary_str}\n=====================================================")

    # Write summary text file
    summary_txt_path = os.path.join(log_dir, "summary_metrics.txt")
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write(summary_str + "\n")
    logger.info(f"Summary written to {summary_txt_path}")

if __name__ == "__main__":
    main()
