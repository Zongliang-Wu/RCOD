import os
import sys
import argparse
import glob
import math
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from model_config import get_config
from de_net import DEResNet
from utils.wavelet_color import wavelet_color_fix, adain_color_fix

def parse_args():
    parser = argparse.ArgumentParser(description="RCOD_S (S3Diff Base) Inference Script")
    parser.add_argument("--input_image", "-i", type=str, required=True, help="Path to input image or directory of images")
    parser.add_argument("--output_dir", "-o", type=str, default="results/RCOD_S", help="Path to output directory")
    default_weight = "weights/rcod_s.pkl" if os.path.exists("weights/rcod_s.pkl") else None
    default_mem = "weights/rcod_s_mem.pkl" if os.path.exists("weights/rcod_s_mem.pkl") else ""
    default_sd = "weights/sd-turbo" if os.path.exists("weights/sd-turbo") else "stabilityai/sd-turbo"
    parser.add_argument("--pretrained_path", type=str, default=default_weight, required=(default_weight is None), help="Path to RCOD_S checkpoint (.pkl)")
    parser.add_argument("--mlp_path", type=str, default=default_mem, help="Path to MEM checkpoint (.pkl), required if --adaptive")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_de_net = os.path.join(script_dir, "assets/mm-realsr/de_net.pth")
    parser.add_argument("--de_net_path", type=str, default=default_de_net, help="Path to degradation estimation net (de_net.pth)")
    parser.add_argument("--sd_path", type=str, default=default_sd, help="Base SD-Turbo model directory or HuggingFace repo ID")
    parser.add_argument("--timestep", "-t", type=int, default=249, choices=[249, 499, 749, 250, 500, 750], help="Timestep: 249 (Fidelity), 499 (Neutral), 749 (Realism)")
    parser.add_argument("--adaptive", action="store_true", help="Enable MEM-based adaptive timestep selection (RCOD_S-Adap)")
    parser.add_argument("--upscale", type=int, default=4, help="Upscale factor")
    parser.add_argument("--align_method", type=str, default="wavelet", choices=["wavelet", "adain", "nofix"], help="Color fix method (S3Diff defaults to wavelet)")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--lora_rank_unet", type=int, default=32)
    parser.add_argument("--lora_rank_vae", type=int, default=16)
    parser.add_argument("--latent_tiled_size", type=int, default=96)
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda:0")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    if torch.cuda.is_available() and "cuda" in args.device:
        device_idx = int(args.device.split(":")[-1]) if ":" in args.device else 0
        torch.cuda.set_device(device_idx)
        device = torch.device(f"cuda:{device_idx}")
    else:
        device = torch.device("cpu")
        
    ts_map = {250: 249, 500: 499, 750: 749, 249: 249, 499: 499, 749: 749}
    manual_step = ts_map[args.timestep]
    args.n_step = manual_step
    args.n_div = 3
    args.model_name = "deg_notext_clip"
    args.model_name_score = "deg3_mlp"
    args.enable_xformers_memory_efficient_attention = True
    args.gradient_checkpointing = False
    args.allow_tf32 = True
    args.padding_offset = 32
    
    print(f"Loading RCOD_S generator from {args.pretrained_path}...")
    S3DiffTile = get_config(model_name=args.model_name, phase="test")
    net_sr = S3DiffTile(
        lora_rank_unet=args.lora_rank_unet,
        lora_rank_vae=args.lora_rank_vae,
        sd_path=args.sd_path,
        pretrained_path=args.pretrained_path,
        args=args
    ).to(device)
    net_sr.set_eval()
    
    print(f"Loading degradation estimator from {args.de_net_path}...")
    net_de = DEResNet(num_in_ch=3, num_degradation=2, use_feat=True).to(device)
    net_de.load_model(args.de_net_path)
    net_de.eval()
    
    net_mlp = None
    if args.adaptive:
        if not args.mlp_path or not os.path.exists(args.mlp_path):
            raise ValueError("Must provide a valid --mlp_path for adaptive inference!")
        print(f"Loading MEM (Metric Estimation Module) from {args.mlp_path}...")
        NetMLP = get_config(model_name=args.model_name_score, phase="train")
        net_mlp = NetMLP(pretrained_path=args.mlp_path, args=args, use_clip_score=False).to(device)
        net_mlp.eval()
        
    # Prepare inputs
    if os.path.isdir(args.input_image):
        image_paths = sorted(glob.glob(os.path.join(args.input_image, "*.[pP][nN][gG]")) + 
                             glob.glob(os.path.join(args.input_image, "*.[jJ][pP][gG]")) + 
                             glob.glob(os.path.join(args.input_image, "*.[jJ][pP][eE][gG]")))
    else:
        image_paths = [args.input_image]
        
    os.makedirs(args.output_dir, exist_ok=True)
    mode_str = "Adaptive (MEM)" if args.adaptive else f"Manual (timestep={manual_step})"
    print(f"Found {len(image_paths)} image(s) to process. Mode: {mode_str}")
    
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    
    for img_path in tqdm(image_paths, desc="Processing"):
        fname = os.path.basename(img_path)
        out_path = os.path.join(args.output_dir, fname)
        
        im_lr = Image.open(img_path).convert("RGB")
        im_lr_tensor = to_tensor(im_lr).unsqueeze(0).to(device)
        
        ori_h, ori_w = im_lr_tensor.shape[2:]
        im_lr_resize = F.interpolate(
            im_lr_tensor,
            size=(ori_h * args.upscale, ori_w * args.upscale),
            mode="bilinear",
            align_corners=False
        )
        im_lr_resize = im_lr_resize.contiguous()
        im_lr_resize_norm = torch.clamp(im_lr_resize * 2 - 1.0, -1.0, 1.0)
        resize_h, resize_w = im_lr_resize_norm.shape[2:]
        
        pad_h = (math.ceil(resize_h / 64)) * 64 - resize_h
        pad_w = (math.ceil(resize_w / 64)) * 64 - resize_w
        im_lr_resize_norm = F.pad(im_lr_resize_norm, pad=(0, pad_w, 0, pad_h), mode="reflect")
        
        with torch.no_grad():
            deg_score = net_de(im_lr_tensor)
            if args.adaptive:
                cs_pred = net_mlp(deg_score[1], None)
                step_list = net_sr.div_step(cs_pred)
                net_sr.timesteps = torch.tensor(step_list, device=device).long()
            else:
                net_sr.timesteps = torch.tensor([manual_step], device=device).long()
                
            x_tgt_pred = net_sr(im_lr_resize_norm, deg_score[0])
            x_tgt_pred = x_tgt_pred[:, :, :resize_h, :resize_w]
            out_img = (x_tgt_pred * 0.5 + 0.5).cpu().detach()
            
        output_pil = to_pil(out_img[0])
        if args.align_method == "wavelet":
            lr_ref_pil = to_pil(im_lr_resize[0].cpu().detach())
            output_pil = wavelet_color_fix(output_pil, lr_ref_pil)
        elif args.align_method == "adain":
            lr_ref_pil = to_pil(im_lr_resize[0].cpu().detach())
            output_pil = adain_color_fix(output_pil, lr_ref_pil)
            
        output_pil.save(out_path)

    print(f"Done! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
