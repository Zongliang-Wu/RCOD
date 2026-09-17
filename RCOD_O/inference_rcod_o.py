import os
import sys
import argparse
import glob
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from tqdm import tqdm

from model_config import get_config
from my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix

def parse_args():
    parser = argparse.ArgumentParser(description="RCOD_O (OSEDiff Base) Inference Script")
    parser.add_argument("--input_image", "-i", type=str, required=True, help="Path to input image or directory of images")
    parser.add_argument("--output_dir", "-o", type=str, default="results/RCOD_O", help="Path to output directory")
    default_weight = "weights/rcod_o.pkl" if os.path.exists("weights/rcod_o.pkl") else None
    default_sd = "weights/stable-diffusion-2-1-base" if os.path.exists("weights/stable-diffusion-2-1-base") else "stabilityai/stable-diffusion-2-1-base"
    parser.add_argument("--osediff_path", type=str, default=default_weight, required=(default_weight is None), help="Path to trained RCOD_O checkpoint (.pkl)")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=default_sd, help="Base SD model path or HuggingFace repo ID")
    parser.add_argument("--timestep", "-t", type=int, default=249, choices=[249, 499, 749, 250, 500, 750], help="Timestep controlling realism: 249 (Fidelity), 499 (Neutral), 749 (Realism)")
    parser.add_argument("--upscale", type=int, default=4, help="Upscale factor")
    parser.add_argument("--process_size", type=int, default=512, help="Target patch processing size")
    parser.add_argument("--align_method", type=str, default="adain", choices=["adain", "wavelet", "nofix"], help="Color fix method")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=96)
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)
    parser.add_argument("--merge_and_unload_lora", action="store_true", help="Merge LoRA before inference")
    parser.add_argument("--device", type=str, default="cuda:0")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Map friendly timesteps to exact model timesteps
    ts_map = {250: 249, 500: 499, 750: 749, 249: 249, 499: 499, 749: 749}
    actual_step = ts_map[args.timestep]
    args.n_step = actual_step
    args.n_div = 3
    args.vae_state = "lora"
    args.model_name = "fix_sche2_notext_clip_vsd_group"
    
    if torch.cuda.is_available() and "cuda" in args.device:
        device_idx = int(args.device.split(":")[-1]) if ":" in args.device else 0
        torch.cuda.set_device(device_idx)
        device = torch.device(f"cuda:{device_idx}")
    else:
        device = torch.device("cpu")
    
    # Load model
    print(f"Loading RCOD_O model from {args.osediff_path} on {device}...")
    OSEDiff_test = get_config(args.model_name, phase="test")
    model = OSEDiff_test(args)
    model.timesteps = torch.tensor([actual_step], device=device).long()
    
    # Prepare inputs
    if os.path.isdir(args.input_image):
        image_paths = sorted(glob.glob(os.path.join(args.input_image, "*.[pP][nN][gG]")) + 
                             glob.glob(os.path.join(args.input_image, "*.[jJ][pP][gG]")) + 
                             glob.glob(os.path.join(args.input_image, "*.[jJ][pP][eE][gG]")))
    else:
        image_paths = [args.input_image]
        
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Found {len(image_paths)} image(s) to process. Timestep={actual_step} (input: {args.timestep}).")
    
    for img_path in tqdm(image_paths, desc="Processing"):
        fname = os.path.basename(img_path)
        out_path = os.path.join(args.output_dir, fname)
        
        input_image = Image.open(img_path).convert("RGB")
        ori_w, ori_h = input_image.size
        rscale = args.upscale
        resize_flag = False
        
        if ori_w < args.process_size // rscale or ori_h < args.process_size // rscale:
            scale = (args.process_size // rscale) / min(ori_w, ori_h)
            input_image = input_image.resize((int(scale * ori_w), int(scale * ori_h)))
            resize_flag = True
            
        input_image = input_image.resize((input_image.size[0] * rscale, input_image.size[1] * rscale))
        new_w = input_image.width - input_image.width % 8
        new_h = input_image.height - input_image.height % 8
        input_image = input_image.resize((new_w, new_h), Image.LANCZOS)
        
        lq = transforms.ToTensor()(input_image).unsqueeze(0).to(device)
        lq = (lq * 2 - 1).half() if args.mixed_precision == "fp16" else (lq * 2 - 1).float()
        
        with torch.no_grad():
            output_image = model(lq, prompt=None)
            
        output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
        
        if args.align_method == "adain":
            output_pil = adain_color_fix(target=output_pil, source=input_image)
        elif args.align_method == "wavelet":
            output_pil = wavelet_color_fix(target=output_pil, source=input_image)
            
        if resize_flag:
            output_pil = output_pil.resize((int(args.upscale * ori_w), int(args.upscale * ori_h)))
            
        output_pil.save(out_path)

    print(f"Done! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
