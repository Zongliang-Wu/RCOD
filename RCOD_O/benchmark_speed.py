import os
import sys
import time
import argparse
import torch
from tqdm import tqdm

from rcod_o_model import RCOD_O_inference_time, OSEDiff_inference_time


def parse_args():
    parser = argparse.ArgumentParser(description="RCOD-O Model Inference Speed Benchmark")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="weights/stable-diffusion-2-1-base" if os.path.exists("weights/stable-diffusion-2-1-base") else "stabilityai/stable-diffusion-2-1-base",
        help="Stable Diffusion 2.1 base model path or HuggingFace repo ID",
    )
    parser.add_argument(
        "--model_path",
        "--osediff_path",
        dest="osediff_path",
        type=str,
        default="weights/RCOD_O/rcod_o.pkl" if os.path.exists("weights/RCOD_O/rcod_o.pkl") else "preset/models/osediff.pkl",
        help="Path to RCOD-O checkpoint (.pkl)",
    )
    parser.add_argument("--mixed_precision", type=str, choices=["fp16", "fp32"], default="fp16", help="Mixed precision mode")
    parser.add_argument("--merge_and_unload_lora", action="store_true", default=True, help="Merge LoRA weights before inference")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use for inference")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--process_size", type=int, default=512, help="Resolution (H=W) for benchmark inputs")
    parser.add_argument("--inference_iterations", type=int, default=20, help="Number of timed inference iterations")
    parser.add_argument("--warmup_iterations", type=int, default=5, help="Number of warm-up iterations")
    
    return parser.parse_args()


def main():
    args = parse_args()
    args.merge_and_unload_lora = True

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16 if (args.mixed_precision == "fp16" and device.type == "cuda") else torch.float32

    print(f"Loading RCOD-O model from {args.osediff_path}...")
    model = RCOD_O_inference_time(args)
    model.to(device)
    model.eval()

    # Pre-allocate input tensors
    input_tensors = torch.randn(
        (args.inference_iterations, args.batch_size, 3, args.process_size, args.process_size),
        device=device,
        dtype=weight_dtype,
    )

    # Warm-up
    print(f"Running {args.warmup_iterations} warm-up iterations...")
    for _ in range(args.warmup_iterations):
        lq = torch.randn((args.batch_size, 3, args.process_size, args.process_size), device=device, dtype=weight_dtype)
        with torch.no_grad():
            _ = model(lq)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking inference over {args.inference_iterations} iterations...")
    total_time = 0.0
    for idx in tqdm(range(args.inference_iterations), desc="Benchmarking"):
        lq = input_tensors[idx]
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.no_grad():
            _ = model(lq)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        total_time += (end_time - start_time)

    avg_time = total_time / args.inference_iterations
    fps = 1.0 / avg_time if avg_time > 0 else 0.0
    print(f"\nResults on {device} ({args.process_size}x{args.process_size}, {args.mixed_precision}):")
    print(f"Average latency : {avg_time * 1000:.2f} ms ({avg_time:.4f} s)")
    print(f"Throughput      : {fps:.2f} FPS")


if __name__ == "__main__":
    main()