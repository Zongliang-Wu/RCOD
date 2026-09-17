import os

from cv2 import phase
if os.environ.get("USE_HF_MIRROR", "false").lower() in ("true", "1"):
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse

import time
def parse_float_list(arg):
    try:
        return [float(x) for x in arg.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("List elements should be floats")

def parse_int_list(arg):
    try:
        return [int(x) for x in arg.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("List elements should be integers")

def parse_str_list(arg):
    return arg.split(',')

def parse_args(input_args=None):
    """
    Parses command-line arguments used for configuring an paired session (pix2pix-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='osediff',)
    parser.add_argument("--train_data_type", type=str, default='orig',) # add_deg
    

    parser.add_argument("--revision", type=str, default=None,)
    parser.add_argument("--variant", type=str, default=None,)
    parser.add_argument("--tokenizer_name", type=str, default=None)


    # val dataset
    parser.add_argument('--input_image', '-i', type=str, default='assets/samples', help='path to the input image')
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain')
    parser.add_argument("--osediff_path", type=str, default='weights/rcod_o.pkl')
    parser.add_argument('--prompt', type=str, default='', help='user prompts')
    parser.add_argument('--ram_ft_path', type=str, default='weights/DAPE.pth')
    parser.add_argument('--save_prompts', type=bool, default=True)
    # merge lora
    parser.add_argument("--merge_and_unload_lora", default=False) # merge lora weights before inference
    # tile setting
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224) 
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024) 
    parser.add_argument("--latent_tiled_size", type=int, default=96) 
    parser.add_argument("--latent_tiled_overlap", type=int, default=32) 
    parser.add_argument("--inp_imgs",nargs="+",help="Path(s) to the input (SR) images directories.")
    parser.add_argument("--gt_imgs", nargs="+",required=True,help="Path(s) to the ground truth (GT) images directories.")
    parser.add_argument("--log_name",type=str,default='METRICS',help="Base name for the log files.")
    
    

    
    # training details #
    parser.add_argument("--output_dir", '-o', type=str, default='experience/osediff')
    parser.add_argument("--seed", type=int, default=123, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512,)
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=10000)
    parser.add_argument("--max_train_steps", type=int, default=100000,)
    parser.add_argument("--checkpointing_steps", type=int, default=2000,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true",)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=2,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--report_to", type=str, default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true",)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--vae_state",type=str,default='lora',help="vae encoder state in training ")
    
    parser.add_argument("--tracker_project_name", type=str, default="train_rcod_o", help="The name of the wandb project to log to.")
    parser.add_argument('--dataset_txt_paths_list', type=parse_str_list, default=['datasets/LSDIR/file_paths.txt', 'datasets/ffhq/file_paths.txt'], help='Comma-separated paths to dataset index files')
    parser.add_argument('--dataset_prob_paths_list', type=parse_int_list, default=[1, 1], help='A comma-separated list of probabilities')
    parser.add_argument("--deg_file_path", default="params_realesrgan.yml", type=str)
    parser.add_argument("--pretrained_model_name_or_path", default="stabilityai/stable-diffusion-2-1-base", type=str)
    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_lpips", default=2.0, type=float)
    parser.add_argument("--lambda_vsd", default=1.0, type=float)
    parser.add_argument("--lambda_vsd_lora", default=1.0, type=float)
    parser.add_argument("--neg_prompt", default="painting, oil painting, illustration, drawing, art, sketch, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth", type=str)
    parser.add_argument("--cfg_vsd", default=7.5, type=float)

    # lora setting
    parser.add_argument("--lora_rank", default=4, type=int)
    # ram path
    parser.add_argument('--ram_path', type=str, default="ram_swin_large_14m.pth", help='Path to RAM model')
    parser.add_argument('--val_4step', type=bool, default=True)
    parser.add_argument('--single_cs_step_train', type=bool, default=False, help='only select one cs score')
    parser.add_argument('--text_prompt', type=bool, default=False, help='')
    parser.add_argument("--n_div", default=3, type=int)
    parser.add_argument("--n_step", type=int, default=249,)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


