import os

from cv2 import phase
if os.environ.get("USE_HF_MIRROR", "false").lower() in ("true", "1"):
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import sys
import gc
import lpips
import clip
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
import glob
import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler




from pathlib import Path
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate import DistributedDataParallelKwargs
from my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
from test_rcod_o import test_and_val
import pyiqa
from ram.models.ram_lora import ram
from ram import inference_ram as inference
from copy import copy
import logging

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
    parser.add_argument('--input_image', '-i', type=str, default='preset/datasets/test_dataset/input', help='path to the input image')
    # parser.add_argument('--output_dir', '-o', type=str, default='preset/datasets/test_dataset/output', help='the directory to save the output')
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain')
    parser.add_argument("--osediff_path", type=str, default='weights/rcod_o.pkl')
    parser.add_argument('--prompt', type=str, default='', help='user prompts')
    parser.add_argument('--ram_ft_path', type=str, default='weights/DAPE.pth')
    parser.add_argument('--save_prompts', type=bool, default=True)
    # merge lora
    parser.add_argument("--merge_and_unload_lora", default=False)
    # tile setting
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=96)
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)
    parser.add_argument("--inp_imgs", nargs="+", help="Path(s) to the input (SR) images directories.")
    parser.add_argument("--gt_imgs", nargs="+", required=True, help="Path(s) to the ground truth (GT) images directories.")
    parser.add_argument("--log_name", type=str, default='METRICS', help="Base name for the log files.")

    # training details #
    parser.add_argument("--output_dir", default='experience/rcod_o')
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
    parser.add_argument("--vae_state", type=str, default='lora', help="vae encoder state in training ")

    parser.add_argument("--tracker_project_name", type=str, default="train_rcod_o", help="The name of the wandb project to log to.")
    parser.add_argument('--dataset_txt_paths_list', type=parse_str_list, default=['datasets/LSDIR/file_paths.txt', 'datasets/ffhq/file_paths.txt'], help='Comma-separated list of dataset paths')
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






_loggers = {}

def setup_logger_single(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    """
    Sets up a logger with specified configurations, implementing a singleton pattern.
    """
    global _loggers

    if logger_name in _loggers:
        logger = _loggers[logger_name]
        return logger

    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        fmt='%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
        datefmt='%y-%m-%d %H:%M:%S'
    )
    logger.setLevel(level)

    if tofile:
        log_file = os.path.join(root, f"{phase}_{get_timestamp()}.log")
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    _loggers[logger_name] = logger
    return logger

def get_timestamp():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())    

from model_config import get_config
def main(args):
    OSEDiff_reg, OSEDiff_gen = get_config(args.model_name,phase='train')

        
    
    if args.train_data_type == 'orig':
        from dataloaders.realsr_dataset import PairedSROnlineTxtDataset  
    elif args.train_data_type == 'add_deg':
        from dataloaders.realsr_dataset_add_deg import PairedSROnlineTxtDataset
        
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    args.log = os.path.join(args.output_dir, 'metrics')
    args.inp_imgs = os.path.join(args.output_dir)
    try:
        args.log_name = args.inp_imgs[0].split('/')[8]
    except IndexError:
        args.log_name = 'METRICS'
        
    os.makedirs(args.log, exist_ok=True)
    logger = setup_logger_single('base', args.log, 'train', level=logging.INFO, screen=True, tofile=True)
    logger.info(args)
    
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
    fid_metric = pyiqa.create_metric('fid', device=device)
    
    
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
        
        # get ram model
    if args.text_prompt:
        DAPE = ram(pretrained=args.ram_path,
                pretrained_condition=args.ram_ft_path,
                image_size=384,
                vit='swin_l')
        DAPE.eval()
        DAPE.to("cuda")


        # set weight type
        DAPE = DAPE.to(dtype=weight_dtype)
    else:
        DAPE = None
        logger.info('do not use text model')


    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        args_file = os.path.join(args.output_dir, "config_args.txt")
        with open(args_file, "w") as f:
            for arg in vars(args):
                f.write(f"--{arg} {getattr(args, arg)}\n")

        sh_file = os.path.join(args.output_dir, "run_command.sh")
        with open(sh_file, "w") as f:
            f.write("#!/bin/bash\n")
            cmd = ["python", sys.argv[0]]
            for k, v in vars(args).items():
                if isinstance(v, list):
                    v = ",".join(map(str, v))
                cmd.append(f"--{k} {v}")
            f.write(" ".join(cmd) + "\n")
        os.chmod(sh_file, 0o755)


        
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    model_gen = OSEDiff_gen(args)
    model_gen.set_train()

    if args.osediff_path !='':
        print(f'===> Load osediff from {args.osediff_path}')
        osediff = torch.load(args.osediff_path)
        model_gen.load_ckpt(osediff)
            
            
    model_reg = OSEDiff_reg(args=args, accelerator=accelerator)
    model_reg.set_train()

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)

    # set vae adapter
    if args.vae_state == 'lora':
        model_gen.vae.set_adapter(['default_encoder'])
    # set gen adapter
    model_gen.unet.set_adapter(['default_encoder', 'default_decoder', 'default_others'])

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            model_gen.unet.enable_xformers_memory_efficient_attention()
            model_reg.unet_fix.enable_xformers_memory_efficient_attention()
            model_reg.unet_update.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        model_gen.unet.enable_gradient_checkpointing()
        model_reg.unet_fix.enable_gradient_checkpointing()
        model_reg.unet_update.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # make the optimizer
    layers_to_opt = []
    sum_param = []
    for n, _p in model_gen.unet.named_parameters():
        if "lora" in n:
            layers_to_opt.append(_p)
            sum_param.append(_p.numel())
    layers_to_opt += list(model_gen.unet.conv_in.parameters())
    sum_param_conv = sum(param.numel() for param in model_gen.unet.conv_in.parameters())

    if args.vae_state == 'lora':
        for n, _p in model_gen.vae.named_parameters():
            if "lora" in n:
                layers_to_opt.append(_p)
                sum_param.append(_p.numel())

    trainable_params = sum(sum_param) + sum_param_conv       
    print(f"Trainable parameters lora: {trainable_params}")
    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,)

    layers_to_opt_reg = []
    for n, _p in model_reg.unet_update.named_parameters():
        if "lora" in n:
            layers_to_opt_reg.append(_p)
            sum_param.append(_p.numel())
    trainable_params = sum(sum_param)     + sum_param_conv   
    print(f"Trainable parameters lora: {trainable_params}")
    optimizer_reg = torch.optim.AdamW(layers_to_opt_reg, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler_reg = get_scheduler(args.lr_scheduler, optimizer=optimizer_reg,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles, power=args.lr_power)

    dataset_train = PairedSROnlineTxtDataset(split="train", args=args)
    dataset_val = PairedSROnlineTxtDataset(split="test", args=args)
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)
    
    # init vlm model

    ram_transforms = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    model_vlm = ram(pretrained=args.ram_path,
            pretrained_condition=None,
            image_size=384,
            vit='swin_l')
    model_vlm.eval()
    model_vlm.to("cuda", dtype=torch.float16)

    # Prepare everything with our `accelerator`.
    model_gen, model_reg, optimizer, optimizer_reg, dl_train, lr_scheduler, lr_scheduler_reg = accelerator.prepare(
        model_gen, model_reg, optimizer, optimizer_reg, dl_train, lr_scheduler, lr_scheduler_reg
    )
    net_lpips = accelerator.prepare(net_lpips)
    # renorm with image net statistics


    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        args.dataset_txt_paths_list = str(args.dataset_txt_paths_list)
        args.dataset_prob_paths_list = str(args.dataset_prob_paths_list)
        
        tracker_config = dict(vars(args))
        tracker_config['gt_imgs'] = str(tracker_config['gt_imgs'])
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps",
        disable=not accelerator.is_local_main_process,)


    if args.single_cs_step_train:
        if args.n_step == 249:
            cs_bar = [999,0.7]
        elif args.n_step == 499:
            cs_bar = [0.7,0.5]
        elif args.n_step == 749:
            cs_bar = [0.5,0.3]
        elif args.n_step == 999:
            cs_bar = [0.3,-999]
                        
                        
    # start the training loop
    global_step = 0
    best_maniqa = 0
    best_step = 0
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            m_acc = [model_gen, model_reg]
            with accelerator.accumulate(*m_acc):
                x_src = batch["conditioning_pixel_values"]
                x_tgt = batch["output_pixel_values"]
                B, C, H, W = x_src.shape
                # get text prompts from GT
           
                x_tgt_ram = ram_transforms(x_tgt*0.5+0.5)
                
                if args.single_cs_step_train:
                    encoded_control = model_gen.vae.encode(x_src).latent_dist.sample() * model_gen.vae.config.scaling_factor # src
                    encoded_control_gt = model_gen.vae.encode(x_tgt).latent_dist.sample() * model_gen.vae.config.scaling_factor
                    cs_gt_bc = model_gen.cosine_similarity(encoded_control.detach(), encoded_control_gt.detach())
                
                    if cs_gt_bc <= cs_bar[0] and cs_gt_bc > cs_bar[1]:
                        continue
                    else:
                        pass
                caption = ['']
                caption2 = inference(x_tgt_ram.to(dtype=torch.float16), model_vlm)
                batch["prompt"] = [f'{each_caption}' for each_caption in caption]
                batch["prompt_vsd"] = [f'{each_caption}' for each_caption in caption2]
                # forward pass
                x_tgt_pred, latents_pred = model_gen(x_src, batch=batch, args=args)
                # Reconstruction loss
                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean") * args.lambda_l2
                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean() * args.lambda_lpips
                loss = loss_l2 + loss_lpips
                


                # KL loss
                if 'vsd_group' in args.model_name:
                    if torch.cuda.device_count() > 1:
                        prompt_embeds = model_reg.module.encode_prompt(batch["prompt_vsd"]).to(torch.float32)
                        neg_prompt_embeds = model_reg.module.encode_prompt(batch["neg_prompt"]).to(torch.float32)
                        loss_kl = model_reg.module.distribution_matching_loss(latents=latents_pred, prompt_embeds=prompt_embeds, 
                                                                              neg_prompt_embeds=neg_prompt_embeds, args=args,model_gen_timesteps=model_gen.module.timesteps) * args.lambda_vsd
                    else:
                        prompt_embeds = model_reg.encode_prompt(batch["prompt_vsd"]).to(torch.float32)
                        neg_prompt_embeds = model_reg.encode_prompt(batch["neg_prompt"]).to(torch.float32)
                        loss_kl = model_reg.distribution_matching_loss(latents=latents_pred, prompt_embeds=prompt_embeds,
                                                                       neg_prompt_embeds=neg_prompt_embeds, args=args,model_gen_timesteps=model_gen.timesteps) * args.lambda_vsd
                else:
                    if torch.cuda.device_count() > 1:
                        prompt_embeds = model_reg.module.encode_prompt(batch["prompt_vsd"]).to(torch.float32)
                        neg_prompt_embeds = model_reg.module.encode_prompt(batch["neg_prompt"]).to(torch.float32)
                        loss_kl = model_reg.module.distribution_matching_loss(latents=latents_pred, prompt_embeds=prompt_embeds, neg_prompt_embeds=neg_prompt_embeds, args=args) * args.lambda_vsd
                    else:
                        prompt_embeds = model_reg.encode_prompt(batch["prompt_vsd"]).to(torch.float32)
                        neg_prompt_embeds = model_reg.encode_prompt(batch["neg_prompt"]).to(torch.float32)
                        loss_kl = model_reg.distribution_matching_loss(latents=latents_pred, prompt_embeds=prompt_embeds, neg_prompt_embeds=neg_prompt_embeds, args=args) * args.lambda_vsd
                loss = loss + loss_kl
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                """
                diff loss: let lora model closed to generator 
                """
                if torch.cuda.device_count() > 1:
                    loss_d = model_reg.module.diff_loss(latents=latents_pred, prompt_embeds=prompt_embeds, args=args)*args.lambda_vsd_lora
                else:
                    loss_d = model_reg.diff_loss(latents=latents_pred, prompt_embeds=prompt_embeds, args=args)*args.lambda_vsd_lora
                accelerator.backward(loss_d)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model_reg.parameters(), args.max_grad_norm)
                optimizer_reg.step()
                lr_scheduler_reg.step()
                optimizer_reg.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    
                    logs = {}
                    # log all the losses
                    logs["loss_d"] = loss_d.detach().item()
                    logs["loss_kl"] = loss_kl.detach().item()
                    logs["loss_l2"] = loss_l2.detach().item()
                    logs["loss_lpips"] = loss_lpips.detach().item()
                    progress_bar.set_postfix(**logs)

                    # checkpoint the model
                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(model_gen).save_model(outf)
                        logger.info(f"Saving model at step {global_step}")

                        # make the output dir
                        args.osediff_path = outf
                        if accelerator.is_main_process:
                            test_args = copy(args)
                            
                            avg_metrics_str = test_and_val(test_args,DAPE,iqa_metrics=iqa_metrics,fid_metrics=fid_metric)
                            maniq = avg_metrics_str['MANIQA']
                            if maniq > best_maniqa:
                                best_maniqa = maniq
                                best_step = global_step
                            logger.info(f"best maniqa: {best_maniqa} at step {best_step}")

                    accelerator.log(logs, step=global_step)

if __name__ == "__main__":
    args = parse_args()
    args.text_prompt = False
    main(args)
