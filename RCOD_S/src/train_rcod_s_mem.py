import os

from sklearn import get_config
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
if os.environ.get("USE_HF_MIRROR", "false").lower() in ("true", "1"):
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import sys
sys.path.append(os.getcwd())


import gc
import lpips
import clip
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers

from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from tqdm import tqdm
import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler
from utils import util_image
from utils.wavelet_color import wavelet_color_fix, adain_color_fix
import math
from utils import util_image
from pathlib import Path
import logging
from de_net import DEResNet
from my_utils.training_utils2 import parse_args_paired_training, PairedDataset, degradation_proc, PlainDataset
from my_utils.testing_utils import evaluate
from model_config import get_config
import time, pyiqa
import threading
_loggers = {}

def setup_logger_single(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    """
    Sets up a logger with specified configurations, implementing a singleton pattern.
    """
    global _loggers

    if logger_name in _loggers:
        logger = _loggers[logger_name]
        logger.info(f"Logger '{logger_name}' already initialized, returning existing logger.")
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

def test_and_val(S3DiffTile,sd_path,outf,net_de,dl_val,config,metric_dict,metric_paired_dict,
                 fid_metric,device,args,logger,global_step,best_maniq,best_step,NetMLP):
    net_sr_test = S3DiffTile(lora_rank_unet=args.lora_rank_unet,
                                lora_rank_vae=args.lora_rank_vae, 
                                sd_path=sd_path, pretrained_path=args.pretrained_path, args=args)
    net_sr_test = net_sr_test.cuda()
    net_sr_test.set_eval()
    
    if 'notext2' in args.model_name:
        use_clip_score = True
    else:
        use_clip_score = False
    net_mlp_test = NetMLP(pretrained_path=outf, args=args,use_clip_score=use_clip_score)
    net_mlp_test = net_mlp_test.cuda()
    net_mlp_test.set_eval()

    l_l2, l_lpips = [], []

    val_count = 0

    step_list = [249] # 999
    cs_gt_list = []
    cs_pred_list = []

    for time_step in step_list: # 999

        current_output_dir = os.path.join(args.output_dir, f'{config.validation.lr_path.split("/")[-3]}')
        os.makedirs(current_output_dir, exist_ok=True)
        for step, batch_val in enumerate(tqdm(dl_val)):

            
            im_lr = batch_val['lr'].to(device)
            im_lr = im_lr.to(memory_format=torch.contiguous_format).float()

            ori_h, ori_w = im_lr.shape[2:]
            im_lr_resize = F.interpolate(
                im_lr,
                size=(ori_h * config.sf,
                    ori_w * config.sf),
                mode='bilinear',
                align_corners=False # align_corners with this model causes the output to be shifted, presumably due to training without align_corners
            )

            im_lr_resize = im_lr_resize.contiguous()
            im_lr_resize_norm = im_lr_resize * 2 - 1.0
            im_lr_resize_norm = torch.clamp(im_lr_resize_norm, -1.0, 1.0)
            resize_h, resize_w = im_lr_resize_norm.shape[2:]

            pad_h = (math.ceil(resize_h / 64)) * 64 - resize_h
            pad_w = (math.ceil(resize_w / 64)) * 64 - resize_w
            im_lr_resize_norm = F.pad(im_lr_resize_norm, pad=(0, pad_w, 0, pad_h), mode='reflect')
            
            
            
            # img_tgt_resize_norm = 

            B = im_lr_resize.size(0)
            with torch.no_grad():
                # forward pass
                deg_score = net_de(im_lr)
                if 'notext' in args.model_name:
                    # clip_score = net_sr_test.text_encoder(im_lr_resize_norm)
                    clip_score = None
                else:
                    clip_score = None
                    
                cs_pred = net_mlp_test(deg_score[1],clip_score)
                step_list = net_sr_test.div_step(cs_pred)
                net_sr_test.timesteps = torch.tensor(step_list, device=device).long()
                cs_pred_list.append(cs_pred.item())
                
                
                if 'notext' in args.model_name:
                    x_tgt_pred = net_sr_test(im_lr_resize_norm, deg_score[0])
                else:
                    pos_tag_prompt = [args.pos_prompt for _ in range(B)]
                    neg_tag_prompt = [args.neg_prompt for _ in range(B)]
                    x_tgt_pred =net_sr_test(im_lr_resize_norm, deg_score[0], pos_prompt=pos_tag_prompt, neg_prompt=neg_tag_prompt)
                x_tgt_pred = x_tgt_pred[:, :, :resize_h, :resize_w]
                out_img = (x_tgt_pred * 0.5 + 0.5).cpu().detach()
                
                # lq_latent = net_sr_test.vae.encode(im_lr_resize_norm).latent_dist.sample() * net_sr_test.vae.config.scaling_factor
                # gt_latent = net_sr_test.vae.encode(img_tgt_resize_norm).latent_dist.sample() * net_sr_test.vae.config.scaling_factor
                # cs_gt = cosine_similarity(lq_latent,gt_latent)
                
                # cs_gt_list.append(cs_gt)

            output_pil = transforms.ToPILImage()(out_img[0])

            if args.align_method == 'nofix':
                output_pil = output_pil
            else:
                im_lr_resize = transforms.ToPILImage()(im_lr_resize[0].cpu().detach())
                if args.align_method == 'wavelet':
                    output_pil = wavelet_color_fix(output_pil, im_lr_resize)
                elif args.align_method == 'adain':
                    output_pil = adain_color_fix(output_pil, im_lr_resize)

            fname = batch_val['lr_path'][0].split('/')[-1]
            outf = os.path.join(current_output_dir,fname)
            output_pil.save(outf)

        avg_metrics_str, result_dict, num_images = evaluate(current_output_dir, config.validation.gt_path , 
                                                            None,metric_dict,metric_paired_dict,fid_metric=fid_metric,device=device)
        # out_t = os.path.join(args.output_dir, 'results.txt')
        dir_name = 'realsr'
        logger.info(f"\n===== Average Metrics for [{dir_name}] | step={str(time_step)}=====\n{avg_metrics_str}\n")
        
        

        if result_dict['maniqa'] > best_maniq:
            best_maniq = result_dict['maniqa']
            best_step = global_step

        logger.info(f'Best Maniqa: {best_maniq} at {best_step}')
            
    del net_sr_test
                        
def main(args):

    # init and save configs
    config = OmegaConf.load(args.base_config)


    sd_path = args.sd_path
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    

    args.log = os.path.join(args.output_dir, 'metrics')
    args.inp_imgs = os.path.join(args.output_dir)
    try:
        args.log_name = args.inp_imgs[0].split('/')[8]
    except IndexError:
        args.log_name = 'METRICS'
        
    os.makedirs(args.log, exist_ok=True)
    logger = setup_logger_single('base', args.log, f'train_{args.model_name}', level=logging.INFO, screen=True, tofile=True)
    logger.info(args)

    
    metric_paired_dict = {}
    metric_paired_dict["psnr"]=pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(device)
    metric_paired_dict["ssim"]=pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr' ).to(device)
    metric_paired_dict["lpips"]=pyiqa.create_metric('lpips').to(device)
    metric_paired_dict["dists"]=pyiqa.create_metric('dists').to(device)



    metric_dict = {}
    metric_dict["niqe"] = pyiqa.create_metric('niqe').to(device)
    metric_dict["musiq"] = pyiqa.create_metric('musiq').to(device)
    metric_dict["maniqa"] = pyiqa.create_metric('maniqa-pipal').to(device)
    metric_dict["clipiqa"] = pyiqa.create_metric('clipiqa').to(device)
    
    
    if args.reduce_metrics:
        fid_metric = None
    else:
        fid_metric = pyiqa.create_metric('fid')
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save args to text file
        args_file = os.path.join(args.output_dir, "config_args.txt")
        with open(args_file, "w") as f:
            for arg in vars(args):
                f.write(f"--{arg} {getattr(args, arg)}\n")
        
        # Save run command script
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

    # initialize degradation estimation network
    if 'notext2' in args.model_name or 'deg3' in args.model_name or args.model_name_score!='No':
        use_feat = True
        
    else:
        use_feat = False
    net_de = DEResNet(num_in_ch=3, num_degradation=2,use_feat=use_feat)
    net_de.load_model(args.de_net_path)
    net_de = net_de.cuda()
    net_de.eval()

    # initialize net_sr

    S3DiffTile = get_config(model_name=args.model_name,phase='test')
    net_sr = S3DiffTile(lora_rank_unet=args.lora_rank_unet, 
                    lora_rank_vae=args.lora_rank_vae, sd_path=sd_path, pretrained_path=args.pretrained_path, args=args)
    net_sr.cuda()
    net_sr.set_eval()
    
    NetMLP = get_config(model_name=args.model_name_score,phase='train')
    if 'notext2' in args.model_name :
        use_clip_score = True
    else:
        use_clip_score = False
    net_mlp = NetMLP(pretrained_path=args.mlp_path, args=args,use_clip_score=use_clip_score)
    net_mlp.set_train()


    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_sr.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_sr.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True


    # make the optimizer
    layers_to_opt = []
    layers_to_opt = list(net_mlp.parameters())
   
    dataset_train = PairedDataset(config.train)
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    dataset_val = PlainDataset(config.validation)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)


    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,)


    # Prepare everything with our `accelerator`.
    net_mlp,  optimizer , dl_train, lr_scheduler = accelerator.prepare(
        net_mlp,  optimizer,  dl_train, lr_scheduler,  
    )
    net_de = accelerator.prepare(net_de)
    # # renorm with image net statistics
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move al networksr to device and cast to weight_dtype
    net_sr.to(accelerator.device, dtype=weight_dtype)
    net_de.to(accelerator.device, dtype=weight_dtype)
    net_mlp.to(accelerator.device, dtype=weight_dtype)


    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps",
        disable=not accelerator.is_local_main_process,)



    # start the training loop
    global_step = 0
    best_maniq = 0
    best_step = 0
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            l_acc = [net_sr, net_mlp]
            with accelerator.accumulate(*l_acc):
                x_src, x_tgt, x_ori_size_src = degradation_proc(config, batch, accelerator.device)
                B, C, H, W = x_src.shape
                with torch.no_grad():
                    deg_score = net_de(x_ori_size_src.detach())
                    if 'notext2' in args.model_name or use_feat==True:
                        deg_score[0],deg_score[1] = deg_score[0].detach(),deg_score[1].detach()
                    else:
                        deg_score = deg_score.detach()
                if 'notext' in args.model_name:
                    mixed_tgt = x_tgt
                    clip_score = None
                    if global_step == 0:
                        y_sr = net_sr(x_src, deg_score[0])
                    lq_latent = net_sr.vae.encode(x_src).latent_dist.sample() * net_sr.vae.config.scaling_factor
                    gt_latent = net_sr.vae.encode(mixed_tgt).latent_dist.sample() * net_sr.vae.config.scaling_factor
                    cs_gt = net_sr.cosine_similarity(lq_latent,gt_latent)
                    
                else:
                    pos_tag_prompt = [args.pos_prompt for _ in range(B)]                
                    neg_tag_prompt = [args.neg_prompt for _ in range(B)]

                    neg_probs = torch.rand(B).to(accelerator.device)
                    
                    # build mixed prompt and target
                    mixed_tag_prompt = [_neg_tag if p_i < args.neg_prob else _pos_tag for _neg_tag, _pos_tag, p_i in zip(neg_tag_prompt, pos_tag_prompt, neg_probs)]
                    neg_probs = neg_probs.reshape(B, 1, 1, 1)
                    mixed_tgt = torch.where(neg_probs < args.neg_prob, x_src, x_tgt)

                    clip_score = None
                    if global_step  == 0:
                        y_sr = net_sr(x_src,deg_score[0],pos_tag_prompt,neg_tag_prompt)
                    lq_latent = net_sr.vae.encode(x_src).latent_dist.sample() * net_sr.vae.config.scaling_factor
                    gt_latent = net_sr.vae.encode(mixed_tgt).latent_dist.sample() * net_sr.vae.config.scaling_factor
                    cs_gt = cosine_similarity(lq_latent,gt_latent)
                    step_gt = net_sr.div_step(cs_gt)
                
                
                cs_pred = net_mlp(deg_score[1],clip_score)[0]


                loss = F.mse_loss(cs_pred,cs_gt.detach())

                accelerator.backward(loss, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

        

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {}
                    logs["loss"] = loss.detach().item()

                    progress_bar.set_postfix(**logs)

                    # checkpoint the model
                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(net_mlp).save_model(outf)
                        logger.info(f"Saving model at step {global_step}")

                    # compute validation set FID, L2, LPIPS, CLIP-SIM
                    if global_step % args.checkpointing_steps == 1:
                        # validation_thread = threading.Thread(target=test_and_val,
                        #                                         args=(S3DiffTile,sd_path,outf,net_de,dl_val,
                        #                                               config,metric_dict,metric_paired_dict,
                        #                                             fid_metric,device,args,logger,global_step,
                        #                                             best_maniq,best_step,NetMLP))
                        test_and_val(S3DiffTile,sd_path,outf,net_de,dl_val,
                                                                      config,metric_dict,metric_paired_dict,
                                                                    fid_metric,device,args,logger,global_step,
                                                                    best_maniq,best_step,NetMLP)
                        # validation_thread.start()
                        # test_and_val()
                        # gc.collect()
                        # torch.cuda.empty_cache()
                    accelerator.log(logs, step=global_step)
def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    x1_ = x1.reshape(x1.shape[0], -1)
    x2_ = x2.reshape(x2.shape[0], -1)
    output = torch.nn.functional.cosine_similarity(x1_, x2_, dim=1)
    return output



if __name__ == "__main__":
    args = parse_args_paired_training()
    main(args)
