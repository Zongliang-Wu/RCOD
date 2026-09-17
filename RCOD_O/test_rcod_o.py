import os
if os.environ.get("USE_HF_MIRROR", "false").lower() in ("true", "1"):
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.path.append(os.getcwd())
import glob
import argparse
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image

from my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix

from ram.models.ram_lora import ram
from ram import inference_ram as inference
import logging
import pyiqa
from basicsr.utils import img2tensor
import time
import cv2
from tqdm import tqdm
from accelerate import Accelerator
tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

ram_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])



def get_validation_prompt(args, image, model, device='cuda',weight_dtype=torch.float16):
    validation_prompt = ""
    lq = tensor_transforms(image).unsqueeze(0).to(device)
    lq_ram = ram_transforms(lq).to(dtype=weight_dtype)
    captions = inference(lq_ram, model)
    validation_prompt = f"{captions[0]}, {args.prompt},"
    
    return validation_prompt, lq

from model_config import get_config
def test_and_val(args, DAPE=None,iqa_metrics=None,model_test=None,added_cond_kwargs_list=None,fid_metrics=None):
    
    OSEDiff_test = get_config(args.model_name,phase='test')

    # initialize the model
    if model_test is None:
        model = OSEDiff_test(args)
    else:
        model = model_test
    # weight type
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    if DAPE is None and args.text_prompt:
        # get ram model
        DAPE = ram(pretrained=args.ram_path,
                pretrained_condition=args.ram_ft_path,
                image_size=384,
                vit='swin_l')
        DAPE.eval()
        DAPE.to("cuda")

        # set weight type
        DAPE = DAPE.to(dtype=weight_dtype)



    # get all input images
    if os.path.isdir(args.input_image):
        image_names = sorted(glob.glob(f'{args.input_image}/*.png'))
    else:
        image_names = [args.input_image]

    args.output_dir = os.path.join(args.output_dir, 'out')
    if args.save_prompts:
        txt_path = os.path.join(args.output_dir, 'txt')
        os.makedirs(txt_path, exist_ok=True)
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'There are {len(image_names)} images.')

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger = logging.getLogger('base')

    # Initialize IQA metrics excluding FID
    if iqa_metrics is None:
        logger.info("Initializing IQA metrics...")
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
        logger.info("IQA metrics initialized.\n")
    if fid_metrics is None:
        fid_metrics = pyiqa.create_metric('fid', device=device)

    # Validate input and GT directories
    if len([args.input_image]) != len(args.gt_imgs):
        logger.error("The number of input image directories and GT image directories must be the same.")
        sys.exit(1)

    dir_name = os.path.basename(os.path.normpath(args.gt_imgs[0]))
    logger.info(f"Testing Directory: [{dir_name}]")
        
      
    if args.val_4step:
        step_list = [249,499,749,999]
    else:
        step_list = [999]
    avg_metrics_str_list = []
    for n_step in step_list:
        img_idx = 0  
        metrics_accum = {metric: 0.0 for metric in iqa_metrics.keys()}
        model.timesteps = torch.tensor([n_step], device="cuda").long()   
        step_output_dir = os.path.join(args.output_dir,f'{dir_name}_{str(n_step)}')
        os.makedirs(step_output_dir, exist_ok=True)
        for image_name in tqdm(image_names):
            # make sure that the input image is a multiple of 8
            input_image = Image.open(image_name).convert('RGB')
            ori_width, ori_height = input_image.size
            rscale = args.upscale
            resize_flag = False
            if ori_width < args.process_size//rscale or ori_height < args.process_size//rscale:
                scale = (args.process_size//rscale)/min(ori_width, ori_height)
                input_image = input_image.resize((int(scale*ori_width), int(scale*ori_height)))
                resize_flag = True
            input_image = input_image.resize((input_image.size[0]*rscale, input_image.size[1]*rscale))

            new_width = input_image.width - input_image.width % 8
            new_height = input_image.height - input_image.height % 8
            input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
            bname = os.path.basename(image_name)

            # get caption
            if args.text_prompt:
                validation_prompt, lq = get_validation_prompt(args, input_image, DAPE,weight_dtype=weight_dtype)
                if args.save_prompts:
                    txt_save_path = f"{txt_path}/{bname.split('.')[0]}.txt"
                    with open(txt_save_path, 'w', encoding='utf-8') as f:
                        f.write(validation_prompt)
                        f.close()
            else:
                lq = tensor_transforms(input_image).unsqueeze(0).to(device)
                validation_prompt = None
            # print(f"process {image_name}, tag: {validation_prompt}".encode('utf-8'))

            # translate the image
            with torch.no_grad():
                lq = lq*2-1
                if 'sdxl' in args.model_name:
                    output_image = model(lq, prompt=validation_prompt,added_cond_kwargs=added_cond_kwargs_list)
                else:
                    output_image = model(lq, prompt=validation_prompt)
                output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
                if args.align_method == 'adain':
                    output_pil = adain_color_fix(target=output_pil, source=input_image)
                elif args.align_method == 'wavelet':
                    output_pil = wavelet_color_fix(target=output_pil, source=input_image)
                else:
                    pass
                if resize_flag:
                    output_pil.resize((int(args.upscale*ori_width), int(args.upscale*ori_height)))
                    
                #######
                
                gt_dir = args.gt_imgs[dir_idx]
                img_gt_list = sorted(glob.glob(os.path.join(gt_dir, '*.png')))
                gt_path = img_gt_list[img_idx]

                img_name = os.path.basename(gt_path)
                start_time = time.time()

                # Read and preprocess images
                gt_img = cv2.imread(gt_path, cv2.IMREAD_COLOR)

                if  gt_img is None:
                    logger.warning(f"Image read failed for {dir_idx}. Skipping.")
                    continue
                            
                sr_tensor = transforms.ToTensor()(output_pil).unsqueeze(0).to(device).contiguous()
                gt_tensor = img2tensor(gt_img, bgr2rgb=True, float32=True).unsqueeze(0).to(device).contiguous() / 255.0
                # if n_step==249:
                output_pil.save(os.path.join(step_output_dir, bname))
                # Compute metrics

                metrics = {}
                for name, metric in iqa_metrics.items():
                    if name in ['CLIPIQA', 'NIQE', 'MUSIQ', 'MANIQA']:
                        try:
                            metrics[name] = metric(sr_tensor).item()
                        except Exception as e:
                            logger.warning(f"Error computing {name} for {dir_name}/{img_name}. Skipping.")
                            metrics[name] = 0.0
                    else:
                        metrics[name] = metric(sr_tensor, gt_tensor).item()

                # Accumulate metrics
                for name in metrics_accum:
                    metrics_accum[name] += metrics[name]

                # Calculate runtime
                end_time = time.time()
                runtime = end_time - start_time

                # Log per-image metrics and runtime
                metrics_str = "; ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
                # logger.info(f"{dir_name}/{img_name} | {metrics_str} | Runtime: {runtime:.2f} sec")
                img_idx+=1
                
                
                
                
            # Compute average metrics
        fid_value = fid_metrics(gt_dir, step_output_dir).item()
        num_images = len(img_gt_list)
        avg_metrics = {k: round(v / num_images, 4) for k, v in metrics_accum.items()}
        avg_metrics['FID'] = round(fid_value,4)



        # Log average metrics for the directory
        avg_metrics_str = "; ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()])

        logger.info(f"\n===== Average Metrics for [{dir_name}] | step={str(n_step)}=====\n{avg_metrics_str}\n")
        if n_step==249 or len(step_list)==1:
            avg_metrics_str_list = avg_metrics
    print('save to '+args.output_dir)
    return avg_metrics_str_list




if __name__ == "__main__":
    from parse_args import parse_args
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    args = parse_args()
    logger = logging.getLogger('base')
    dir_name = os.path.basename(os.path.normpath(args.gt_imgs[0]))
    os.makedirs(args.output_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(args.output_dir, f'{dir_name}_metrics.txt'))
    logger.addHandler(file_handler)

    test_and_val(args)

    