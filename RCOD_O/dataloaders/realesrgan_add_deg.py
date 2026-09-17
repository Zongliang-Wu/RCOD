import os
import numpy as np
import cv2
import glob
import math
import yaml
import random
from collections import OrderedDict
import torch
import torch.nn.functional as F

from basicsr.data.transforms import augment
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.utils import DiffJPEG, USMSharp, img2tensor, tensor2img
from basicsr.utils.img_process_util import filter2D
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from torchvision.transforms.functional import (adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation,
                                               normalize, rgb_to_grayscale)
from dataloaders.realesrgan import RealESRGAN_degradation
cur_path = os.path.dirname(os.path.abspath(__file__))

def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

def opt_parse(opt_path):
    with open(opt_path, mode='r') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)  # ignore_security_alert_wait_for_fix RCE

    return opt
class RealESRGAN_degradation_with_degpipe(RealESRGAN_degradation):
    def __init__(self, opt_name='params_realesrgan.yml', device='cuda'):
        super().__init__(opt_name, device)
        self.jpeger = DiffJPEG(differentiable=False).to(self.device)

    def deg_pipe_1_cuda(self, source):
        """
        GPU-accelerated degradation: Gaussian blur -> random resize -> JPEG compression
        """
        bs, ch, h_, w_ = source.size()
        kernel_size = random.randint(0, 10) * 2 + 1
        sigma = random.choice(np.arange(1, 15, step=0.1))
        kernel = self.create_gaussian_kernel(kernel_size, sigma)
        kernel = kernel.repeat(3, 1, 1, 1)
        blurred = F.conv2d(source, kernel, padding=kernel_size // 2, groups=ch)

        resize_mode = random.choice(['nearest', 'bilinear', 'bicubic'])
        scale_factor = random.uniform(0.5, 1.5)
        resized = F.interpolate(blurred, scale_factor=scale_factor, mode=resize_mode)

        quality_parameter = random.randint(50, 99)

        if np.random.uniform() < 0.5:
            resized = F.interpolate(resized, (h_, w_), mode=resize_mode)
            jpeg_compressed = self.jpeger(resized, quality=quality_parameter)
            return jpeg_compressed
        else:
            jpeg_compressed = self.jpeger(resized, quality=quality_parameter)
            resized = F.interpolate(jpeg_compressed, (h_, w_), mode=resize_mode)
            return resized

    def create_gaussian_kernel(self, kernel_size, sigma):
        """
        Generate Gaussian blur kernel.
        """
        k = torch.tensor(np.fromfunction(
            lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * 
                        np.exp(- ((x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2) / (2 * sigma ** 2)),
            (kernel_size, kernel_size)
        ), dtype=torch.float32)

        k = k / k.sum()
        return k.unsqueeze(0).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def degrade_process(self, img_gt, resize_bak=False):
        """
        Apply RealESRGAN degradation followed by CUDA degradation pipeline.
        """
        img_gt, img_lq = super().degrade_process(img_gt, resize_bak)
        img_lq2 = self.deg_pipe_1_cuda(img_lq)
        return img_gt, img_lq2