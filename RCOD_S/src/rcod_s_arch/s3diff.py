import os
import re
import requests
import sys
import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import AutoTokenizer#, CLIPTextModel,CLIPTextModelWithProjection

from transformers import CLIPVisionModel,CLIPVisionModelWithProjection
from diffusers import AutoencoderKL
# from diffusers import UNet2DConditionModel
# from diffusers import EulerDiscreteScheduler as DDPMScheduler
from models.scheculer_DDPM import DDPMScheduler as DDPMScheduler
from models.unet_2d_condition import UNet2DConditionModel
from peft import LoraConfig, get_peft_model
import torchvision.transforms as T
p = "src/"
sys.path.append(p)
from model import make_1step_sched, my_lora_fwd
from basicsr.archs.arch_util import default_init_weights
from torch.nn import functional as F
def get_layer_number(module_name):
    base_layers = {
        'down_blocks': 0,
        'mid_block': 4,
        'up_blocks': 5
    }

    if module_name == 'conv_out':
        return 9

    base_layer = None
    for key in base_layers:
        if key in module_name:
            base_layer = base_layers[key]
            break

    if base_layer is None:
        return None

    additional_layers = int(re.findall(r'\.(\d+)', module_name)[0]) #sum(int(num) for num in re.findall(r'\d+', module_name))
    final_layer = base_layer + additional_layers
    return final_layer


class S3Diff(torch.nn.Module):
    def __init__(self, sd_path=None, pretrained_path=None, lora_rank_unet=32, lora_rank_vae=16, block_embedding_dim=64, args=None):
        super().__init__()
        self.args = args
        self.text_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").cuda()
        self.sched = DDPMScheduler.from_pretrained(sd_path, subfolder="scheduler")
        self.sched.set_timesteps(1, device="cuda")
        self.sched.alphas_cumprod = self.sched.alphas_cumprod.cuda()
        self.alpha_prod_t = torch.tensor([0.6754], device='cuda')
        self.beta_prod_t = 1 - self.alpha_prod_t

        vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet")

        target_modules_vae = r"^encoder\..*(conv1|conv2|conv_in|conv_shortcut|conv|conv_out|to_k|to_q|to_v|to_out\.0)$"
        target_modules_unet = [
            "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
            "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
        ]

        self.clip_feature_dim = self.text_encoder.config.hidden_size
        self.clip_projection = torch.nn.Linear(self.clip_feature_dim, unet.config.cross_attention_dim).cuda()

        kernel_size = (224, 224)
        stride = (224, 224)
        self.unfold = torch.nn.Unfold(kernel_size=kernel_size, stride=stride)
        self.resize = T.Resize((224, 224))
        
        

        num_embeddings = 64
        self.W = nn.Parameter(torch.randn(num_embeddings), requires_grad=False)

        self.vae_de_mlp = nn.Sequential(
            nn.Linear(num_embeddings * 4, 256),
            nn.ReLU(True),
        )

        self.unet_de_mlp = nn.Sequential(
            nn.Linear(num_embeddings * 4, 256),
            nn.ReLU(True),
        )

        self.vae_block_mlp = nn.Sequential(
            nn.Linear(block_embedding_dim, 64),
            nn.ReLU(True),
        )

        self.unet_block_mlp = nn.Sequential(
            nn.Linear(block_embedding_dim, 64),
            nn.ReLU(True),
        )

        self.vae_fuse_mlp = nn.Linear(256 + 64, lora_rank_vae ** 2)
        self.unet_fuse_mlp = nn.Linear(256 + 64, lora_rank_unet ** 2)

        default_init_weights([self.vae_de_mlp, self.unet_de_mlp, self.vae_block_mlp, self.unet_block_mlp, \
            self.vae_fuse_mlp, self.unet_fuse_mlp,self.clip_projection], 1e-5)

        # vae
        self.vae_block_embeddings = nn.Embedding(6, block_embedding_dim)
        self.unet_block_embeddings = nn.Embedding(10, block_embedding_dim)

        if pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu")
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)

            unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"])
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

            _vae_de_mlp = self.vae_de_mlp.state_dict()
            for k in sd["state_dict_vae_de_mlp"]:
                _vae_de_mlp[k] = sd["state_dict_vae_de_mlp"][k]
            self.vae_de_mlp.load_state_dict(_vae_de_mlp)

            _unet_de_mlp = self.unet_de_mlp.state_dict()
            for k in sd["state_dict_unet_de_mlp"]:
                _unet_de_mlp[k] = sd["state_dict_unet_de_mlp"][k]
            self.unet_de_mlp.load_state_dict(_unet_de_mlp)

            _vae_block_mlp = self.vae_block_mlp.state_dict()
            for k in sd["state_dict_vae_block_mlp"]:
                _vae_block_mlp[k] = sd["state_dict_vae_block_mlp"][k]
            self.vae_block_mlp.load_state_dict(_vae_block_mlp)

            _unet_block_mlp = self.unet_block_mlp.state_dict()
            for k in sd["state_dict_unet_block_mlp"]:
                _unet_block_mlp[k] = sd["state_dict_unet_block_mlp"][k]
            self.unet_block_mlp.load_state_dict(_unet_block_mlp)

            _vae_fuse_mlp = self.vae_fuse_mlp.state_dict()
            for k in sd["state_dict_vae_fuse_mlp"]:
                _vae_fuse_mlp[k] = sd["state_dict_vae_fuse_mlp"][k]
            self.vae_fuse_mlp.load_state_dict(_vae_fuse_mlp)

            _unet_fuse_mlp = self.unet_fuse_mlp.state_dict()
            for k in sd["state_dict_unet_fuse_mlp"]:
                _unet_fuse_mlp[k] = sd["state_dict_unet_fuse_mlp"][k]
            self.unet_fuse_mlp.load_state_dict(_unet_fuse_mlp)
            
            _clip_proj = self.clip_projection.state_dict()
            for k in sd["state_dict_clip_proj"]:
                _clip_proj[k] = sd["state_dict_clip_proj"][k]
            self.clip_projection.load_state_dict(_clip_proj)
            

            


            self.W = nn.Parameter(sd["w"], requires_grad=False)

            embeddings_state_dict = sd["state_embeddings"]
            self.vae_block_embeddings.load_state_dict(embeddings_state_dict['state_dict_vae_block'])
            self.unet_block_embeddings.load_state_dict(embeddings_state_dict['state_dict_unet_block'])
        else:
            print("Initializing model with random weights")
            vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian",
                target_modules=target_modules_vae)
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            unet_lora_config = LoraConfig(r=lora_rank_unet, init_lora_weights="gaussian",
                target_modules=target_modules_unet
            )
            unet.add_adapter(unet_lora_config)

        self.lora_rank_unet = lora_rank_unet
        self.lora_rank_vae = lora_rank_vae
        self.target_modules_vae = target_modules_vae
        self.target_modules_unet = target_modules_unet

        self.vae_lora_layers = []
        for name, module in vae.named_modules():
            if 'base_layer' in name:
                self.vae_lora_layers.append(name[:-len(".base_layer")])
                
        for name, module in vae.named_modules():
            if name in self.vae_lora_layers:
                module.forward = my_lora_fwd.__get__(module, module.__class__)

        self.unet_lora_layers = []
        for name, module in unet.named_modules():
            if 'base_layer' in name:
                self.unet_lora_layers.append(name[:-len(".base_layer")])

        for name, module in unet.named_modules():
            if name in self.unet_lora_layers:
                module.forward = my_lora_fwd.__get__(module, module.__class__)

        self.unet_layer_dict = {name: get_layer_number(name) for name in self.unet_lora_layers}

        unet.to("cuda")
        vae.to("cuda")
        self.unet, self.vae = unet, vae
        self.timesteps = torch.tensor([args.n_step], device="cuda").long()
        self.text_encoder.requires_grad_(False)


    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.vae_de_mlp.eval()
        self.unet_de_mlp.eval()
        self.vae_block_mlp.eval()
        self.unet_block_mlp.eval()
        self.vae_fuse_mlp.eval()
        self.unet_fuse_mlp.eval()
        self.clip_projection.eval()


        self.vae_block_embeddings.requires_grad_(False)
        self.unet_block_embeddings.requires_grad_(False)

        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        self.vae_de_mlp.train()
        self.unet_de_mlp.train()
        self.vae_block_mlp.train()
        self.unet_block_mlp.train()
        self.vae_fuse_mlp.train()
        self.unet_fuse_mlp.train()    
        self.clip_projection.train()


        self.vae_block_embeddings.requires_grad_(True)
        self.unet_block_embeddings.requires_grad_(True)

        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True

        self.unet.conv_in.requires_grad_(True)

        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
    def cosine_similarity(self, x1, x2, dim=1, eps=1e-8):
        x1_ = x1.reshape(x1.shape[0], -1)
        x2_ = x2.reshape(x2.shape[0], -1)
        output = F.cosine_similarity(x1_, x2_, dim=1)
       
        return output
    def encode_prompt(self, prompt_batch):
        
        prompt_batch = self.resize(prompt_batch)

        with torch.no_grad():

            prompt_embeds = self.text_encoder(
                prompt_batch.cuda(),
            ).last_hidden_state
        return prompt_embeds

    def get_prompt_encoder(self, prompt, batch_size=1):
        with torch.no_grad():
            prompt_embeds_pre = self.encode_prompt(prompt)
        prompt_embeds = self.clip_projection(prompt_embeds_pre)
        return prompt_embeds

    def div_step(self, cs_gt_bc):
        step_list = []
        if self.args.n_div == 3:
            for cs_gt in cs_gt_bc:
                if cs_gt < 0.5:
                    cs_time_step = 749
                elif cs_gt < 0.7 and cs_gt >= 0.5:
                    cs_time_step = 499
                elif cs_gt >= 0.7:
                    cs_time_step = 249
                else:
                    cs_time_step = 499
                step_list.append(cs_time_step)
        elif self.args.n_div == 4:
            for cs_gt in cs_gt_bc:
                if cs_gt < 0.3:
                    cs_time_step = 999
                elif cs_gt < 0.5 and cs_gt >= 0.3:
                    cs_time_step = 749
                elif cs_gt < 0.7 and cs_gt >= 0.5:
                    cs_time_step = 499
                elif cs_gt >= 0.7:
                    cs_time_step = 249
                else:
                    cs_time_step = 499
                step_list.append(cs_time_step)

        return step_list
    def forward(self, c_t, deg_score,c_t_gt=None):
 

            
        caption_enc = self.get_prompt_encoder(c_t,batch_size=c_t.shape[0])
            


        # degradation fourier embedding
        deg_proj = deg_score[..., None] * self.W[None, None, :] * 2 * np.pi
        deg_proj = torch.cat([torch.sin(deg_proj), torch.cos(deg_proj)], dim=-1)
        deg_proj = torch.cat([deg_proj[:, 0], deg_proj[:, 1]], dim=-1)

        # degradation mlp forward
        vae_de_c_embed = self.vae_de_mlp(deg_proj)
        unet_de_c_embed = self.unet_de_mlp(deg_proj)

        # block embedding mlp forward
        vae_block_c_embeds = self.vae_block_mlp(self.vae_block_embeddings.weight)
        unet_block_c_embeds = self.unet_block_mlp(self.unet_block_embeddings.weight)
        vae_embeds = self.vae_fuse_mlp(torch.cat([vae_de_c_embed.unsqueeze(1).repeat(1, vae_block_c_embeds.shape[0], 1), \
            vae_block_c_embeds.unsqueeze(0).repeat(vae_de_c_embed.shape[0],1,1)], -1))
        unet_embeds = self.unet_fuse_mlp(torch.cat([unet_de_c_embed.unsqueeze(1).repeat(1, unet_block_c_embeds.shape[0], 1), \
            unet_block_c_embeds.unsqueeze(0).repeat(unet_de_c_embed.shape[0],1,1)], -1))

        for layer_name, module in self.vae.named_modules():
            if layer_name in self.vae_lora_layers:
                split_name = layer_name.split(".")
                if split_name[1] == 'down_blocks':
                    block_id = int(split_name[2])
                    vae_embed = vae_embeds[:, block_id]
                elif split_name[1] == 'mid_block':
                    vae_embed = vae_embeds[:, -2]
                else:
                    vae_embed = vae_embeds[:, -1]
                module.de_mod = vae_embed.reshape(-1, self.lora_rank_vae, self.lora_rank_vae)

        for layer_name, module in self.unet.named_modules():
            if layer_name in self.unet_lora_layers:
                split_name = layer_name.split(".")

                if split_name[0] == 'down_blocks':
                    block_id = int(split_name[1])
                    unet_embed = unet_embeds[:, block_id]
                elif split_name[0] == 'mid_block':
                    unet_embed = unet_embeds[:, 4]
                elif split_name[0] == 'up_blocks':
                    block_id = int(split_name[1]) + 5
                    unet_embed = unet_embeds[:, block_id]
                else:
                    unet_embed = unet_embeds[:, -1]
                module.de_mod = unet_embed.reshape(-1, self.lora_rank_unet, self.lora_rank_unet)

        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
        
        if c_t_gt is not None:
            encoded_control_gt = self.vae.encode(c_t_gt).latent_dist.sample() * self.vae.config.scaling_factor

            cs_gt_bc = self.cosine_similarity(encoded_control, encoded_control_gt.detach())
            # lsdir[0,0.63, 0.7, 0.76, 1.0]
            # lsdir+ffhq [0,0.65,0.72, 0.78, 1.0]
            step_list = self.div_step(cs_gt_bc)

            self.timesteps = torch.tensor(step_list, device="cuda").long()
        else:
            self.timesteps = torch.tensor([999], device="cuda").long()
            
        
        fix_step = torch.tensor([249], device="cuda").long()

        model_pred = self.unet(encoded_control, self.timesteps,
                               encoder_hidden_states=caption_enc).sample
        self.sched.timesteps = fix_step

        x_denoised = (encoded_control-self.beta_prod_t**(0.5)*model_pred)/self.alpha_prod_t**(0.5) 
        
 
        output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        return output_image

    def save_model(self, outf):
        sd = {}
        sd["unet_lora_target_modules"] = self.target_modules_unet
        sd["vae_lora_target_modules"] = self.target_modules_vae
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip_conv" in k}
        sd["state_dict_vae_de_mlp"] = {k: v for k, v in self.vae_de_mlp.state_dict().items()}
        sd["state_dict_unet_de_mlp"] = {k: v for k, v in self.unet_de_mlp.state_dict().items()}
        sd["state_dict_vae_block_mlp"] = {k: v for k, v in self.vae_block_mlp.state_dict().items()}
        sd["state_dict_unet_block_mlp"] = {k: v for k, v in self.unet_block_mlp.state_dict().items()}
        sd["state_dict_vae_fuse_mlp"] = {k: v for k, v in self.vae_fuse_mlp.state_dict().items()}
        sd["state_dict_unet_fuse_mlp"] = {k: v for k, v in self.unet_fuse_mlp.state_dict().items()}
        sd["state_dict_clip_proj"] = {k: v for k, v in self.clip_projection.state_dict().items()}

        sd["w"] = self.W

        sd["state_embeddings"] = {
                    "state_dict_vae_block": self.vae_block_embeddings.state_dict(),
                    "state_dict_unet_block": self.unet_block_embeddings.state_dict(),
                }

        torch.save(sd, outf)


# Backward compatibility and standard aliases
RCOD_S_arch = S3Diff
