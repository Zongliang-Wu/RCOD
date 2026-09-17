
import os

if os.environ.get("USE_HF_MIRROR", "false").lower() in ("true", "1"):
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import sys
sys.path.append(os.getcwd())
import yaml
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModel
from models.scheculer_DDPM import DDPMScheduler
from models.autoencoder_kl import AutoencoderKL
from models.unet_2d_condition import UNet2DConditionModel
from peft import LoraConfig
from transformers import CLIPVisionModel
import timm
from my_utils.vaehook import VAEHook, perfcount
import torchvision.transforms as T
def initialize_vae(args):
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    vae.requires_grad_(False)
    vae.train()
    
    l_target_modules_encoder = []
    l_grep = ["conv1","conv2","conv_in", "conv_shortcut", "conv", "conv_out", "to_k", "to_q", "to_v", "to_out.0"]
    for n, p in vae.named_parameters():
        if "bias" in n or "norm" in n: 
            continue
        for pattern in l_grep:
            if pattern in n and ("encoder" in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
            elif ('quant_conv' in n) and ('post_quant_conv' not in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
    
    lora_conf_encoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder)
    vae.add_adapter(lora_conf_encoder, adapter_name="default_encoder")

    return vae, l_target_modules_encoder


def initialize_unet(args, return_lora_module_names=False, pretrained_model_name_or_path=None):
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet",)
    unet.requires_grad_(False)
    unet.train()

    l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
                break
            elif pattern in n and ("up_blocks" in n or "conv_out" in n):
                l_target_modules_decoder.append(n.replace(".weight",""))
                break
            elif pattern in n:
                l_modules_others.append(n.replace(".weight",""))
                break

    lora_conf_encoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder)
    lora_conf_decoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",target_modules=l_target_modules_decoder)
    lora_conf_others = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian",target_modules=l_modules_others)
    unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    unet.add_adapter(lora_conf_others, adapter_name="default_others")

    return unet, l_target_modules_encoder, l_target_modules_decoder, l_modules_others


class OSEDiff_gen(torch.nn.Module):
    def __init__(self, args=None):
        super().__init__()

        self.text_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").cuda()
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(1, device="cuda")
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.cuda()
        self.args = args

        self.vae, self.lora_vae_modules_encoder = self._init_tiled_vae(self.args)
        self.unet, self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others = self.initialize_unet(self.args)

        self.unet.to("cuda")
        self.vae.to("cuda")
        self.timesteps = torch.tensor([249], device="cuda").long()
        self.text_encoder.requires_grad_(False)
        
        self.alpha_prod_t = torch.tensor([0.7537], device='cuda')
        self.beta_prod_t = 1 - self.alpha_prod_t
        
        self.clip_feature_dim = self.text_encoder.config.hidden_size
        self.clip_projection = torch.nn.Linear(self.clip_feature_dim, self.unet.config.cross_attention_dim).cuda()

        kernel_size = (224, 224)
        stride = (224, 224)
        self.unfold = torch.nn.Unfold(kernel_size=kernel_size, stride=stride)
        self.resize = T.Resize((224, 224))

    def preprocess_image(self, image):
        """
        Preprocess image for VPIM visual embedding extraction.
        Args:
            image: (bs, c, w, h) image tensor on CUDA, w, h >= 224
        Returns:
            (bs, n*c, 224, 224) tensor on CUDA
        """
        bs, c, w, h = image.shape
        resized_image = self.resize(image)
        unfolded_images = self.unfold(image)
        n = unfolded_images.shape[2] + 1
        unfolded_images = unfolded_images.view(bs, c, 224, 224, n - 1)
        unfolded_images = unfolded_images.permute(0, 4, 1, 2, 3)

        output = torch.cat([resized_image.unsqueeze(1), unfolded_images], dim=1)
        output = output.view(bs, -1, 224, 224)
        return output
    def _init_tiled_vae(self,args,
            encoder_tile_size = 256,
            decoder_tile_size = 256,
            fast_decoder = False,
            fast_encoder = False,
            color_fix = False,
            vae_to_gpu = True):
        self.lora_rank_vae = self.args.lora_rank
        return initialize_vae(args)
    def initialize_unet(self,args):
        self.lora_rank_unet = self.args.lora_rank

        return initialize_unet(args)
    def encode_prompt(self, prompt_batch):
        
        prompt_batch = self.resize(prompt_batch)

        with torch.no_grad():

            prompt_embeds = self.text_encoder(
                prompt_batch.cuda(),
            ).last_hidden_state
               

        return prompt_embeds
    
    def forward(self, c_t,batch=None, args=None):

        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor # src
        c_t_gt = batch["output_pixel_values"]
        encoded_control_gt = self.vae.encode(c_t_gt).latent_dist.sample() * self.vae.config.scaling_factor

        cs_gt_bc = self.cosine_similarity(encoded_control, encoded_control_gt.detach())
        # lsdir[0,0.63, 0.7, 0.76, 1.0]
        # lsdir+ffhq [0,0.65,0.72, 0.78, 1.0]
        step_list = self.div_step(cs_gt_bc)
        
        self.timesteps = torch.tensor(step_list, device="cuda").long()
        # calculate prompt_embeddings and neg_prompt_embeddings
        prompt_embeds_pre = self.encode_prompt(c_t)
        prompt_embeds = self.clip_projection(prompt_embeds_pre)
        
        prompt_embeds_neg_pre = self.encode_prompt(torch.zeros_like(c_t, device="cuda", dtype=c_t.dtype))
        prompt_embeds_neg = self.clip_projection(prompt_embeds_neg_pre)
        
        fix_step = torch.tensor([249], device="cuda").long()

        model_pred = self.unet(encoded_control, self.timesteps, encoder_hidden_states=prompt_embeds.to(torch.float32)).sample
        x_denoised = (encoded_control - self.beta_prod_t**(0.5) * model_pred) / self.alpha_prod_t**(0.5)

        output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        return output_image, x_denoised
    def cosine_similarity(self, x1, x2, dim=1, eps=1e-8):
        x1_ = x1.reshape(x1.shape[0], -1)
        x2_ = x2.reshape(x2.shape[0], -1)
        output = F.cosine_similarity(x1_, x2_, dim=1)
       
        return output
    def div_step(self, cs_gt_bc):
        step_list = []
        if self.args.n_div == 3:
            for cs_gt in cs_gt_bc:
                if cs_gt<0.5:
                    cs_time_step = 749
                elif cs_gt<0.7 and cs_gt>=0.5:
                    cs_time_step = 499
                elif cs_gt>=0.7:
                    cs_time_step=249
                step_list.append(cs_time_step)   
        elif self.args.n_div == 4:
            for cs_gt in cs_gt_bc:
                if cs_gt<0.3:
                    cs_time_step = 999
                elif cs_gt<0.5 and cs_gt>=0.3:
                    cs_time_step = 749
                elif cs_gt<0.7 and cs_gt>=0.5:
                    cs_time_step = 499
                elif cs_gt>=0.7:
                    cs_time_step=249
                step_list.append(cs_time_step)
        
        return step_list

    def save_model(self, outf):
        sd = {}
        sd["vae_lora_encoder_modules"] = self.lora_vae_modules_encoder
        sd["unet_lora_encoder_modules"], sd["unet_lora_decoder_modules"], sd["unet_lora_others_modules"] =\
            self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k}
        sd["state_dict_proj"] = {k: v for k, v in self.clip_projection.state_dict().items()}
        torch.save(sd, outf)

    def load_ckpt(self, model):

        for n, p in self.unet.named_parameters():
            if "lora" in n or "conv_in" in n:
                p.data.copy_(model["state_dict_unet"][n])

        for n, p in self.vae.named_parameters():
            if "lora" in n:
                p.data.copy_(model["state_dict_vae"][n])

        for n, p in self.clip_projection.named_parameters():
            p.data.copy_(model["state_dict_proj"][n])
    def set_train(self):
        self.unet.train()
        self.vae.train()
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)
        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
    def set_val(self):
        self.unet.eval()
        self.vae.eval()


class OSEDiff_gen_vis(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").cuda()
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(1, device="cuda")
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.cuda()
        self.vae = AutoencoderKL.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="unet")
        

        # self.vae, self.lora_vae_modules_encoder = initialize_vae(self.args)
        # self.unet, self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others = initialize_unet(self.args)
        # self.lora_rank_unet = self.args.lora_rank
        # self.lora_rank_vae = self.args.lora_rank
        
        
        osediff = torch.load(args.osediff_path)
        self.load_ckpt(osediff)
        
        # merge lora
        # if self.args.merge_and_unload_lora:
        #     print(f'===> MERGE LORA <===')
        #     self.vae = self.vae.merge_and_unload()
        #     self.unet = self.unet.merge_and_unload()

        self.unet.to("cuda")
        self.vae.to("cuda")
        self.timesteps = torch.tensor([999], device="cuda").long()
        self.text_encoder.requires_grad_(False)
    def load_ckpt(self, model):
        # load unet lora
        lora_conf_encoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_encoder_modules"])
        lora_conf_decoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_decoder_modules"])
        lora_conf_others = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_others_modules"])
        self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        self.unet.add_adapter(lora_conf_others, adapter_name="default_others")
        for n, p in self.unet.named_parameters():
            if "lora" in n or "conv_in" in n:
                p.data.copy_(model["state_dict_unet"][n])
        self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])

        # load vae lora
        vae_lora_conf_encoder = LoraConfig(r=model["rank_vae"], init_lora_weights="gaussian", target_modules=model["vae_lora_encoder_modules"])
        self.vae.add_adapter(vae_lora_conf_encoder, adapter_name="default_encoder")
        for n, p in self.vae.named_parameters():
            if "lora" in n:
                p.data.copy_(model["state_dict_vae"][n])
        self.vae.set_adapter(['default_encoder'])
    


    def set_train(self):
        self.unet.train()
        self.vae.train()
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)
        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
    def set_val(self):
        self.unet.eval()
        self.vae.eval()

    def encode_prompt(self, prompt_batch):
        prompt_batch = self.resize(prompt_batch)
        with torch.no_grad():

            prompt_embeds = self.text_encoder(
                prompt_batch.to(self.text_encoder.device),
            ).pooler_output 
               

        return prompt_embeds
 
    def cosine_similarity(self, x1, x2, dim=1, eps=1e-8):
        x1_ = x1.reshape(x1.shape[0], -1)
        x2_ = x2.reshape(x2.shape[0], -1)
        output = F.cosine_similarity(x1_, x2_, dim=1)
       
        return output

    def forward(self, c_t,batch=None, args=None):

        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor # src
        c_t_gt = batch["output_pixel_values"]
        encoded_control_gt = self.vae.encode(c_t_gt).latent_dist.sample() * self.vae.config.scaling_factor

        cs_gt_bc = self.cosine_similarity(encoded_control, encoded_control_gt.detach())
        gt_mean = encoded_control_gt.mean()
        gt_std = encoded_control_gt.std()
        lr_mean = encoded_control.mean()
        lr_std = encoded_control.std()
        mean_std_lr = [lr_mean, lr_std]
        mean_std_gt = [gt_mean, gt_std]
        step_list = []
        for cs_gt in cs_gt_bc:
            if cs_gt<0.5:
                cs_time_step = 749
            elif cs_gt<0.7 and cs_gt>=0.5:
                cs_time_step = 499
            elif cs_gt>=0.7:
                cs_time_step=249
            step_list.append(cs_time_step)   
            
        

        return cs_gt_bc[0], cs_time_step,mean_std_gt,mean_std_lr

    def save_model(self, outf):
        sd = {}
        sd["vae_lora_encoder_modules"] = self.lora_vae_modules_encoder
        sd["unet_lora_encoder_modules"], sd["unet_lora_decoder_modules"], sd["unet_lora_others_modules"] =\
            self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k}
        sd["state_dict_proj"] = {k: v for k, v in self.clip_projection.state_dict().items()}
        torch.save(sd, outf)



class OSEDiff_reg(torch.nn.Module):
    def __init__(self, args, accelerator):
        super().__init__() 

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder").cuda()
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.args = args

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype

        self.vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
        self.unet_fix = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
        self.unet_update, self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others =\
                initialize_unet(args)

        self.text_encoder.to(accelerator.device, dtype=weight_dtype)
        self.unet_fix.to(accelerator.device, dtype=weight_dtype)
        self.unet_update.to(accelerator.device)
        self.vae.to(accelerator.device)
        
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet_fix.requires_grad_(False)

    def set_train(self):
        self.unet_update.train()
        for n, _p in self.unet_update.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
    def encode_prompt(self, prompt_batch):
        prompt_embeds_list = []
        with torch.no_grad():
            for caption in prompt_batch:
                text_input_ids = self.tokenizer(
                    caption, max_length=self.tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(self.text_encoder.device),
                )[0]
                prompt_embeds_list.append(prompt_embeds)
        prompt_embeds = torch.concat(prompt_embeds_list, dim=0)
        return prompt_embeds
    def diff_loss(self, latents, prompt_embeds, args):

        latents, prompt_embeds = latents.detach(), prompt_embeds.detach()
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        noise_pred = self.unet_update(
        noisy_latents,
        timestep=timesteps,
        encoder_hidden_states=prompt_embeds,
        ).sample

        loss_d = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        
        return loss_d

    def eps_to_mu(self, scheduler, model_output, sample, timesteps):
        
        alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        alpha_prod_t = alphas_cumprod[timesteps]
        while len(alpha_prod_t.shape) < len(sample.shape):
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        return pred_original_sample

    def time_step_group(self, model_gen_timesteps: torch.Tensor,device):
        # 249 499 749 999
        timestep_list = []
        vsd_cfg_list = []
        


        for model_gen_timestep in model_gen_timesteps:
            step = int(model_gen_timestep.item())
            if step < 749:
                timestep_start = 20
                timestep_end = 980
            else:
                timestep_start = 200
                timestep_end = 980
            timestep_i = torch.randint(timestep_start, timestep_end, (1,), device=device).long()
            timestep_list.append(timestep_i)

            if step in (249, 499, 749, 999):
                vsd_cfg_list.append(torch.tensor([7.5], device=device))
            else:
                vsd_cfg_list.append(torch.tensor([7.5], device=device))

        timesteps = torch.cat(timestep_list, dim=0)
        vsd_cfgs = torch.cat(vsd_cfg_list, dim=0)
        
        return timesteps,vsd_cfgs

    def distribution_matching_loss(self, latents, prompt_embeds, neg_prompt_embeds, args,model_gen_timesteps):
        bsz = latents.shape[0]
        timesteps,vsd_cfgs = self.time_step_group(model_gen_timesteps, device=latents.device)
        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        with torch.no_grad():

            noise_pred_update = self.unet_update(
                noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds.float(),
                ).sample

            x0_pred_update = self.eps_to_mu(self.noise_scheduler, noise_pred_update, noisy_latents, timesteps)

            noisy_latents_input = torch.cat([noisy_latents] * 2)
            timesteps_input = torch.cat([timesteps] * 2)
            prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)

            noise_pred_fix = self.unet_fix(
                noisy_latents_input.to(dtype=self.weight_dtype),
                timestep=timesteps_input,
                encoder_hidden_states=prompt_embeds.to(dtype=self.weight_dtype),
                ).sample

            noise_pred_uncond, noise_pred_text = noise_pred_fix.chunk(2)
            # cfg up, text semantic info up
            noise_pred_fix = noise_pred_uncond + vsd_cfgs * (noise_pred_text - noise_pred_uncond)
            noise_pred_fix.to(dtype=torch.float32)

            x0_pred_fix = self.eps_to_mu(self.noise_scheduler, noise_pred_fix, noisy_latents, timesteps)

        weighting_factor = torch.abs(latents - x0_pred_fix).mean(dim=[1, 2, 3], keepdim=True)

        grad = (x0_pred_update - x0_pred_fix) / weighting_factor
        loss = F.mse_loss(latents, (latents - grad).detach())

        return loss

class OSEDiff_test(OSEDiff_gen):
    def __init__(self, args):
        super().__init__(args)

        self.args = args

        # vae tile
        # self._init_tiled_vae(encoder_tile_size=args.vae_encoder_tiled_size, decoder_tile_size=args.vae_decoder_tiled_size)

        self.weight_dtype = torch.float32
        if args.mixed_precision == "fp16":
            self.weight_dtype = torch.float16

        osediff = torch.load(args.osediff_path)
        self.load_ckpt(osediff)

        # merge lora
        if self.args.merge_and_unload_lora:
            print(f'===> MERGE LORA <===')
            self.vae = self.vae.merge_and_unload()
            self.unet = self.unet.merge_and_unload()

        self.unet.to("cuda", dtype=self.weight_dtype)
        self.vae.to("cuda", dtype=self.weight_dtype)
        self.text_encoder.to("cuda", dtype=self.weight_dtype)
        self.clip_projection.to("cuda", dtype=self.weight_dtype)
        # 999 749 499 249
        self.timesteps = torch.tensor([args.n_step], device="cuda").long() 



        

    def load_ckpt(self, model):
        # load unet lora
        lora_conf_encoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_encoder_modules"])
        lora_conf_decoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_decoder_modules"])
        lora_conf_others = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_others_modules"])
        self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        self.unet.add_adapter(lora_conf_others, adapter_name="default_others")
        for n, p in self.unet.named_parameters():
            if "lora" in n or "conv_in" in n:
                p.data.copy_(model["state_dict_unet"][n])
        self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])

        # load vae lora
        if self.args.vae_state=='lora':
            vae_lora_conf_encoder = LoraConfig(r=model["rank_vae"], init_lora_weights="gaussian", target_modules=model["vae_lora_encoder_modules"])
            self.vae.add_adapter(vae_lora_conf_encoder, adapter_name="default_encoder")
            for n, p in self.vae.named_parameters():
                if "lora" in n:
                    p.data.copy_(model["state_dict_vae"][n])
            self.vae.set_adapter(['default_encoder'])
        elif self.args.vae_state=='full_ft':
            for n, p in self.vae.named_parameters():
                if 'encoder' in n:
                    p.data.copy_(model["state_dict_vae"][n])
                elif ('quant_conv' in n) and ('post_quant_conv' not in n):
                    p.data.copy_(model["state_dict_vae"][n])
        elif self.args.vae_state=='fix':
            pass
        

        for n, p in self.clip_projection.named_parameters():
            p.data.copy_(model["state_dict_proj"][n])

            
    # @perfcount
    @torch.no_grad()
    def forward(self, lq, prompt):

        prompt_embeds_pre = self.encode_prompt(lq)
        prompt_embeds = self.clip_projection(prompt_embeds_pre)
        lq_latent = self.vae.encode(lq.to(self.weight_dtype)).latent_dist.sample() * self.vae.config.scaling_factor
        ## add tile function
        _, _, h, w = lq_latent.size()
        tile_size, tile_overlap = (self.args.latent_tiled_size, self.args.latent_tiled_overlap)
        if h * w <= tile_size * tile_size:
            # print(f"[Tiled Latent]: the input size is tiny and unnecessary to tile.")
            model_pred = self.unet(lq_latent, self.timesteps, encoder_hidden_states=prompt_embeds).sample
        else:
            print(f"[Tiled Latent]: the input size is {lq.shape[-2]}x{lq.shape[-1]}, need to tiled")
            tile_weights = self._gaussian_weights(tile_size, tile_size, 1)
            tile_size = min(tile_size, min(h, w))
            tile_weights = self._gaussian_weights(tile_size, tile_size, 1)

            grid_rows = 0
            cur_x = 0
            while cur_x < lq_latent.size(-1):
                cur_x = max(grid_rows * tile_size-tile_overlap * grid_rows, 0)+tile_size
                grid_rows += 1

            grid_cols = 0
            cur_y = 0
            while cur_y < lq_latent.size(-2):
                cur_y = max(grid_cols * tile_size-tile_overlap * grid_cols, 0)+tile_size
                grid_cols += 1

            input_list = []
            noise_preds = []
            for row in range(grid_rows):
                noise_preds_row = []
                for col in range(grid_cols):
                    if col < grid_cols-1 or row < grid_rows-1:
                        # extract tile from input image
                        ofs_x = max(row * tile_size-tile_overlap * row, 0)
                        ofs_y = max(col * tile_size-tile_overlap * col, 0)
                        # input tile area on total image
                    if row == grid_rows-1:
                        ofs_x = w - tile_size
                    if col == grid_cols-1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    # input tile dimensions
                    input_tile = lq_latent[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                    input_list.append(input_tile)

                    if len(input_list) == 1 or col == grid_cols-1:
                        input_list_t = torch.cat(input_list, dim=0)
                        # predict the noise residual
                        model_out = self.unet(input_list_t, self.timesteps, encoder_hidden_states=prompt_embeds.to(self.weight_dtype),).sample
                        input_list = []
                    noise_preds.append(model_out)

            # Stitch noise predictions for all tiles
            noise_pred = torch.zeros(lq_latent.shape, device=lq_latent.device)
            contributors = torch.zeros(lq_latent.shape, device=lq_latent.device)
            # Add each tile contribution to overall latents
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if col < grid_cols-1 or row < grid_rows-1:
                        # extract tile from input image
                        ofs_x = max(row * tile_size-tile_overlap * row, 0)
                        ofs_y = max(col * tile_size-tile_overlap * col, 0)
                        # input tile area on total image
                    if row == grid_rows-1:
                        ofs_x = w - tile_size
                    if col == grid_cols-1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += noise_preds[row*grid_cols + col] * tile_weights
                    contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
            # Average overlapping areas with more than 1 contributor
            noise_pred /= contributors
            model_pred = noise_pred

        # x_denoised = self.noise_scheduler.step(model_pred, self.timesteps, lq_latent, return_dict=True).prev_sample
        x_denoised = (lq_latent-self.beta_prod_t**(0.5)*model_pred)/self.alpha_prod_t**(0.5) 
        output_image = (self.vae.decode(x_denoised.to(self.weight_dtype) / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        return output_image

    def _init_tiled_vae(self,args,
            encoder_tile_size = 1024,
            decoder_tile_size = 224,
            fast_decoder = False,
            fast_encoder = False,
            color_fix = False,
            vae_to_gpu = True):
        # save original forward (only once)
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
        vae.requires_grad_(False)
    
        if not hasattr(vae.encoder, 'original_forward'):
            setattr(vae.encoder, 'original_forward', vae.encoder.forward)
        if not hasattr(vae.decoder, 'original_forward'):
            setattr(vae.decoder, 'original_forward', vae.decoder.forward)

        encoder = vae.encoder
        decoder = vae.decoder

        vae.encoder.forward = VAEHook(
            encoder, encoder_tile_size, is_decoder=False, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)
        vae.decoder.forward = VAEHook(
            decoder, decoder_tile_size, is_decoder=True, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)
        return vae,[]
    
    def initialize_unet(self,args):
        unet = UNet2DConditionModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="unet")
        return unet, [], [], []
    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        """Generates a gaussian mask of weights for tile contributions"""
        from numpy import pi, exp, sqrt
        import numpy as np

        latent_width = tile_width
        latent_height = tile_height

        var = 0.01
        midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width)/(2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
        midpoint = latent_height / 2
        y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height)/(2*var)) / sqrt(2*pi*var) for y in range(latent_height)]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(torch.tensor(weights, device=self.device), (nbatches, self.unet.config.in_channels, 1, 1))


class OSEDiff_inference_time(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").cuda()
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(1, device="cuda")
        self.vae = AutoencoderKL.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="unet")

        self.weight_dtype = torch.float32
        if args.mixed_precision == "fp16":
            self.weight_dtype = torch.float16

        osediff = torch.load(args.osediff_path)
        self.load_ckpt(osediff)

        # merge lora
        if self.args.merge_and_unload_lora:
            print(f'===> MERGE LORA <===')
            self.vae = self.vae.merge_and_unload()
            self.unet = self.unet.merge_and_unload()

        self.unet.to("cuda", dtype=self.weight_dtype)
        self.vae.to("cuda", dtype=self.weight_dtype)
        self.text_encoder.to("cuda", dtype=self.weight_dtype)

            
        self.alpha_prod_t = torch.tensor([0.7537], device='cuda')
        self.beta_prod_t = 1 - self.alpha_prod_t
        
        # CLIP feature dimension
        self.clip_feature_dim = self.text_encoder.config.hidden_size  # projection_dim

        # Add a linear layer to project CLIP embeddings to UNet's input dimension
        self.clip_projection = torch.nn.Linear(self.clip_feature_dim, self.unet.config.cross_attention_dim)  # Example: Project to a suitable dimension
        self.clip_projection.to("cuda", dtype=self.weight_dtype)
        # print(f"CLIP projection layer: {self.clip_feature_dim}")
        # print(f"UNet input dimension: {self.unet.config.cross_attention_dim}")

        # self.clip_projection = torch.nn.Identity().cuda()
        # Unfold
        kernel_size = (224, 224)
        stride = (224, 224)

        self.unfold = torch.nn.Unfold(kernel_size=kernel_size, stride=stride)
        self.resize = T.Resize((224, 224))
        
        self.timesteps = torch.tensor([249], device="cuda").long() 
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.cuda()

        

    def load_ckpt(self, model):
        # load unet lora
        lora_conf_encoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_encoder_modules"])
        lora_conf_decoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_decoder_modules"])
        lora_conf_others = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_others_modules"])
        self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        self.unet.add_adapter(lora_conf_others, adapter_name="default_others")
        for n, p in self.unet.named_parameters():
            if "lora" in n or "conv_in" in n:
                p.data.copy_(model["state_dict_unet"][n])
        self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])

        # load vae lora
        vae_lora_conf_encoder = LoraConfig(r=model["rank_vae"], init_lora_weights="gaussian", target_modules=model["vae_lora_encoder_modules"])
        self.vae.add_adapter(vae_lora_conf_encoder, adapter_name="default_encoder")
        for n, p in self.vae.named_parameters():
            if "lora" in n:
                p.data.copy_(model["state_dict_vae"][n])
        self.vae.set_adapter(['default_encoder'])

    def encode_prompt(self, prompt_batch):
        
        prompt_batch = self.resize(prompt_batch)

        with torch.no_grad():

            prompt_embeds = self.text_encoder(
                prompt_batch.to(self.text_encoder.device),
            ).last_hidden_state
               

        return prompt_embeds

    @torch.no_grad()
    def forward(self, lq):

        prompt_embeds_pre = self.encode_prompt(lq)
        prompt_embeds = self.clip_projection(prompt_embeds_pre)
        lq_latent = self.vae.encode(lq.to(self.weight_dtype)).latent_dist.sample() * self.vae.config.scaling_factor
        model_pred = self.unet(lq_latent, self.timesteps, encoder_hidden_states=prompt_embeds).sample
        # x_denoised = self.noise_scheduler.step(model_pred, self.timesteps, lq_latent, return_dict=True).prev_sample
        x_denoised = (lq_latent-self.beta_prod_t**(0.5)*model_pred)/self.alpha_prod_t**(0.5) 
        output_image = (self.vae.decode(x_denoised.to(self.weight_dtype) / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        return output_image
    
# Backward compatibility and standard aliases
RCOD_O_gen = OSEDiff_gen
RCOD_O_reg = OSEDiff_reg
RCOD_O_test = OSEDiff_test
RCOD_O_inference_time = OSEDiff_inference_time

