
import os
import json
from typing import Any

import torch
import numpy as np
from einops import rearrange
from PIL import Image

from unet_sd import UNetSD
from autoencoder import AutoencoderKL
from clip_embedder import FrozenOpenCLIPEmbedder
from diffusion import GaussianDiffusion, beta_schedule


class SDVideo:
    def __init__(self, model_path: str, device: str | torch.device = torch.device('cpu')):
        self.device = torch.device(device)
        with open(os.path.join(model_path, 'configuration.json'), 'r') as f:
            self.config: dict[str, Any] = json.load(f)
        cfg = self.config['model']['model_cfg']
        cfg['temporal_attention'] = True if cfg[
            'temporal_attention'] == 'True' else False
        self.max_frames = self.config['model']['model_args']['max_frames']

        self.unet: UNetSD = UNetSD(
                in_dim = cfg['unet_in_dim'],
                dim = cfg['unet_dim'],
                y_dim = cfg['unet_y_dim'],
                context_dim = cfg['unet_context_dim'],
                out_dim = cfg['unet_out_dim'],
                dim_mult = cfg['unet_dim_mult'],
                num_heads = cfg['unet_num_heads'],
                head_dim = cfg['unet_head_dim'],
                num_res_blocks = cfg['unet_res_blocks'],
                attn_scales = cfg['unet_attn_scales'],
                dropout = cfg['unet_dropout'],
                temporal_attention = cfg['temporal_attention']
        )
        self.unet.load_state_dict(
                torch.load(os.path.join(model_path, self.config['model']['model_args']['ckpt_unet'])),
                strict = True
        )
        self.unet = self.unet.eval().requires_grad_(False)
        self.unet.to(self.device)

        betas = beta_schedule(
                'linear_sd',
                cfg['num_timesteps'],
                init_beta=0.00085,
                last_beta=0.0120
        )
        self.diffusion = GaussianDiffusion(
                betas = betas,
                mean_type = cfg['mean_type'],
                var_type = cfg['var_type'],
                loss_type = cfg['loss_type'],
                rescale_timesteps = False
        )

        ddconfig = {
                'double_z': True,
                'z_channels': 4,
                'resolution': 256,
                'in_channels': 3,
                'out_ch': 3,
                'ch': 128,
                'ch_mult': [1, 2, 4, 4],
                'num_res_blocks': 2,
                'attn_resolutions': [],
                'dropout': 0.0
        }
        self.vae: AutoencoderKL = AutoencoderKL(
                ddconfig,
                4,
                os.path.join(model_path, self.config['model']['model_args']['ckpt_autoencoder'])
        )
        self.vae = self.vae.eval().requires_grad_(False)
        self.vae.to(self.device)

        self.text_encoder: FrozenOpenCLIPEmbedder = FrozenOpenCLIPEmbedder(
                version = os.path.join(model_path, self.config['model']['model_args']['ckpt_clip']),
                layer = 'penultimate'
        )
        self.text_encoder = self.text_encoder.eval().requires_grad_(False)
        self.text_encoder.to(self.device)

    def __call__(self, text: str, text_neg: str = '') -> list[list[np.ndarray]]:
        text_emb, text_emb_neg = self.preprocess(text, text_neg)
        y = self.process(text_emb, text_emb_neg)
        out = self.postprocess(y)
        return out

    def preprocess(self, text: str, text_neg: str = '') -> tuple[torch.Tensor, torch.Tensor]:
        text_emb = self.text_encoder(text)
        text_emb_neg = self.text_encoder(text_neg)
        return text_emb, text_emb_neg

    def postprocess(self, x: torch.Tensor) -> dict[str, list[np.ndarray]]:
        return tensor2vid(x)

    def process(self, text_emb: torch.Tensor, text_emb_neg: torch.Tensor) -> torch.Tensor:
        context = torch.cat([text_emb_neg, text_emb], dim = 0).to(self.device)
        # synthesis
        with torch.no_grad():
            num_sample = 1  # here let b = 1
            latent_h, latent_w = 32, 32
            with torch.autocast(self.device.type, enabled=True):
                x0 = self.diffusion.ddim_sample_loop(
                        noise = torch.randn(num_sample, 4, self.max_frames, latent_h, latent_w).to(self.device),  # shape: b c f h w
                        model = self.unet,
                        model_kwargs = [{
                            'y': context[1].unsqueeze(0).repeat(num_sample, 1, 1)
                        }, {
                            'y': context[0].unsqueeze(0).repeat(num_sample, 1, 1)
                        }],
                        guide_scale = 9.0,
                        ddim_timesteps = 50,
                        eta = 0.0
                )

                scale_factor = 0.18215
                video_data = 1. / scale_factor * x0
                bs_vd = video_data.shape[0]
                video_data = rearrange(video_data, 'b c f h w -> (b f) c h w')
                self.vae.to(self.device)
                video_data = self.vae.decode(video_data)
                video_data = rearrange(
                    video_data, '(b f) c h w -> b c f h w', b = bs_vd)
        return video_data


def tensor2vid(
        video: torch.Tensor,
        mean: tuple[float, float, float] | float = (0.5, 0.5, 0.5),
        std: tuple[float, float, float] | float = (0.5, 0.5, 0.5)
) -> torch.Tensor:
    if isinstance(mean, float):
        mean = (mean,) * 3
    if isinstance(std, float):
        std = (std,) * 3
    mean = torch.tensor(mean, device = video.device).reshape(1, -1, 1, 1, 1)  # n c f h w
    std = torch.tensor(std, device=video.device).reshape(1, -1, 1, 1, 1)  # n c f h w
    video = video.mul_(std).add_(mean)
    images = rearrange(video, 'i c f h w -> f h (i w) c')  # f h w c
    return images

def save_pngs(images: torch.Tensor, path: str) -> None:
    images = images.mul(255).round().clamp(0, 255).to(dtype = torch.uint8, device = 'cpu').numpy()
    os.makedirs(path, exist_ok=True)
    for i,x in enumerate(images):
        Image.fromarray(x).save(os.path.join(path, str(i).zfill(4) + '.png'))

def save_gif(images: torch.Tensor, path: str, duration=2, optimize=False):
    images = images.mul(255).round().clamp(0, 255).to(dtype = torch.uint8, device = 'cpu').numpy()
    image_list = [Image.fromarray(x) for x in images]
    duration = duration*1000/len(image_list)
    image_list[0].save(path, save_all=True, append_images=image_list[1:], duration=duration, optimize=optimize, loop=0)