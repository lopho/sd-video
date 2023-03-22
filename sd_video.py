
import os
import json
from typing import Any

import torch
import numpy as np
from einops import rearrange
from PIL import Image, ImageOps

from unet_sd import UNetSD
from autoencoder import AutoencoderKL
from clip_embedder import FrozenOpenCLIPEmbedder
from diffusion import GaussianDiffusion, beta_schedule

class SDVideo:
    def __init__(self,
            model_path: str,
            device: str | torch.device = torch.device('cpu'),
            dtype: torch.dtype = torch.float32,
            amp: bool = True
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.amp = amp
        with open(os.path.join(model_path, 'configuration.json'), 'r') as f:
            self.config: dict[str, Any] = json.load(f)
        cfg = self.config['model']['model_cfg']
        cfg['temporal_attention'] = True if cfg[
            'temporal_attention'] == 'True' else False
        self.default_frames = self.config['model']['model_args']['max_frames']

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
        self.unet = self.unet.to(dtype).eval().requires_grad_(False)
        self.unet = self.unet.to(device, memory_format = torch.contiguous_format)

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
        self.diffusion.to(dtype = dtype, device = device, memory_format = torch.contiguous_format)

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
        self.vae = self.vae.to(dtype).eval().requires_grad_(False)
        self.vae = self.vae.to(device, memory_format = torch.contiguous_format)

        self.text_encoder: FrozenOpenCLIPEmbedder = FrozenOpenCLIPEmbedder(
                version = os.path.join(model_path, self.config['model']['model_args']['ckpt_clip']),
                layer = 'penultimate'
        )
        self.text_encoder = self.text_encoder.to(dtype).eval().requires_grad_(False)
        self.text_encoder = self.text_encoder.to(device, memory_format = torch.contiguous_format)

    def enable_xformers(self, enable: bool = True) -> None:
        self.unet.enable_xformers(enable)
        self.vae.enable_xformers(enable)

    @torch.inference_mode()
    def __call__(self,
            text: str,
            text_neg: str = '',
            guidance_scale: float = 9.0,
            timesteps: int = 50,
            image_size: tuple[int, int] = (256, 256),
            initial_frames: list[Image.Image] | Image.Image | None = None,
            num_frames: int = None,
            t_start: int = 0,
            bar: bool = False
    ) -> torch.Tensor:
        if initial_frames is not None:
            if isinstance(initial_frames, Image.Image):
                initial_frames = [initial_frames]
            num_frames = len(initial_frames)
        text_emb, text_emb_neg, init_frames = self.preprocess(
                text = text,
                text_neg = text_neg,
                image_size = image_size,
                num_frames = num_frames or self.default_frames,
                batch_size = 1,
                timesteps = timesteps,
                t_start = t_start,
                initial_frames = initial_frames
        )
        y = self.process(
                text_emb = text_emb,
                text_emb_neg = text_emb_neg,
                guidance_scale = guidance_scale,
                timesteps = timesteps,
                init_frames = init_frames,
                t_start = t_start,
                bar = bar
        )
        frames = self.postprocess(y)
        return frames # f c h w

    def preprocess(self,
            text: str,
            image_size: tuple[int, int],
            num_frames: int,
            batch_size: int,
            timesteps: int,
            t_start: int,
            text_neg: str = '',
            initial_frames: list[Image.Image] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text_emb = self.text_encoder(text)
        text_emb_neg = self.text_encoder(text_neg)
        # txt2img
        if initial_frames is None:
            latent_h, latent_w = image_size[1] // 8, image_size[0] // 8
            init_x = torch.randn(
                    batch_size, 4, num_frames, latent_h, latent_w,
                    device = self.device, dtype=self.dtype
            )  # shape: b c f h w
        # img2img
        else:
            fit_h = round(initial_frames[0].size[1] / 64) * 64
            fit_w = round(initial_frames[0].size[0] / 64) * 64
            latent_h, latent_w = fit_h // 8, fit_w // 8
            init_x: torch.Tensor = torch.stack(
                    [ torch.tensor(np.asarray(f)) for f in initial_frames ]
            )
            init_x = init_x.permute(0,3,1,2).to(
                    dtype = self.dtype,
                    device = self.device,
                    memory_format = torch.contiguous_format
            ).div(255).mul(2).sub(1)
            with torch.autocast(self.device.type, enabled = self.amp):
                init_x = self.vae.encode(init_x).mean * 0.18215
            steps = torch.full((num_frames, ), t_start-1, device=self.device)
            init_x = self.diffusion.stochastic_encode(init_x, steps, timesteps)
            # f c h w -> b f c h w -> b c f h w
            init_x = init_x.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1).permute(0,2,1,3,4)

        return text_emb, text_emb_neg, init_x

    def postprocess(self, x: torch.Tensor) -> dict[str, list[np.ndarray]]:
        return denormalize(x, 0.5, 0.5)

    def process(self,
            text_emb: torch.Tensor,
            text_emb_neg: torch.Tensor,
            guidance_scale: float,
            timesteps: int,
            init_frames: torch.Tensor | None = None,
            t_start: int = 0,
            batch_size: int = 1,
            eta: float = 0.0,
            bar: bool = False
    ) -> torch.Tensor:
        context = torch.cat([text_emb_neg, text_emb], dim = 0).to(self.device)
        with torch.autocast(self.device.type, enabled=self.amp):
            x0 = self.diffusion.ddim_sample_loop(
                    noise = init_frames,
                    model = self.unet,
                    model_kwargs = [
                            { 'y': context[1].unsqueeze(0).repeat(batch_size, 1, 1) },
                            { 'y': context[0].unsqueeze(0).repeat(batch_size, 1, 1) }
                    ],
                    guide_scale = guidance_scale,
                    ddim_timesteps = timesteps,
                    t_start = t_start,
                    eta = eta,
                    bar = bar
            )
            scale_factor = 0.18215
            video_data = 1. / scale_factor * x0
            bs_vd = video_data.shape[0]
            video_data = rearrange(video_data, 'b c f h w -> (b f) c h w')
            video_data = self.vae.decode(video_data)
            video_data = rearrange(
                video_data, '(b f) c h w -> b c f h w', b = bs_vd)
        return video_data


def denormalize(
        video: torch.Tensor,
        mean: tuple[float, float, float] | float = (0.5, 0.5, 0.5),
        std: tuple[float, float, float] | float = (0.5, 0.5, 0.5)
) -> torch.Tensor:
    if isinstance(mean, float):
        mean = (mean,) * 3
    if isinstance(std, float):
        std = (std,) * 3
    mean = torch.tensor(mean, device = video.device).reshape(1, -1, 1, 1, 1)  # n c f h w
    std = torch.tensor(std, device = video.device).reshape(1, -1, 1, 1, 1)  # n c f h w
    video = video.mul(std).add(mean)
    return video

def load_sequence(path: str, frames: int = 16, offset: int = 0, stride: int = 1, image_size: tuple[int, int] = (256, 256)):
    ff = os.listdir(path)
    assert len(ff) >= offset + (frames - 1) * stride + 1, "not enough images in input directory"
    ff.sort()
    ff = ff[offset:][:frames * stride:stride]
    images: list[Image.Image] = []
    for f in ff:
        im = Image.open(os.path.join(path, f)).convert('RGB')
        im = ImageOps.fit(im, image_size, method = Image.Resampling.LANCZOS)
        images.append(im)
    return images

def to_images(images: torch.Tensor) -> np.ndarray:
    images = rearrange(images, 'i c f h w -> f h (i w) c')  # f h w c
    images = images.mul(255).round().clamp(0, 255).to(dtype = torch.uint8, device = 'cpu').numpy()
    return images

def save_pngs(images: torch.Tensor, path: str) -> None:
    images = to_images(images)
    os.makedirs(path, exist_ok=True)
    for i,x in enumerate(images):
        Image.fromarray(x).save(os.path.join(path, str(i).zfill(4) + '.png'))

def save_gif(images: torch.Tensor, path: str, duration=2, optimize=False):
    images = to_images(images)
    image_list = [Image.fromarray(x) for x in images]
    duration = duration*1000/len(image_list)
    image_list[0].save(path, save_all=True, append_images=image_list[1:], duration=duration, optimize=optimize, loop=0)
