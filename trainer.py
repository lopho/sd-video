
import os
import random
import math
import gc
import numpy as np
import torch
from tqdm.auto import tqdm

import torch._dynamo
from accelerate import Accelerator
from torch.utils.data import DataLoader
from einops import rearrange

from sd_video import SDVideo
from diffusion import GaussianDiffusion
from unet_sd import UNetSD
from autoencoder import AutoencoderKL
from clip_embedder import FrozenOpenCLIPEmbedder

from scheduler.dahd import lr_dahd_cyclic


class SDVideoTrainer:
    def __init__(self,
            model: SDVideo,
            dataloader: DataLoader,
            lr: float = 1e-4,
            scale_lr: bool = False, # scale lr by batch size * grad acc * gpus
            lr_warmup: float = 0.05,
            lr_decay: float = 0.95, # aka annealing, if warmup + decay < 1.0 -> cyclic schedule
            lr_min: float = 0., # minimum lr
            epochs: int = 1,
            gradient_accumulation: int = 1,
            unconditional_ratio: float = 0., # percentage of unconditional training steps
            dynamo: bool = False, # it's slow for now
            xformers: bool = True, # it's fast(er)
            output_dir: str = 'output',
            load_state: str | None = None,
            log_with: str | None = None, # wandb, aim, tensorboard, comet
            seed: int = 0,
            preencoded_img: bool = False, # sampler returns batches of image latents instead of images
            preencoded_txt: bool = False, # sampler returns batches of text embeddings instead of str list
            adam8bit: bool = False, # lower vram usage
            gradient_checkpointing: bool = False # lower vram usage, minimally slower
    ) -> None:
        """
        training expects batches from dataloader in the following format:
        'pixel_values': tensor shape b f c h w  OR  shape b c f h w  if preencoded_img
        'text': [str, ...] of length b  OR  tensor shape b 77 1024  if preencoded_txt
        """
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.output_dir = output_dir
        self.accel = Accelerator(
            mixed_precision = 'fp16',
            project_dir = os.path.join(output_dir, 'logs'),
            gradient_accumulation_steps = gradient_accumulation,
            #dynamo_backend = 'inductor' if dynamo else None,
            log_with = log_with
        )

        unet = model.unet.train().requires_grad_(True).to(self.accel.device)
        unet.set_use_memory_efficient_attention_xformers(xformers)
        if gradient_checkpointing:
            unet.enable_gradient_checkpointing()
        else:
            unet.disable_gradient_checkpointing()
        if dynamo:
            import logging
            torch._dynamo.config.cache_size_limit = 256
            torch._dynamo.config.log_level = logging.ERROR
            if self.accel.is_main_process:
                tqdm.write('compiling on first step, may take a few minutes ...')
            unet = torch._dynamo.optimize()(unet)

        self.batch_size = dataloader.batch_size
        self.epochs = epochs
        self.total_steps: int = math.ceil(len(dataloader) / (self.accel.gradient_accumulation_steps * self.accel.num_processes)) * epochs

        if scale_lr:
            lr = lr * self.batch_size * self.accel.gradient_accumulation_steps * self.accel.num_processes

        optim_cls = torch.optim.AdamW
        if adam8bit:
            try:
                os.environ['BITSANDBYTES_NOWELCOME'] = '1'
                import bitsandbytes as bnb
                optim_cls = bnb.optim.AdamW8bit
            except ImportError:
                tqdm.write('install bitsandbytes to use 8-Bit AdamW')
        optimizer = optim_cls(unet.parameters(), lr = lr)
        scheduler_steps = math.ceil(len(dataloader) / (self.accel.gradient_accumulation_steps)) * epochs
        scheduler = lr_dahd_cyclic(
                optimizer,
                delay = 1,
                warmup = math.ceil(lr_warmup * scheduler_steps),
                decay = math.ceil(lr_decay * scheduler_steps),
                min_lr = lr_min
        )

        self.optimizer: torch.optim.Optimizer = self.accel.prepare(optimizer)
        self.scheduler: torch.optim.lr_scheduler.LRScheduler = self.accel.prepare_scheduler(scheduler)
        self.dataloader: DataLoader = self.accel.prepare(dataloader)
        self.unet: UNetSD = self.accel.prepare(unet)
        self.diffusion: GaussianDiffusion = model.diffusion.to(self.accel.device)

        if not preencoded_img:
            model.vae.enable_xformers(xformers)
            self.vae: AutoencoderKL = model.vae.to(self.accel.device)
        self.preencoded_img = preencoded_img
        if not preencoded_txt:
            self.text_encoder: FrozenOpenCLIPEmbedder = model.text_encoder.to(self.accel.device)
        self.preencoded_txt = preencoded_txt

        if unconditional_ratio > 0:
            with torch.no_grad(), self.accel.autocast():
                self.t_emb_uncond = model.text_encoder([''] * dataloader.batch_size).to(
                        dtype = torch.float32,
                        device = self.accel.device,
                        memory_format = torch.contiguous_format
                )
        self.unconditional_ratio = unconditional_ratio

        if load_state is not None:
            self.accel.load_state(load_state)

    @torch.no_grad()
    def train(self,
            run_name: str = 'vid',
            log_every: int = 10,
            save_every: int = 100,
            verbose: bool = True
    ) -> None:
        loss = torch.tensor(0., device = self.accel.device)
        track_loss: list[float] = []
        track_lr: list[float] = []
        update_step: int = 0
        samples_seen: int = 0
        samples_per_update = (
                self.batch_size * 
                self.accel.gradient_accumulation_steps *
                self.accel.num_processes
        )
        if self.accel.is_main_process:
            os.makedirs(self.output_dir, exist_ok = True)
            self.accel.init_trackers(run_name)
            pbar = tqdm(total = self.total_steps, dynamic_ncols = True, smoothing = 0)
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        self.accel.wait_for_everyone()
        for epoch in range(self.epochs):
            for b in self.dataloader:
                loss = loss + self.step(b) / self.accel.gradient_accumulation_steps
                if self.accel.sync_gradients:
                    mean_loss = self.accel.gather(loss).mean()
                    loss.zero_()
                    update_step += 1
                    if self.accel.is_main_process:
                        samples_seen += samples_per_update
                        track_loss.append(mean_loss.item())
                        track_lr.append(self.scheduler.get_last_lr()[0])
                        if update_step % log_every == 0 or update_step == self.total_steps:
                            stats = {
                                'loss': sum(track_loss) / len(track_loss),
                                'lr': sum(track_lr) / len(track_lr),
                                'samples': samples_seen,
                                'epoch': epoch
                            }
                            if verbose:
                                tqdm.write(f'{update_step}: {stats}')
                            pbar.set_postfix(stats)
                            self.accel.log(stats, step = update_step)
                            track_loss.clear()
                            track_lr.clear()
                        pbar.update(1)
                    if update_step % save_every == 0 or update_step == self.total_steps:
                        self.accel.save_state(os.path.join(
                                self.output_dir,
                                f'{run_name}_{str(update_step).zfill(len(str(self.total_steps)))}'
                        ))
        self.accel.wait_for_everyone()
        if self.accel.is_main_process:
            net = self.accel.unwrap_model(self.unet, False).cpu()
            torch.save(net.state_dict(), os.path.join(self.output_dir, f'{run_name}_unet.pt'))
        self.accel.end_training()

    @torch.no_grad()
    def prepare_txt(self, text: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.unconditional_ratio:
            t_emb = self.t_emb_uncond
        else:
            with self.accel.autocast():
                t_emb = self.text_encoder(text).to(dtype = torch.float32)
        return t_emb

    @torch.no_grad()
    def prepare_img(self, frames: torch.Tensor) -> torch.Tensor:
        bs = len(frames)
        frames = rearrange(frames, 'b f c h w -> (b f) c h w')
        with self.accel.autocast():
            x0 = self.vae.encode(frames).sample().to(dtype = torch.float32) * 0.18215
        x0 = rearrange(x0, '(b f) c h w -> b c f h w', b = bs)
        return x0

    @torch.no_grad()
    def step(self, batch: dict[str, torch.Tensor | list[str]]) -> torch.Tensor:
        if self.preencoded_img:
            x0 = batch['pixel_values']
        else:
            x0 = self.prepare_img(batch['pixel_values'])
        if self.preencoded_txt:
            t_emb = batch['text'] if random.random() > self.unconditional_ratio else self.t_emb_uncond
        else:
            t_emb = self.prepare_txt(batch['text'])

        t = torch.randint(
                0, self.diffusion.num_timesteps, (len(x0),),
                device = self.accel.device, dtype = torch.int64
        )

        noise = torch.randn_like(x0, memory_format = torch.contiguous_format)

        x_noisy = self.diffusion.q_sample(
                x_start = x0,
                t = t,
                noise = noise
        ).to(memory_format = torch.contiguous_format)

        with torch.enable_grad(), self.accel.accumulate(self.unet):
            with self.accel.autocast():
                y = self.unet(x_noisy, t, t_emb).to(dtype = torch.float32)
            loss = torch.nn.functional.mse_loss(y, noise)
            self.accel.backward(loss)
            if self.accel.sync_gradients:
                self.accel.clip_grad_norm_(self.unet.parameters(), 1.0)
            self.optimizer.step()
            if not self.accel.optimizer_step_was_skipped and self.accel.sync_gradients:
                self.scheduler.step()
            self.optimizer.zero_grad(set_to_none = True)
        return loss.detach()

