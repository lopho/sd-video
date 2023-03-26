
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
            lr: 1e-4,
            scale_lr: bool = False,
            lr_warmup: float = 0.,
            lr_decay: float = 1.0,
            epochs: int = 1,
            gradient_accumulation: int = 1,
            unconditional_ratio: float = 0.,
            dynamo: bool = False,
            xformers: bool = True,
            output_dir: str = 'output',
            load_state: str | None = None,
            log_with: str | None = None,
            seed: int = 0,
            preencoded_img: bool = False,
            preencoded_txt: bool = False,
            adam8bit: bool = False,
    ) -> None:
        """
        Expects batches from dataloader in the following format:
        'pixel_values': tensor shape b f c h w
        'text': [str, ...] of length b
        """
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        torch._dynamo.config.cache_size_limit = 256
        torch._dynamo.config.allow_rnn = True

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.output_dir = output_dir
        self.accelerator = Accelerator(
            mixed_precision = 'fp16',
            project_dir = os.path.join(output_dir, 'logs'),
            gradient_accumulation_steps = gradient_accumulation,
            dynamo_backend = 'inductor' if dynamo else None,
            log_with = log_with
        )

        unet = model.unet.train().requires_grad_(True)
        unet.enable_xformers(xformers)

        self.batch_size = dataloader.batch_size
        self.epochs = epochs
        self.total_steps: int = math.ceil(len(dataloader) / (self.accelerator.gradient_accumulation_steps * self.accelerator.num_processes)) * epochs

        if scale_lr:
            lr = lr * self.batch_size * self.accelerator.gradient_accumulation_steps * self.accelerator.num_processes
        optim_cls = torch.optim.AdamW
        if adam8bit:
            try:
                import bitsandbytes as bnb
                optim_cls = bnb.optim.AdamW8bit
            except ImportError:
                tqdm.write('install bitsandbytes to use 8-Bit AdamW')
        optimizer = optim_cls(unet.parameters(), lr = lr)
        scheduler_steps = math.ceil((len(dataloader) * epochs) / (self.accelerator.gradient_accumulation_steps))
        scheduler = lr_dahd_cyclic(
                optimizer,
                delay = 1,
                warmup = math.ceil(lr_warmup * scheduler_steps),
                decay = math.ceil(lr_decay * scheduler_steps)
        )

        self.optimizer: torch.optim.Optimizer = self.accelerator.prepare(optimizer)
        self.scheduler: torch.optim.lr_scheduler.LRScheduler = self.accelerator.prepare_scheduler(scheduler)
        self.dataloader: DataLoader = self.accelerator.prepare(dataloader)
        self.unet: UNetSD = self.accelerator.prepare(unet)
        self.diffusion: GaussianDiffusion = model.diffusion.to(self.accelerator.device)
        if not preencoded_img:
            model.vae.enable_xformers(xformers)
            self.vae: AutoencoderKL = model.vae.to(self.accelerator.device)
        self.preencoded_img = preencoded_img
        if not preencoded_txt:
            self.text_encoder: FrozenOpenCLIPEmbedder = model.text_encoder.to(self.accelerator.device)
        self.preencoded_txt = preencoded_txt

        if load_state is not None:
            self.accelerator.load_state(load_state)

        if unconditional_ratio > 0:
            with torch.no_grad(), self.accelerator.autocast():
                self.t_emb_uncond = model.text_encoder([''] * dataloader.batch_size).to(
                        dtype = torch.float32,
                        device = self.accelerator.device,
                        memory_format = torch.contiguous_format
                )
        self.unconditional_ratio = unconditional_ratio

    @torch.no_grad()
    def train(self, run_name: str = 'vid', log_every: int = 10, save_every: int = 100, verbose: bool = True):
        loss = torch.tensor(0., device = self.accelerator.device)
        track_loss: list[float] = []
        track_lr: list[float] = []
        update_step: int = 0
        samples_seen: int = 0
        samples_per_update = (
                self.batch_size * 
                self.accelerator.gradient_accumulation_steps *
                self.accelerator.num_processes
        )
        if self.accelerator.is_main_process:
            os.makedirs(self.output_dir, exist_ok = True)
            self.accelerator.init_trackers(run_name)
            pbar = tqdm(total = self.total_steps, dynamic_ncols = True, smoothing = 0.01)
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        self.accelerator.wait_for_everyone()
        for epoch in range(self.epochs):
            for b in self.dataloader:
                loss = loss + self.step(b) / self.accelerator.gradient_accumulation_steps
                if self.accelerator.sync_gradients:
                    mean_loss = self.accelerator.gather(loss).mean()
                    loss.zero_()
                    update_step += 1
                    if self.accelerator.is_main_process:
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
                            self.accelerator.log(stats, step = update_step)
                            track_loss.clear()
                            track_lr.clear()
                        pbar.update(1)
                    if update_step % save_every == 0 or update_step == self.total_steps:
                        self.accelerator.save_state(os.path.join(
                                self.output_dir,
                                f'{run_name}_{str(update_step).zfill(len(str(self.total_steps)))}'
                        ))
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            net = self.accelerator.unwrap_model(self.unet, False).cpu()
            torch.save(net.state_dict(), os.path.join(self.output_dir, f'{run_name}_unet.pt'))
        self.accelerator.end_training()

    @torch.no_grad()
    def prepare_txt(self, text: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.unconditional_ratio:
            t_emb = self.t_emb_uncond
        else:
            with self.accelerator.autocast():
                t_emb = self.text_encoder(text).to(dtype = torch.float32)
        return t_emb

    @torch.no_grad()
    def prepare_img(self, frames: torch.Tensor) -> torch.Tensor:
        bs = len(frames)
        frames = rearrange(frames, 'b f c h w -> (b f) c h w')
        with self.accelerator.autocast():
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
                device = self.accelerator.device, dtype = torch.int64
        )

        noise = torch.randn_like(x0, memory_format = torch.contiguous_format)

        x_noisy = self.diffusion.q_sample(
                x_start = x0,
                t = t,
                noise = noise
        ).to(memory_format = torch.contiguous_format)

        with torch.enable_grad(), self.accelerator.accumulate(self.unet):
            with self.accelerator.autocast():
                y = self.unet(x_noisy, t, t_emb).to(dtype = torch.float32)
            loss = torch.nn.functional.mse_loss(y, noise)
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.unet.parameters(), 1.0)
            self.optimizer.step()
            if not self.accelerator.optimizer_step_was_skipped and self.accelerator.sync_gradients:
                self.scheduler.step()
            self.optimizer.zero_grad(set_to_none = True)
        return loss.detach()

