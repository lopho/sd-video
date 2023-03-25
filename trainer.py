
import os
import random
import math
import gc
import numpy as np
import torch
from tqdm.auto import tqdm

import torch._dynamo
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset, Sampler
from einops import rearrange

from sd_video import SDVideo
from diffusion import GaussianDiffusion
from unet_sd import UNetSD
from autoencoder import AutoencoderKL
from clip_embedder import FrozenOpenCLIPEmbedder

from PIL import Image, ImageOps


class MiniTestSet(Dataset):
    def __init__(self, path: str, cap_ext: str = 'txt') -> None:
        super().__init__()
        self.files = sorted(
                os.path.join(path, f)
                for f in os.listdir(path)
                if not f.endswith(f'.{cap_ext}')
        )
        self.cap_ext = cap_ext

    def __getitem__(self, index) -> tuple[str, str]: # file/folder path, caption path
        f = self.files[index]
        return f, f'{f}.{self.cap_ext}'

    def __len__(self) -> int:
        return len(self.files)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

def collate_fn(
        batch: list[tuple[str, str]],
        num_frames: int = 16,
        image_size: tuple[int, int] = (256, 256),
        dtype = torch.float32
) -> dict[str, torch.Tensor | str]:
    captions: list[str] = []
    videos: list[torch.Tensor] = []
    for s in batch:
        with open(s[1], 'r') as f:
            captions.append(f.read().strip())
        # just assume its a gif for now
        im = Image.open(s[0])
        im.load()
        frames: list[torch.Tensor] = []
        while len(frames) < num_frames:
            # ping pong loop if too short
            dir = 1
            try:
                imf = ImageOps.fit(im.convert('RGB'), image_size, method = Image.Resampling.LANCZOS)
                x = torch.tensor(np.asarray(imf))
                frames.append(x)
                im.seek(im.tell() + dir)
            except EOFError:
                dir = -1
                im.seek(im.tell() + dir)
        im.close()
        xs = torch.stack(frames) # f h w c
        videos.append(xs)
    videos = torch.stack(videos) # b f h w c
    videos = videos.permute(0, 1, 4, 2, 3).to(dtype).div(255).mul(2).sub(1).to(memory_format = torch.contiguous_format)
    return { 'pixel_values': videos, 'text': captions }



def lr_dahd_cyclic(
        optimizer: torch.optim.Optimizer,
        warmup: int = 0,
        delay: int = 0,
        attack: int = 0,
        hold: int = 0,
        decay: int = 0,
        min_lr: float = 0.0,
        max_lr: float = 1.0,
        time_scale: float = 1.0,
        last_step: int = -1
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Delay-Attack-Hold-Decay signal envelope
            |const| cos |const| cos |const| cos   ...
    max_lr  |           A=====V                 A ...
            |        A           V           A
    min_lr  |=====A                 V=====A
            |delay|attack|hold|decay|delay|attack ...
            | -"- |warmup| ---""--- | ------""--- ...
            |------- cycle 1 -------|---- cycle 2 ...
    constant lr (max_lr):
        hold = 0 or 1, delay = 0, attack = 0, decay = 0
    constant lr (min_lr):
        delay = 1, attack = 0, hold = 0, decay = 0
    constant lr ((max_lr + min_lr) / 2):
        attack or decay = 1, delay = 0, hold = 0, attack or decay = 0
    constant with warmup (from 0 to max_lr):
        warmup > 0
    cosine lr (cyclic if attack + decay < total steps):
        attack == decay, hold = 0, delay = 0
    cosine annealing (with hard restarts if decay < total steps):
        decay > 0, delay = 0, attack = 0, hold = 0
    cosine annealing with warmup:
        warmup > 0, decay = total - warmup, delay = 0, attack = Any, hold = 0
    """
    assert warmup >= 0
    assert delay >= 0
    assert attack >= 0
    assert hold >= 0
    assert decay >= 0
    if warmup == 0:
        warmup = attack
    cycle_length = delay + attack + hold + decay
    scale_lr = max_lr - min_lr
    def lr_lambda(current_step: int) -> float:
        if cycle_length == 0:
            return max_lr
        current_step = round(current_step * time_scale)
        in_warmup = current_step < (delay + warmup)
        scaled_attack = warmup if in_warmup else attack
        if not in_warmup:
            current_step = current_step - warmup + attack
            current_step = current_step % cycle_length
        if current_step < delay:
            return min_lr
        elif current_step < delay + scaled_attack:
            progress = (current_step - delay) / scaled_attack
            if in_warmup:
                return max(0.0, 1.0 - 0.5 * (1.0 + math.cos(math.pi * progress))) * max_lr
            else:
                return max(0.0, 1.0 - 0.5 * (1.0 + math.cos(math.pi * progress))) * scale_lr + min_lr
        elif current_step < delay + scaled_attack + hold:
            return max_lr
        else:
            progress = (current_step - (delay + scaled_attack + hold)) / decay
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress))) * scale_lr + min_lr
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_step)


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
        model.vae.enable_xformers(xformers)

        self.batch_size = dataloader.batch_size
        self.epochs = epochs
        self.total_steps: int = math.ceil(len(dataloader) / (self.accelerator.gradient_accumulation_steps * self.accelerator.num_processes)) * epochs

        if scale_lr:
            lr = lr * self.batch_size * self.accelerator.gradient_accumulation_steps * self.accelerator.num_processes
        optimizer = torch.optim.AdamW(unet.parameters(), lr = lr)
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
        self.vae: AutoencoderKL = model.vae.to(self.accelerator.device)
        self.text_encoder: FrozenOpenCLIPEmbedder = model.text_encoder.to(self.accelerator.device)

        if load_state is not None:
            self.accelerator.load_state(load_state)

        if unconditional_ratio > 0:
            with torch.no_grad(), self.accelerator.autocast():
                self.t_emb_uncond = self.text_encoder([''] * dataloader.batch_size).to(dtype = torch.float32)
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
    def step(self, batch: dict[str, torch.Tensor | str]) -> torch.Tensor:
        frames = batch['pixel_values'].to(device = self.accelerator.device) # b f c h w
        text = batch['text']

        if random.random() < self.unconditional_ratio:
            t_emb = self.t_emb_uncond
        else:
            with self.accelerator.autocast():
                t_emb = self.text_encoder(text).to(dtype = torch.float32)

        bs = len(frames)
        frames = rearrange(frames, 'b f c h w -> (b f) c h w')
        with self.accelerator.autocast():
            x0 = self.vae.encode(frames).sample().to(dtype = torch.float32) * 0.18215
        x0 = rearrange(x0, '(b f) c h w -> b c f h w', b = bs)

        t = torch.randint(
                0, self.diffusion.num_timesteps, (bs,),
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

