
import math
import torch


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

