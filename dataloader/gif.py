
import os
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import numpy as np


class GifSet(Dataset):
    def __init__(self, path: str, cap_ext: str = 'txt') -> None:
        super().__init__()
        self.files = sorted(
                os.path.join(path, f)
                for f in os.listdir(path)
                if f.endswith(f'.gif')
        )
        self.cap_ext = cap_ext

    def __getitem__(self, index) -> tuple[str, str]: # file/folder path, caption path
        f = self.files[index]
        return f, os.path.splitext(f)[0] + '.' + self.cap_ext

    def __len__(self) -> int:
        return len(self.files)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def gif_collate_fn(
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

