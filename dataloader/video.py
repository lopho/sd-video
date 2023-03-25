
import os
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


class FrameSequenceDataset(Dataset):
    """
    Frame Sequence Dataset

    Args:
        root_path ([`str`]):
            Path to the directory with frame and caption data
        num_frames ([`int`]):
            Number of frames per sample. Defaults to 16

    Load data from a directory in the following layout:
    ```
    root_path
    ├── frames
    │   ├── 1024544732
    │   │   ├── 000001.png
    │   │   ├── ...
    │   │   └── 000432.png
    │   └── 1024595066
    │       ├── 000001.png
    │       ├── ...
    │       └── 000252.png
    └── text
        ├── 1024544732.txt
        └── 1024595066.txt
    ```
    """
    def __init__(self, root_path: str, num_frames: int = 16) -> None:
        super().__init__()

        # check for valid data
        frames_folder_path = os.path.join(root_path, 'frames')
        self.frames_folder_path = frames_folder_path
        text_folder_path = os.path.join(root_path, 'text')
        self.text_folder_path = text_folder_path

        if not os.path.exists(text_folder_path):
            raise OSError("text directory not found")
        if not os.path.exists(frames_folder_path):
            raise OSError("frames directory not found")

        frames_folders_map = os.listdir(frames_folder_path)
        text_folders_map = os.listdir(text_folder_path)

        for entry in text_folders_map:
            filename, _ = os.path.splitext(entry)
            if filename not in frames_folders_map:
                raise OSError(f"no video data found for caption {str(filename)}")

        # build video <-> text pairs
        self.files: list[dict[str, list[str]]] = []
        for entry in text_folders_map:
            filename = os.path.splitext(entry)[0] # 0001
            text_path = os.path.join(text_folder_path, entry) # ie. root_path/text/0001.txt
            frames_path = os.path.join(frames_folder_path, filename) # root_path/frames/0001

            frame_list = []
            frame_count = 0

            for frame_img_path in sorted(os.listdir(frames_path)):
                frame_list.append(os.path.join(frames_path, frame_img_path)) # root_path/frames/0001/000001.jpg
                frame_count = frame_count + 1
                if frame_count >= num_frames:
                    self.files.append({
                        "text_path": text_path,
                        "frame_path_list": frame_list
                    })
                    frame_list = []
                    frame_count = 0

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, list[str]]:
        """
        {
            "text_path": 001.txt,
            "frame_path_list": [
                "/mnt/drive1/vids/001/0001.png",
                "/mnt/drive1/vids/001/0002.png",
                "/mnt/drive1/vids/001/0003.png",
            ]
        }
        """
        return self.files[index]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def frame_sequence_collate_fn(
        batch,
        image_size: tuple[int, int] = (256, 256),
        dtype = torch.float32
) -> tuple[torch.Tensor, list[str]]:
    videos = []
    captions = []

    for s in batch:
        text_path = s['text_path']
        frames_paths = s['frame_path_list']

        with open(os.path.join(text_path), 'r') as f:
            text = f.read().strip()
        text = text + ", watermark"
        captions.append(text)
        frames = []

        for indiv_frame_path in frames_paths:
            img = Image.open(indiv_frame_path)
            img = ImageOps.fit(img.convert('RGB'), image_size, method = Image.Resampling.LANCZOS)
            x = torch.tensor(np.asarray(img))
            frames.append(x)
            img.close()

        xs = torch.stack(frames)
        videos.append(xs)
    videos = torch.stack(videos) # b f h w c
    videos = videos.permute(0, 1, 4, 2, 3).to(dtype).div(255).mul(2).sub(1).to(memory_format = torch.contiguous_format)
    return { 'pixel_values': videos, 'text': captions }

