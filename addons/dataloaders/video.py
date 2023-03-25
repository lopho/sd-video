import os
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

class VideoIDFolderAndVideoIDTextPairs(Dataset):
    r"""
    VideoIDFolderAndVideoIDTextPairs Dataset

    Args:
        root_path ([`str`]):
            Path to the folder with the folders
        num_frames ([`int`]):
            Number of frames to slice the folder into. Default 16

    Load data from a folder with the following format:
    ```
    datasets/snailchan/framed/
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
    def __init__(self, root_path: str, 
                 num_frames: int = 16) -> None:
        super().__init__()
        # first check if frames & text folders exist
        frames_folder_path = os.path.join(root_path, 'frames')
        self.frames_folder_path = frames_folder_path
        text_folder_path = os.path.join(root_path, 'text')
        self.text_folder_path = text_folder_path

        if os.path.exists(text_folder_path) is False or os.path.exists(frames_folder_path) is False:
            raise OSError("Frames path or text path does not exists")
        
        # map frames  folder first
        frames_folders_map = os.listdir(frames_folder_path)
        print("frames map:", frames_folders_map)
        #text only contains txt files
        text_folders_map = os.listdir(text_folder_path)
        print("text map:", text_folders_map)

        #check len
        if len(frames_folders_map) != len(text_folders_map):
            raise OSError("Len mismatch")

        #check pairs
        for entry in text_folders_map:
            filename, ext = os.path.splitext(entry)
            if filename not in frames_folders_map:
                raise OSError(f"TXT Pair {str(filename)} not in frames_folders_map")

        self.files = []
            
        for entry in text_folders_map:
            filename, ext = os.path.splitext(entry) # 0001 , txt
            text_path = os.path.join(text_folder_path, entry) # ie. semi/text/0001.txt
            indiv_frame_folder = os.path.join(frames_folder_path, filename) # semi/frames/0001
            print("textpath:", text_path)
            print("frame folder:", indiv_frame_folder)

            tmp_frame_list = []
            tmp_frame_count = 0

            for frame_img_path in sorted(os.listdir(indiv_frame_folder)):
                tmp_frame_list.append(os.path.join(indiv_frame_folder, frame_img_path)) # semi/frames/0001/000001.jpg
                tmp_frame_count = tmp_frame_count + 1
                if tmp_frame_count >= num_frames:
                    self.files.append({
                        "text_path": text_path, #FULL PATH
                        "frame_path_list": tmp_frame_list #FULL PATH
                    }) 
                    tmp_frame_list = []
                    tmp_frame_count = 0

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index) -> dict:
        entry = self.files[index]

        """
        {
            "text_path": path_to_text_file,
            "frame_path_list": [
            "/mnt/drive1/vids/001/0001.png",
            "/mnt/drive1/vids/001/0002.png",
            "/mnt/drive1/vids/001/0003.png",
            ]
        }
        """

        return entry

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    
def VideoIDFolderAndVideoIDTextPairs_collate_fn(
        batch,
        image_size: tuple[int, int] = (256, 256),
        dtype = torch.float32
        ):
        # batch = list o' f h w c???
        
        videos = []
        captions = []

        for s in batch:
            text_path = s['text_path']
            frames_paths = s['frame_path_list']

            text_caption = open(os.path.join(text_path), 'r').read().strip()
            fin_cap = text_caption + ", watermark"
            print("Current sample:", fin_cap)
            captions.append(fin_cap)
            frames_list = []

            for indiv_frame_path in frames_paths:
                img = Image.open(indiv_frame_path)
                img = ImageOps.fit(img.convert('RGB'), image_size, method = Image.Resampling.LANCZOS)
                x = torch.tensor(np.asarray(img))
                frames_list.append(x)

            xs = torch.stack(frames_list)
            videos.append(xs)
        videos = torch.stack(videos) # b f h w c
        videos = videos.permute(0, 1, 4, 2, 3).to(dtype).div(255).mul(2).sub(1).to(memory_format = torch.contiguous_format)
        return { 'pixel_values': videos, 'text': captions }