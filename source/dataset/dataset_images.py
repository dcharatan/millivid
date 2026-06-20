import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor

from .dataset import ConfigurableDataset, Split
from .format_blobs import load_blobs
from .video_index import VideoIndex, split_index


@dataclass(frozen=True)
class DatasetImagesCfg:
    name: Literal["images"]
    path: tuple[Path, ...]
    vis_frames_per_clip: int
    test_frames_per_clip: int


class DatasetImages(ConfigurableDataset[DatasetImagesCfg]):
    index: dict[str, int]
    video_index: VideoIndex

    def __init__(self, cfg: DatasetImagesCfg, split: Split):
        super().__init__(cfg, split)
        with (self.path / "index.json").open("r") as f:
            self.index = json.load(f)

        # Split videos into clips.
        self.video_index = VideoIndex(
            split_index(self.index, split),
            self.frames_per_clip,
        )

    def __len__(self) -> int:
        return len(self.video_index)

    def decode_image(self, raw_image: bytes) -> Float[Tensor, "rgb=3 height width"]:
        image = Image.open(BytesIO(raw_image))
        image = np.transpose(np.array(image, dtype=np.float32) / 255, (2, 0, 1))
        return torch.tensor(image, dtype=torch.bfloat16)

    def load_frames(
        self,
        key: str,
        frame_start: int,
        frame_end: int,
    ) -> Float[Tensor, "frame rgb=3 height width"]:
        """This is used during evaluation."""
        assert frame_start < frame_end
        frames = load_blobs(
            (self.path / key).with_suffix(".frames"),
            self.index[key],
            frame_start,
            frame_end,
        )
        frames = [self.decode_image(frame) for frame in frames]
        return torch.stack(frames)

    def __getitem__(self, index: int):
        # Determine which video to load.
        key, frame_index = self.video_index[index]
        f = self.index[key] if self.frames_per_clip is None else self.frames_per_clip
        assert frame_index + f <= self.index[key]

        # Load and decode the video.
        frames = load_blobs(
            (self.path / key).with_suffix(".frames"),
            self.index[key],
            frame_index,
            frame_index + f,
        )
        frames = torch.stack([self.decode_image(frame) for frame in frames])

        return {"frames": frames, "key": key}

    @property
    def path(self) -> Path:
        for path in self.cfg.path:
            if path.exists():
                return path
        raise FileNotFoundError("Could not find a valid dataset path!")

    @property
    def frames_per_clip(self) -> int | None:
        return {
            "train": 1,
            "vis": self.cfg.vis_frames_per_clip,
            "test": self.cfg.test_frames_per_clip,
            "all": None,  # no limit
        }[self.split]
